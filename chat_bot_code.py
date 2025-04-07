import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values
from PyPDF2 import PdfReader
import chardet
import json

DB_NAME = os.getenv("DB_NAME", "book_reader_db")
DB_USER = os.getenv("DB_USER", "book_reader_user")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

USER_BOOKS_BASE_DIR = "books"
os.makedirs(USER_BOOKS_BASE_DIR, exist_ok=True)

def get_db_connection():
    """
    Устанавливает соединение с базой данных PostgreSQL.
    Возвращает объект соединения или None в случае ошибки.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_client_encoding('UTF8')
        return conn
    except psycopg2.OperationalError as e:
        print(f"ОШИБКА: Не удалось подключиться к базе данных PostgreSQL: {e}")
        print("Проверьте параметры подключения (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT) и статус сервера PostgreSQL.")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка при подключении к БД: {e}")
        return None

def initialize_database():
    """
    Создает необходимые таблицы в базе данных, если они еще не существуют.
    Включает все поля, необходимые для работы бота.
    """
    commands = (
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            preferences JSONB DEFAULT '{}'::jsonb,
            registered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS books (
            book_id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            book_name VARCHAR(512) NOT NULL,
            file_path VARCHAR(1024) NOT NULL,
            file_format VARCHAR(10),
            total_pages INTEGER,
            added_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id, book_name)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_books_user_id ON books (user_id);
        """,
        """
        CREATE TABLE IF NOT EXISTS reading_state (
            state_id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            book_id INTEGER NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
            current_page INTEGER NOT NULL DEFAULT 0,
            last_read_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            is_session BOOLEAN DEFAULT FALSE,
            is_opened BOOLEAN DEFAULT FALSE,
            update_page INTEGER DEFAULT 0,
            book_context TEXT,
            chat_history JSONB DEFAULT '[]'::jsonb,
            UNIQUE (user_id, book_id)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_reading_state_user_book ON reading_state (user_id, book_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_reading_state_user_session ON reading_state (user_id, is_session);
        """,
        """
        CREATE TABLE IF NOT EXISTS recommendation_history (
            history_id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            recommendation_text TEXT NOT NULL,
            generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_recommendation_history_user ON recommendation_history (user_id);
        """
    )
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("Не удалось подключиться к БД для инициализации.")
            return False

        with conn.cursor() as cur:
            print("Инициализация/проверка таблиц базы данных...")
            for command in commands:
                cur.execute(command)
            conn.commit()
            print("Инициализация базы данных успешно завершена.")
            return True
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при инициализации базы данных: {error}")
        if conn: conn.rollback()
        return False
    finally:
        if conn: conn.close()

def _ensure_user_exists(cursor, user_id: int):
    """Проверяет, существует ли пользователь, и создает его при необходимости."""
    cursor.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING;", (user_id,))

def _get_book_info(cursor, user_id: int, book_name: str) -> dict | None:
    """Получает информацию о книге (включая book_id и другие поля) по user_id и book_name."""
    cursor.execute(
        "SELECT book_id, file_path, total_pages, file_format FROM books WHERE user_id = %s AND book_name = %s;",
        (user_id, book_name)
    )
    return cursor.fetchone()

def get_book_id_by_name(user_id: int, book_name: str) -> int | None:
    """Получает ID книги по user_id и book_name."""
    conn = None
    book_id = None
    try:
        conn = get_db_connection()
        if conn is None: return None
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
             book_info = _get_book_info(cur, user_id, book_name)
             if book_info:
                 book_id = book_info['book_id']
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при получении book_id для '{book_name}', user {user_id}: {error}")
    finally:
        if conn: conn.close()
    return book_id

def _calculate_total_pages(full_file_path: str, file_format: str, page_size_txt: int = 1000) -> int | None:
    """
    Вычисляет общее количество "страниц".
    Для PDF - реальные страницы, для TXT - блоки по page_size_txt байт.
    Возвращает число страниц или None при ошибке/пустом файле.
    """
    if not os.path.exists(full_file_path) or os.path.getsize(full_file_path) == 0:
        return 0

    try:
        if file_format == 'pdf':
            with open(full_file_path, "rb") as book_file:
                reader = PdfReader(book_file)
                try:
                    num_pages = len(reader.pages)
                    return num_pages
                except Exception as pdf_err:
                     print(f"Ошибка при доступе к страницам PDF '{os.path.basename(full_file_path)}': {pdf_err}. Невозможно рассчитать страницы.")
                     return None
        elif file_format == 'txt':
            total_size = os.path.getsize(full_file_path)
            return (total_size + page_size_txt - 1) // page_size_txt
        else:
            print(f"Подсчет страниц для формата '{file_format}' не поддерживается.")
            return None
    except Exception as e:
        print(f"Ошибка при подсчете страниц файла '{os.path.basename(full_file_path)}': {e}")
        return None

def add_book_to_db(user_id: int, book_name: str, full_file_path: str, page_size_txt: int = 1000) -> dict | None:
    """
    Добавляет информацию о книге в базу данных после ее физической загрузки/сохранения.
    Вычисляет total_pages, определяет формат.
    Возвращает информацию о добавленной книге (словарь) или None в случае ошибки.
    """
    conn = None
    book_info = None

    if not os.path.exists(full_file_path):
        print(f"Ошибка: Файл не найден по пути для добавления в БД: {full_file_path}")
        return None

    file_format = book_name.split('.')[-1].lower()
    if file_format not in ['txt', 'pdf']:
        print(f"Ошибка: Неподдерживаемый формат файла: {file_format}")
        return None

    total_pages = _calculate_total_pages(full_file_path, file_format, page_size_txt)

    relative_path = os.path.join(str(user_id), os.path.basename(full_file_path))

    try:
        conn = get_db_connection()
        if conn is None: return None

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            _ensure_user_exists(cur, user_id)

            query = sql.SQL("""
                 INSERT INTO books (user_id, book_name, file_path, file_format, total_pages)
                 VALUES (%s, %s, %s, %s, %s)
                 ON CONFLICT (user_id, book_name) DO UPDATE
                 SET file_path = EXCLUDED.file_path,
                     file_format = EXCLUDED.file_format,
                     total_pages = EXCLUDED.total_pages,
                     added_date = CURRENT_TIMESTAMP
                 RETURNING book_id, user_id, book_name, file_path, total_pages, file_format, added_date;
             """)
            cur.execute(query, (user_id, book_name, relative_path, file_format, total_pages))
            book_info = cur.fetchone()
            conn.commit()

            tp_str = f"{book_info['total_pages']} стр." if book_info['total_pages'] is not None else "N/A стр."
            print(f"Книга '{book_info['book_name']}' ({tp_str}, {book_info['file_format']}) добавлена/обновлена для user {user_id}.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при добавлении/обновлении книги в БД: {error}")
        if conn: conn.rollback()
        book_info = None
    finally:
        if conn: conn.close()
    return book_info

def get_user_books(user_id: int) -> list[dict]:
    """
    Получает список книг, добавленных пользователем.
    Возвращает список словарей с book_id, book_name, total_pages, file_format.
    """
    conn = None
    books = []
    try:
        conn = get_db_connection()
        if conn is None: return []
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT book_id, book_name, total_pages, file_format
                FROM books
                WHERE user_id = %s
                ORDER BY book_name ASC;
                """,
                (user_id,)
            )
            books = cur.fetchall()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при получении списка книг пользователя {user_id}: {error}")
    finally:
        if conn: conn.close()
    return books

def update_reading_state(user_id: int, book_id: int, page: int) -> dict | None:
    """
    Обновляет ТЕКУЩУЮ СТРАНИЦУ чтения пользователя в БД.
    Возвращает обновленное состояние {'page': X, 'total_pages': Y} или None при ошибке.
    """
    conn = None
    updated_state = None
    if page < 0: page = 0

    try:
        conn = get_db_connection()
        if conn is None: return None

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT total_pages FROM books WHERE book_id = %s AND user_id = %s;", (book_id, user_id))
            book_res = cur.fetchone()
            if not book_res:
                 print(f"Книга с book_id={book_id} не найдена для пользователя {user_id} при обновлении состояния.")
                 return None
            total_pages = book_res['total_pages']

            if total_pages is not None and page >= total_pages:
                page = total_pages - 1 if total_pages > 0 else 0

            query = sql.SQL("""
                INSERT INTO reading_state (user_id, book_id, current_page, last_read_date)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, book_id) DO UPDATE
                SET current_page = EXCLUDED.current_page,
                    last_read_date = EXCLUDED.last_read_date
                RETURNING current_page;
            """)
            cur.execute(query, (user_id, book_id, page))
            result = cur.fetchone()
            updated_page = result['current_page'] if result else page

            conn.commit()
            updated_state = {"page": updated_page, "total_pages": total_pages}

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при обновлении состояния чтения (page): {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return updated_state

def get_current_page(user_id: int, book_id: int) -> int | None:
    """
    Получает текущую страницу чтения для пользователя и книги из БД.
    Возвращает номер страницы (int, начиная с 0) или 0, если состояние не найдено,
    или None, если произошла ошибка БД.
    """
    conn = None
    current_page = 0
    try:
        conn = get_db_connection()
        if conn is None: return None

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT current_page FROM reading_state WHERE user_id = %s AND book_id = %s;",
                (user_id, book_id)
            )
            result = cur.fetchone()
            if result:
                current_page = result['current_page']

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при получении текущей страницы для user {user_id}, book {book_id}: {error}")
        return None
    finally:
        if conn: conn.close()
    return current_page

def set_book_session_status(user_id: int, book_id: int, status: bool) -> bool:
    """Устанавливает статус is_session для конкретной книги пользователя."""
    conn = None
    success = False
    try:
        conn = get_db_connection()
        if conn is None: return False
        with conn.cursor() as cur:
            cur.execute(
                 """
                 INSERT INTO reading_state (user_id, book_id, current_page)
                 VALUES (%s, %s, 0) ON CONFLICT (user_id, book_id) DO NOTHING;
                 """, (user_id, book_id)
            )
            cur.execute(
                "UPDATE reading_state SET is_session = %s WHERE user_id = %s AND book_id = %s;",
                (status, user_id, book_id)
            )
            conn.commit()
            success = cur.rowcount > 0

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при установке статуса сессии книги (user {user_id}, book {book_id}): {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return success

def clear_all_book_sessions(user_id: int) -> bool:
    """Сбрасывает is_session = FALSE для всех книг пользователя."""
    conn = None
    success = False
    try:
        conn = get_db_connection()
        if conn is None: return False
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reading_state SET is_session = FALSE WHERE user_id = %s AND is_session = TRUE;",
                (user_id,)
            )
            conn.commit()
            print(f"Сброшено {cur.rowcount} активных сессий для пользователя {user_id}.")
            success = True
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при сбросе всех сессий книг пользователя {user_id}: {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return success

def set_book_opened_status(user_id: int, book_id: int, status: bool) -> bool:
    """Устанавливает статус is_opened для конкретной книги пользователя."""
    conn = None
    success = False
    try:
        conn = get_db_connection()
        if conn is None: return False
        with conn.cursor() as cur:
             cur.execute(
                 """
                 INSERT INTO reading_state (user_id, book_id, current_page)
                 VALUES (%s, %s, 0) ON CONFLICT (user_id, book_id) DO NOTHING;
                 """, (user_id, book_id)
             )
             cur.execute(
                "UPDATE reading_state SET is_opened = %s WHERE user_id = %s AND book_id = %s;",
                (status, user_id, book_id)
            )
             conn.commit()
             success = cur.rowcount > 0

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при установке статуса is_opened книги (user {user_id}, book {book_id}): {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return success

def get_reading_state_details(user_id: int, book_id: int) -> dict | None:
    """
    Получает расширенное состояние чтения, включая AI-поля.
    Возвращает словарь или None, если состояние не найдено или ошибка.
    """
    conn = None
    details = None
    try:
        conn = get_db_connection()
        if conn is None: return None
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT current_page, is_session, is_opened, update_page, book_context, chat_history
                FROM reading_state
                WHERE user_id = %s AND book_id = %s;
                """,
                (user_id, book_id)
            )
            details = cur.fetchone()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при получении деталей состояния чтения для user {user_id}, book {book_id}: {error}")
        details = None
    finally:
        if conn: conn.close()
    return details

def update_ai_context(user_id: int, book_id: int, update_page: int, book_context: str | None) -> bool:
    """Обновляет update_page и book_context для состояния чтения."""
    conn = None
    success = False
    try:
        conn = get_db_connection()
        if conn is None: return False
        with conn.cursor() as cur:
             cur.execute(
                 """
                 INSERT INTO reading_state (user_id, book_id, current_page)
                 VALUES (%s, %s, 0) ON CONFLICT (user_id, book_id) DO NOTHING;
                 """, (user_id, book_id)
             )
             cur.execute(
                """
                UPDATE reading_state SET update_page = %s, book_context = %s
                WHERE user_id = %s AND book_id = %s;
                """, (update_page, book_context, user_id, book_id)
            )
             conn.commit()
             success = cur.rowcount > 0

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при обновлении AI контекста (user {user_id}, book {book_id}): {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return success

def add_chat_message(user_id: int, book_id: int, role: str, content: str) -> bool:
    """Добавляет сообщение в историю чата (JSONB массив)."""
    conn = None
    success = False
    message_dict = {"role": role, "content": content}
    try:
        conn = get_db_connection()
        if conn is None: return False
        with conn.cursor() as cur:
            cur.execute(
                 """
                 INSERT INTO reading_state (user_id, book_id, current_page)
                 VALUES (%s, %s, 0) ON CONFLICT (user_id, book_id) DO NOTHING;
                 """, (user_id, book_id)
            )
            cur.execute(
                """
                UPDATE reading_state
                SET chat_history = COALESCE(chat_history, '[]'::jsonb) || %s::jsonb
                WHERE user_id = %s AND book_id = %s;
                """,
                (json.dumps(message_dict), user_id, book_id)
            )
            conn.commit()
            success = cur.rowcount > 0

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при добавлении сообщения в историю чата (user {user_id}, book {book_id}): {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return success

def clear_chat_history(user_id: int, book_id: int) -> bool:
    """Очищает историю чата, устанавливая chat_history в '[]'."""
    conn = None
    success = False
    try:
        conn = get_db_connection()
        if conn is None: return False
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reading_state SET chat_history = '[]'::jsonb WHERE user_id = %s AND book_id = %s;",
                (user_id, book_id)
            )
            conn.commit()
            success = True

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при очистке истории чата (user {user_id}, book {book_id}): {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return success

def get_book_page_content(user_id: int, book_id: int, page: int, page_size_txt: int = 1000) -> str | None:
    """
    Получает текст определенной страницы (для PDF) или блока (для TXT).
    Возвращает текст страницы/блока, сообщение об ошибке/конце книги, или None при ошибке БД.
    """
    conn = None
    content = None
    if page < 0: page = 0

    try:
        conn = get_db_connection()
        if conn is None: return "Ошибка: Не удалось подключиться к БД для получения страницы."

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT file_path, file_format, total_pages FROM books WHERE user_id = %s AND book_id = %s;",
                (user_id, book_id)
            )
            book_info = cur.fetchone()

            if not book_info:
                return f"Книга (ID: {book_id}) не найдена в базе данных для пользователя {user_id}."

            relative_path = book_info['file_path']
            file_format = book_info['file_format']
            total_pages = book_info['total_pages']
            full_path = os.path.join(USER_BOOKS_BASE_DIR, relative_path)

            if not os.path.exists(full_path):
                return f"Ошибка: Файл книги не найден по пути: {full_path}"

            if total_pages is not None and page >= total_pages:
                 return "Вы достигли конца книги."

            if file_format == 'pdf':
                try:
                    with open(full_path, "rb") as book_file:
                        reader = PdfReader(book_file)
                        if page < len(reader.pages):
                            pdf_page = reader.pages[page]
                            extracted_text = pdf_page.extract_text()
                            content = extracted_text if extracted_text else f"(Страница {page + 1} пуста или текст не извлечен)"
                        else:
                             content = "Вы достигли конца книги (PDF)."
                except Exception as e:
                    content = f"Ошибка чтения PDF файла '{os.path.basename(full_path)}': {str(e)}"

            elif file_format == 'txt':
                try:
                    detected_encoding = 'utf-8'
                    try:
                        with open(full_path, "rb") as f:
                            sample = f.read(max(page_size_txt * 2, 10240))
                            result = chardet.detect(sample)
                            if result and result['encoding'] and result['confidence'] > 0.6:
                                detected_encoding = result['encoding']
                    except Exception: pass

                    with open(full_path, "r", encoding=detected_encoding, errors='replace') as book_file:
                        start_pos = page * page_size_txt
                        book_file.seek(start_pos)
                        read_content = book_file.read(page_size_txt)

                        if not read_content:
                            current_pos = book_file.tell()
                            book_file.seek(0, os.SEEK_END)
                            end_pos = book_file.tell()
                            if start_pos >= end_pos and end_pos > 0:
                                content = "Вы достигли конца книги."
                            else:
                                content = read_content
                        else:
                            content = read_content
                except FileNotFoundError:
                     content = f"Ошибка: Файл книги (TXT) не найден: {full_path}"
                except Exception as e:
                    content = f"Ошибка чтения книги TXT '{os.path.basename(full_path)}': {str(e)}"
            else:
                 content = f"Чтение постранично для формата '{file_format}' не поддерживается."

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при получении страницы книги (user {user_id}, book {book_id}): {error}")
        content = f"Ошибка базы данных при получении информации о книге: {error}"
    finally:
        if conn: conn.close()

    return content if content is not None else "Не удалось получить содержимое страницы."

def get_book_full_text(user_id: int, book_id: int) -> str | None:
    """
    Получает ПОЛНЫЙ текст книги.
    Возвращает полный текст, сообщение об ошибке или None при ошибке БД.
    """
    conn = None
    full_text = None

    try:
        conn = get_db_connection()
        if conn is None: return "Ошибка: Не удалось подключиться к БД для получения полного текста."

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT file_path, file_format FROM books WHERE user_id = %s AND book_id = %s;",
                (user_id, book_id)
            )
            book_info = cur.fetchone()

            if not book_info:
                return f"Книга (ID: {book_id}) не найдена в базе данных для пользователя {user_id}."

            relative_path = book_info['file_path']
            file_format = book_info['file_format']
            full_path = os.path.join(USER_BOOKS_BASE_DIR, relative_path)

            if not os.path.exists(full_path):
                return f"Ошибка: Файл книги не найден по пути: {full_path}"

            print(f"Чтение полного текста из файла: {full_path} (формат: {file_format})")

            if file_format == 'pdf':
                try:
                    text_parts = []
                    with open(full_path, 'rb') as file:
                        pdf_reader = PdfReader(file)
                        for page_obj in pdf_reader.pages:
                            try: page_text = page_obj.extract_text()
                            except Exception: page_text = ""
                            text_parts.append(page_text if page_text else "")
                    full_text = "\n\n".join(text_parts)
                except Exception as e:
                    full_text = f"Ошибка при чтении PDF файла '{os.path.basename(full_path)}': {e}"

            elif file_format == 'txt':
                try:
                    detected_encoding = 'utf-8'
                    try:
                        with open(full_path, "rb") as f:
                            sample = f.read(20480)
                            result = chardet.detect(sample)
                            if result and result['encoding'] and result['confidence'] > 0.6:
                                detected_encoding = result['encoding']
                    except Exception: pass

                    with open(full_path, 'r', encoding=detected_encoding, errors='replace') as file:
                        full_text = file.read()
                except Exception as e:
                    full_text = f"Ошибка при чтении текстового файла '{os.path.basename(full_path)}': {e}"
            else:
                full_text = f"Получение полного текста для формата '{file_format}' не поддерживается."

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при получении полного текста книги (user {user_id}, book {book_id}): {error}")
        full_text = f"Ошибка базы данных при получении информации о книге: {error}"
    finally:
        if conn: conn.close()

    return full_text

def save_recommendation_history(user_id: int, recommendations: list[str]):
    """Сохраняет список текстовых рекомендаций в историю пользователя в БД."""
    if not recommendations: return

    conn = None
    try:
        conn = get_db_connection()
        if conn is None: return

        with conn.cursor() as cur:
            _ensure_user_exists(cur, user_id)
            values_to_insert = [(user_id, rec_text) for rec_text in recommendations if isinstance(rec_text, str)]
            if not values_to_insert: return

            insert_query = "INSERT INTO recommendation_history (user_id, recommendation_text) VALUES %s"
            execute_values(cur, insert_query, values_to_insert)
            conn.commit()
            print(f"Сохранено {len(values_to_insert)} рекомендаций для пользователя {user_id}.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при сохранении истории рекомендаций (user {user_id}): {error}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()

def load_recommendation_history(user_id: int, limit: int = 50) -> list[str]:
    """
    Загружает историю рекомендаций пользователя из БД.
    Возвращает список строк.
    """
    conn = None
    history = []
    try:
        conn = get_db_connection()
        if conn is None: return []

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT recommendation_text FROM recommendation_history
                WHERE user_id = %s ORDER BY generated_at DESC LIMIT %s;
                """, (user_id, limit)
            )
            history = [item['recommendation_text'] for item in cur.fetchall()]
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при загрузке истории рекомендаций (user {user_id}): {error}")
    finally:
        if conn: conn.close()
    return history

def clear_all_data():
    """
    Полностью очищает все данные из таблиц users, books, reading_state и recommendation_history.
    Использует TRUNCATE ... RESTART IDENTITY CASCADE для сброса счетчиков и удаления зависимых данных.
    Возвращает True в случае успеха, False при ошибке.
    """
    conn = None
    success = False

    tables_to_clear = ["users", "books", "reading_state", "recommendation_history"]

    try:
        conn = get_db_connection()
        if conn is None:
            print("Не удалось подключиться к БД для очистки данных.")
            return False

        with conn.cursor() as cur:
            print(f"Начинается очистка таблиц: {', '.join(tables_to_clear)}...")

            truncate_query = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(
                sql.SQL(', ').join(map(sql.Identifier, tables_to_clear))
            )
            print(f"Выполнение SQL: {truncate_query.as_string(conn)}")
            cur.execute(truncate_query)

            conn.commit()
            print("Очистка всех таблиц успешно завершена.")
            success = True

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при очистке данных: {error}")
        if conn:
            try:
                conn.rollback()
                print("Транзакция отменена.")
            except Exception as rollback_err:
                print(f"Ошибка при отмене транзакции: {rollback_err}")
        success = False
    finally:
        if conn:
            conn.close()

    return success
