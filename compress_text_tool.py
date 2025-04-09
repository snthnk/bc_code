import os
import asyncio
import re
import numpy as np
from huggingface_hub import AsyncInferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import database.database as db

embedding_model_name = 'ai-forever/ru-en-RoSBERTa'
embedding_model = None
try:
    embedding_model = SentenceTransformer(embedding_model_name)
    print(f"Модель для эмбеддингов '{embedding_model_name}' успешно загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели для эмбеддингов: {e}. RAG не будет работать.")


RAG_CHUNK_SIZE = 256
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 2


SUMMARY_API_KEYS_STR = os.getenv("SUMMARY_API_KEYS")
summary_api_keys = []

if SUMMARY_API_KEYS_STR:
    summary_api_keys = [key.strip() for key in SUMMARY_API_KEYS_STR.split(',') if key.strip()]

if not summary_api_keys:
    raise ValueError("ОШИБКА: Ключи API для Summarize не найдены или не удалось распарсить из переменной окружения SUMMARY_API_KEYS!")

print(f"Загружено {len(summary_api_keys)} ключей API для Summarize.")
print(f"Первый ключ после парсинга (начало): {summary_api_keys[0][:5]}...")

current_api_key_index = 0
api_key_lock = asyncio.Lock()


async def _initialize_client_with_current_key() -> AsyncInferenceClient | None:
    """
    Инициализирует AsyncInferenceClient с текущим выбранным ключом API.
    Возвращает None в случае сбоя инициализации.
    """
    global current_api_key_index
    key_to_use = summary_api_keys[current_api_key_index]
    print(f"Попытка использовать ключ API с индексом {current_api_key_index}")
    try:
        client = AsyncInferenceClient(
            provider="together",
            api_key=key_to_use
        )
        print(f"Клиент инициализирован с ключом индекса {current_api_key_index}")
        return client
    except Exception as e:
        print(f"Ошибка инициализации клиента с ключом индекса {current_api_key_index}: {e}")
        return None

async def _rotate_api_key():
    """
    Переключает индекс ключа API на следующий.
    """
    global current_api_key_index
    async with api_key_lock:
        current_api_key_index = (current_api_key_index + 1) % len(summary_api_keys)
        print(f"Переключено на ключ API с индексом {current_api_key_index}")

async def _make_robust_api_call(api_call_func, *args, **kwargs):
    """
    Оборачивает вызов API логикой ротации ключей при определенных ошибках.

    Args:
        api_call_func: Асинхронная функция для вызова (принимающая клиент первым аргументом).
        *args: Позиционные аргументы для api_call_func (исключая клиент).
        **kwargs: Именованные аргументы для api_call_func.

    Returns:
        Результат успешного вызова API.

    Raises:
        Exception: Если все ключи исчерпаны или произошла неустранимая ошибка.
    """
    initial_key_index = current_api_key_index
    attempts = 0

    while attempts < len(summary_api_keys):
        client = await _initialize_client_with_current_key()
        if client is None:
            await _rotate_api_key()
            attempts += 1
            if current_api_key_index == initial_key_index:
                raise Exception("Не удалось инициализировать клиент ни с одним из ключей API.")
            continue

        try:
            result = await api_call_func(client, *args, **kwargs)
            return result
        except Exception as e:
            error_message = str(e).lower()
            is_quota_error = (
                "402" in error_message or
                "payment required" in error_message or
                "quota" in error_message or
                "limit" in error_message or
                "insufficient funds" in error_message or
                "rate limit" in error_message
            )

            if is_quota_error:
                print(f"Обнаружена ошибка квоты/оплаты с ключом индекса {current_api_key_index}: {e}")
                await _rotate_api_key()
                attempts += 1
                if current_api_key_index == initial_key_index:
                    raise Exception("Все ключи API исчерпаны или не сработали из-за проблем с квотой/оплатой.")
            else:
                print(f"Ошибка, не связанная с квотой, с ключом индекса {current_api_key_index}: {e}")
                raise e # Перевыброс других ошибок

    raise Exception("Не удалось выполнить вызов API после перебора всех ключей.")


async def _generate_instructions_internal(client: AsyncInferenceClient, text_fragment: str) -> str:
    """
    Внутренняя вспомогательная функция, которая выполняет фактический вызов API для генерации инструкций.
    """
    messages = [
        {
            "role": "user",
            "content": (
                f'''Analyze the given text fragment and determine its type (e.g., scientific, literary, journalistic, technical, philosophical, etc.). Based on the identified type, extract only the key parameters that characterize this type of text.

                Output the result strictly as a comma-separated list of parameters in russian without any additional text.

                Examples of key parameters for different text types:
                - **Literary**: персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды
                - **Scientific**: основные термины, гипотезы, методы, доказательства, выводы
                - **Journalistic**: ключевые события, участники, место, время, аргументы
                - **Technical**: предмет описания, термины, инструкции, алгоритмы, примеры
                - **Philosophical**: основные идеи, аргументы, философские термины, парадоксы, концепции

                You need to answer just the list of parameters, nothing else.

                Examples of output:
                персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды
                предмет описания, термины, инструкции
                аргументы, философские термины, парадоксы, концепции

                Text Fragment:
                "{text_fragment}"
                '''
            )
        }
    ]
    completion = await client.chat_completion(
        messages=messages,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_tokens=100,
        temperature=0.1
    )
    return completion.choices[0].message.content.strip()

async def generate_instructions(text_fragment: str) -> str:
    """
    Анализирует фрагмент текста для определения его типа и генерации ключевых
    параметров (инструкций) для суммирования, используя ротацию ключей API.
    Возвращает инструкции по умолчанию в случае сбоя после всех попыток.
    """
    try:
        instructions = await _make_robust_api_call(_generate_instructions_internal, text_fragment)
        return instructions
    except Exception as e:
        print(f"Не удалось сгенерировать инструкции после всех попыток: {e}")
        return "основные события, ключевые идеи, главные герои"


def segment_text(text: str, num_segments: int) -> list[str]:
    """
    Разделяет текст на указанное количество сегментов на основе предложений.
    Гарантирует, что сегменты заканчиваются знаками препинания.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    if not sentences:
        return []

    total_sentences = len(sentences)
    actual_num_segments = max(1, min(num_segments, total_sentences))
    segment_size = (total_sentences + actual_num_segments - 1) // actual_num_segments

    segments = []
    for i in range(actual_num_segments):
        start_index = i * segment_size
        end_index = min((i + 1) * segment_size, total_sentences)
        segment_sentences = sentences[start_index:end_index]
        if segment_sentences:
            segment_text_val = '. '.join(segment_sentences)
            if not segment_text_val.endswith(('.', '!', '?')):
                 segment_text_val += '.'
            segments.append(segment_text_val)

    return segments


def chunk_for_rag(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Разделяет текст на перекрывающиеся фрагменты (чанки) указанного размера
    в словах для индексации RAG.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = min(start_index + chunk_size, len(words))
        chunk_words = words[start_index:end_index]
        chunks.append(" ".join(chunk_words))
        next_start = start_index + chunk_size - chunk_overlap
        if next_start <= start_index:
            next_start = start_index + 1
        start_index = next_start
        if start_index >= len(words) and end_index == len(words):
             break

    return chunks


def build_rag_index(rag_chunks: list[str]) -> tuple[list[str], np.ndarray | None]:
    """
    Создает эмбеддинги для текстовых чанков с использованием загруженной модели
    sentence transformer. Предназначено для использования с run_in_executor.
    """
    if not embedding_model or not rag_chunks:
        return rag_chunks, None
    try:
        print(f"Генерация эмбеддингов для {len(rag_chunks)} RAG чанков...")
        embeddings = embedding_model.encode(rag_chunks, show_progress_bar=False)
        print("Эмбеддинги для RAG сгенерированы.")
        return rag_chunks, np.array(embeddings)
    except Exception as e:
        print(f"Ошибка генерации RAG эмбеддингов: {e}")
        return rag_chunks, None


def retrieve_relevant_chunks(query_segment: str, rag_index: tuple[list[str], np.ndarray | None], top_k: int) -> list[str]:
    """
    Находит top_k наиболее релевантных чанков из индекса RAG для заданного
    запроса-сегмента на основе косинусного сходства. Предназначено для
    использования с run_in_executor.
    """
    rag_chunks, rag_embeddings = rag_index
    if rag_embeddings is None or not embedding_model or not query_segment:
        return []

    try:
        query_embedding = embedding_model.encode([query_segment])
        similarities = cosine_similarity(query_embedding, rag_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        relevant_chunks_text = []
        added_count = 0
        for index in sorted_indices:
            if rag_chunks[index] != query_segment and added_count < top_k:
                 relevant_chunks_text.append(rag_chunks[index])
                 added_count += 1
            elif added_count >= top_k:
                 break

        print(f"Найдено {len(relevant_chunks_text)} релевантных чанков для сегмента.")
        return relevant_chunks_text

    except Exception as e:
        print(f"Ошибка извлечения релевантных чанков: {e}")
        return []


async def _summarize_segment_internal(
    client: AsyncInferenceClient,
    segment: str,
    target_words: int,
    instructions: str,
    retrieved_context: list[str] | None = None
    ) -> str:
    """
    Внутренняя вспомогательная функция, которая выполняет фактический вызов API для суммирования сегмента.
    """
    context_prompt_part = ""
    if retrieved_context:
        context_items = [f"- {ctx}" for ctx in retrieved_context]
        context_prompt_part = (
            "Для лучшего понимания важности информации в сжимаемом сегменте, учти следующий дополнительный контекст из других частей документа:\n"
            + "\n".join(context_items)
            + "\nВажно: Не нужно суммировать этот дополнительный контекст, он дан только для справки о связях сегмента с остальным текстом.\n"
        )

    system_prompt = (
        f"Сгенерируй краткое содержание (summary) на русском языке для следующего сегмента текста объемом примерно {target_words} слов. "
        f"Уделяй особое внимание следующим аспектам, характерным для этого типа текста: {instructions}. "
        f"{context_prompt_part}"
        "Это фрагмент из большего текста. Не начинай с фраз 'В этом тексте', 'Этот фрагмент о'. "
        "Просто передай суть сегмента как связный абзац."
        "Избегай прямых цитат. Пересказывай своими словами."
        "Ответ должен быть только сжатым текстом, без дополнительных фраз."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": segment}
    ]

    completion = await client.chat_completion(
        messages=messages,
        model="scb10x/llama-3-typhoon-v1.5x-70b-instruct-awq",
        max_tokens=int(target_words * 2.5),
        temperature=0.5
    )
    summary = completion.choices[0].message.content.strip()
    summary = re.sub(r'^(В этом фрагменте|В данном тексте|Этот сегмент о)\s*:?\s*', '', summary, flags=re.IGNORECASE)
    return summary

async def summarize_segment(
    segment: str,
    target_words: int,
    instructions: str,
    retrieved_context: list[str] | None = None
    ) -> str:
    """
    Суммирует один текстовый сегмент с использованием LLM, опционально используя
    контекст RAG. Использует ротацию ключей API. Возвращает пустую строку ""
    в случае сбоя после всех попыток.
    """
    try:
        summary = await _make_robust_api_call(
            _summarize_segment_internal,
            segment,
            target_words,
            instructions,
            retrieved_context
        )
        return summary
    except Exception as e:
        print(f"Не удалось суммировать сегмент после всех попыток: {e}. Начало сегмента: '{segment[:50]}...'")
        return "" # Возвращаем пустую строку при окончательной неудаче


async def summarize_text_and_update_db(user_id: int, book_id: int, overall_target_chars: int):
    """
    Извлекает полный текст, индексирует для RAG, сегментирует, асинхронно
    суммирует сегменты с контекстом RAG (извлекаемым асинхронно),
    объединяет результаты и обновляет базу данных. Использует ротацию ключей API
    для вызовов LLM.
    """
    print(f"[RAG] Начало суммирования для user_id={user_id}, book_id={book_id}, целевое кол-во символов={overall_target_chars}")

    loop = asyncio.get_running_loop()

    full_text = await loop.run_in_executor(None, db.get_book_full_text, user_id, book_id)
    user_books = await loop.run_in_executor(None, db.get_user_books, user_id)
    selected_book = next((book for book in user_books if book['book_id'] == book_id), None)
    total_pages = selected_book.get('total_pages') if selected_book else 0

    if not full_text or (isinstance(full_text, str) and full_text.startswith("Ошибка")):
        print(f"[RAG] Ошибка: Не удалось получить текст для книги {book_id}.")
        return
    if total_pages is None or total_pages <= 0:
         print(f"[RAG] Предупреждение: Не удалось определить количество страниц для книги {book_id}. Значение update_page будет 1.")
         update_page_value = 1
    else:
        update_page_value = total_pages


    rag_index = ([], None)
    if embedding_model:
        print("[RAG] Создание чанков для RAG...")
        rag_chunks = chunk_for_rag(full_text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
        if rag_chunks:
             print("[RAG] Построение индекса RAG (выполняется в executor)...")
             rag_index = await loop.run_in_executor(None, build_rag_index, rag_chunks)
             if rag_index[1] is not None:
                 print(f"[RAG] Индекс RAG построен с {len(rag_index[1])} векторами.")
             else:
                 print("[RAG] Построение индекса RAG завершено, но результат None.")
        else:
             print("[RAG] Не удалось создать чанки для RAG.")
    else:
        print("[RAG] Модель эмбеддингов не загружена, RAG использоваться не будет.")


    mid_point = len(full_text) // 2
    fragment = full_text[max(mid_point - 700, 0): min(mid_point + 700, len(full_text))]
    print("[RAG] Генерация инструкций...")
    instructions = await generate_instructions(fragment)
    print(f"Инструкции для суммирования: {instructions}")


    overall_target_words = max(100, overall_target_chars // 5)
    num_segments = max(5, overall_target_words // 180)
    print(f"[RAG] Сегментация текста примерно на {num_segments} частей...")
    segments = segment_text(full_text, num_segments)
    if not segments:
         print(f"[RAG] Ошибка: Не удалось сегментировать текст для книги {book_id}.")
         return

    actual_num_segments = len(segments)
    print(f"Текст разделен на {actual_num_segments} сегментов для суммирования.")
    target_words_per_segment = (overall_target_words + actual_num_segments - 1) // actual_num_segments
    print(f"Целевое количество слов на сегмент: ~{target_words_per_segment}")

    tasks = []
    print("[RAG] Подготовка задач суммирования с извлечением контекста...")

    async def retrieve_and_summarize_task(segment_index, seg):
        """Вспомогательная задача для извлечения контекста и последующего суммирования."""
        relevant_context = []
        if rag_index[1] is not None:
            print(f"  -> Извлечение контекста для сегмента {segment_index+1} (в executor)...")
            relevant_context = await loop.run_in_executor(
                None, retrieve_relevant_chunks, seg, rag_index, RAG_TOP_K
            )
        print(f"  -> Суммирование сегмента {segment_index+1}...")
        summary = await summarize_segment(
            segment=seg,
            target_words=target_words_per_segment,
            instructions=instructions,
            retrieved_context=relevant_context
        )
        return summary

    for i, seg in enumerate(segments):
        tasks.append(retrieve_and_summarize_task(i, seg))

    print(f"[RAG] Запуск {len(tasks)} задач суммирования...")
    summaries = await asyncio.gather(*tasks)
    print("[RAG] Все задачи суммирования завершены.")


    final_summary = ' '.join(filter(None, summaries)).strip()
    if not final_summary:
        print(f"[RAG] Ошибка: Итоговое саммари пустое для книги {book_id}. Возможно, все сегменты завершились с ошибкой.")
        return

    print(f"Итоговое RAG саммари сгенерировано, длина: {len(final_summary)} символов.")

    print(f"[RAG] Сохранение саммари в БД для книги {book_id}, update_page={update_page_value} (в executor)...")
    success = await loop.run_in_executor(
        None, db.update_ai_context, user_id, book_id, update_page_value, final_summary
    )

    if success:
        print(f"[RAG] Саммари для книги {book_id} (пользователь {user_id}) успешно сохранено в БД.")
    else:
        print(f"[RAG] Ошибка: Не удалось сохранить саммари для книги {book_id} в БД (db.update_ai_context вернул False).")
