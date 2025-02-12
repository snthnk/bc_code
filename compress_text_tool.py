import re
from huggingface_hub import InferenceClient


# Список API-ключей
api_keys = []
api_key_index = 0

client = InferenceClient(
    provider="together",
    api_key=api_keys[api_key_index]
)

def generate_instructions(text):
    mid_point = len(text) // 2
    fragment = text[max(mid_point - 500, 0): mid_point + 500]
    messages = [
        {
            "role": "user",
            "content": (
                f'''Analyze the given text and determine its type (e.g., scientific, literary, journalistic, technical, philosophical, etc.). Based on the identified type, extract only the key parameters that characterize this type of text.  

                Output the result strictly as a comma-separated list of parameters in russian without any additional text.  

                Examples of key parameters for different text types:  
                - **Literary: персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды  
                - **Scientific: основные термины, гипотезы, методы, доказательства, выводы  
                - **Journalistic: ключевые события, участники, место, время, аргументы  
                - **Technical: предмет описания, термины, инструкции, алгоритмы, примеры  
                - **Philosophical: основные идеи, аргументы, философские термины, парадоксы, концепции

                You need to answer just the list of parameters, nothing else.

                Examples of output:
                персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды
                предмет описания, термины, инструкции
                аргументы, философские термины, парадоксы, концепции

                Text:  
                "{fragment}"  

                '''
            )
        }
    ]
    completion = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=messages,
        max_tokens=100
    )
    instructions = completion.choices[0].message.content.strip()
    return instructions

def segment_text(text, overall_target_words):
    """
    Разбивает текст на `n` сегментов, где `n = overall_target_words // 150`.
    """
    sentences = [s.strip() for s in re.split(r'[!.?]\s*', text) if s.strip()]

    n_segments = max(1, overall_target_words // 100)  # Количество сегментов
    segment_size = len(sentences) // n_segments  # Размер сегмента в предложениях

    segments = []
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(sentences)
        segment = '. '.join(sentences[start:end]) + '.'
        segments.append(segment)


    return segments

def summarize_segment(segment, target_words, instructions):
    """
    Сжимает данный сегмент до целевого количества слов (примерно target_words).
    """
    global api_key_index
    system_prompt = (
        f"Сгенерируйте краткое содержание на русском языке объемом {target_words} слов."
        f"Очень важно: уделяйте особое внимание следующим аспектам: {instructions}"
        "Учтите, что данный текст является фрагментом из большего документа."
        "Не начинайте саммари с фраз вроде 'В этом фрагменте описывается'. "
        "Просто передайте суть текста без вступлений и заключений."
        "Формат ответа - пересказ текста в виде абзаца"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": segment}
    ]

    attempts = 0
    while attempts < len(api_keys):
        try:
            completion = client.chat.completions.create(
                model="scb10x/llama-3-typhoon-v1.5x-70b-instruct-awq",
                messages=messages,
                max_tokens=5000
            )
            summary = completion.choices[0].message.content
            return summary
        except Exception as e:
            error_message = str(e)
            if "402" in error_message or "Payment Required" in error_message:
                api_key_index = (api_key_index + 1) % len(api_keys)
                client.api_key = api_keys[api_key_index]
                attempts += 1
            else:
                raise e
    raise Exception("Все API-ключи исчерпаны")

def summarize_text(file_path, overall_target_words):
    """
    Сжимает большой текст до итогового размера overall_target_words.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    instructions = generate_instructions(text)
    segments = segment_text(text, overall_target_words)
    summaries = []
    fixed_segment_target = 100  # Фиксированный целевой размер саммари для каждого сегмента
    for i, seg in enumerate(segments):
        summary = summarize_segment(seg, fixed_segment_target, instructions)
        summaries.append(summary)

    final_summary = ' '.join(summaries)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(final_summary)



# Пример использования
file_path = "example.txt"
summarize_text(file_path, 500)