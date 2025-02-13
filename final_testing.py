import re
import time
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import InferenceClient
import random

# API-ключи
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
    sentences = [s.strip() for s in re.split(r'[!.?]\s*', text) if s.strip()]

    n_segments = max(1, overall_target_words // 100)
    segment_size = len(sentences) // n_segments

    segments = []
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(sentences)
        segment = '. '.join(sentences[start:end]) + '.'
        segments.append(segment)

    return segments


def summarize_segment(segment, target_words, instructions):
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


def summarize_text(text, overall_target_words, use_instructions):
    if use_instructions:
        print("\nГенерация инструкций...")
        instructions = generate_instructions(text)
        print(f"Полученные инструкции: {instructions}")
    else:
        instructions = ""

    print("\nСегментирование текста...")
    segments = segment_text(text, overall_target_words)
    print(f"Текст разделен на {len(segments)} сегментов")

    summaries = []
    fixed_segment_target = 100

    for i, seg in enumerate(segments):
        print(f"\nОбработка сегмента {i + 1}/{len(segments)}...")
        start_time = time.time()

        summary = summarize_segment(
            segment=seg,
            target_words=fixed_segment_target,
            instructions=instructions if use_instructions else ""
        )

        elapsed = time.time() - start_time
        print(f"Сегмент обработан за {elapsed:.2f} сек")
        print(f"Сгенерированное саммари ({len(summary.split())} слов):\n{summary[:150]}...")
        summaries.append(summary)

    final_summary = ' '.join(summaries)
    return final_summary


def evaluate_summary(generated, reference):
    model = SentenceTransformer('sberbank-ai/sbert_large_mt_nlu_ru')
    emb1 = model.encode(generated, convert_to_tensor=True)
    emb2 = model.encode(reference, convert_to_tensor=True)
    bert_score = util.cos_sim(emb1, emb2).item()

    def jaccard(str1, str2):
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union else 0

    jaccard_sim = jaccard(generated, reference)

    return bert_score, jaccard_sim


def run_tests(use_instructions, num_samples):
    print("\nЗагрузка датасетов...")

    datasets_info = {
        "Hacker1337/ru_dialogsum": {"text": "dialogue", "summary": "summary"},
        "RussianNLP/Mixed-Summarization-Dataset": {"text": "text", "summary": "summary"},
        "slon-hk/BooksSummarizationRU": {"text": "Full text", "summary": "Summary"},
        "IlyaGusev/gazeta": {"text": "text", "summary": "summary"}
    }

    all_samples = []

    for ds_name, fields in datasets_info.items():
        try:
            ds = load_dataset(ds_name, split="test")
        except Exception as e:
            print(f"Ошибка при загрузке датасета {ds_name}: {e}")
            continue

        ds_samples = ds.shuffle(seed=42).select(range(num_samples))
        for sample in ds_samples:
            sample["_text_key"] = fields["text"]
            sample["_summary_key"] = fields["summary"]
            all_samples.append(sample)

    if not all_samples:
        print("Не удалось загрузить ни одного датасета для тестирования.")
        return

    random.shuffle(all_samples)

    total_bert = 0
    total_jaccard = 0
    total_throughput = 0
    num_total = len(all_samples)

    for i, sample in enumerate(tqdm(all_samples, desc="Тестирование")):
        text_key = sample.get("_text_key", "text")
        summary_key = sample.get("_summary_key", "summary")
        text = sample.get(text_key, "")
        reference = sample.get(summary_key, "")

        if not text or not reference:
            print(f"Пропущен пример {i + 1} из-за отсутствия текста или саммари.")
            continue

        target_length = len(reference.split())

        print(f"\n\nПример {i + 1}/{num_total}")
        print(f"Целевая длина саммари: {target_length} слов")
        print("Референсное саммари:", reference[:100] + "...")

        start_time = time.time()
        try:
            generated = summarize_text(
                text=text,
                overall_target_words=target_length,
                use_instructions=use_instructions
            )
        except Exception as e:
            print(f"Ошибка при генерации саммари: {str(e)}")
            continue

        elapsed = time.time() - start_time
        input_words = len(text.split())
        gen_length = len(generated.split())

        throughput = (input_words + gen_length) / elapsed if elapsed > 0 else 0

        print(f"\nРезультат ({gen_length} слов):\n{generated[:150]}...")
        print(f"Производительность: {throughput:.2f} слов/сек")

        bert_score, jaccard = evaluate_summary(generated, reference)

        total_bert += bert_score
        total_jaccard += jaccard
        total_throughput += throughput

        print(f"BERT-Score: {bert_score:.3f}")
        print(f"Коэф. Жаккара: {jaccard:.3f}")

    print("\nИтоговые результаты:")
    print(f"Средний BERT-Score: {total_bert / num_total:.3f}")
    print(f"Средний коэф. Жаккара: {total_jaccard / num_total:.3f}")
    print(f"Средняя производительность: {total_throughput / num_total:.2f} слов/сек")


print("=" * 50)
print("ТЕСТИРОВАНИЕ С ИНСТРУКЦИЯМИ")
print("=" * 50)
run_tests(use_instructions=True, num_samples=100)

print("\n\n" + "=" * 50)
print("ТЕСТИРОВАНИЕ БЕЗ ИНСТРУКЦИЙ")
print("=" * 50)
run_tests(use_instructions=False, num_samples=100)
