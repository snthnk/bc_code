import datasets
import time
import pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient
from openai import OpenAI
import torch
import re
from sentence_transformers import SentenceTransformer, util

# Список токенов для доступа к API
tokens = []

token_index_hf = 0

client_hf = InferenceClient(
    provider="together",
    api_key=tokens[token_index_hf]
)

bert_model = SentenceTransformer('sberbank-ai/sbert_large_mt_nlu_ru')


def eval_model(model_name, prompt, size):
    global token_index_hf
    system_prompt = f"Сгенерируйте краткое содержание на русском языке объемом примерно {size} слов. Сохраняйте ключевые слова, используйте фразы из текста."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    attempts = 0

    while attempts < len(tokens):
        try:
            completion = client_hf.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=500
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if "402" in error_message or "Payment Required" in error_message:
                token_index_hf = (token_index_hf + 1) % len(tokens)
                client_hf.api_key = tokens[token_index_hf]
                attempts += 1
            else:
                raise e
    raise Exception("All tokens exhausted")

def compute_jaccard_similarity(text1, text2):
    text1 = set(re.findall(r'\b\w+\b', text1.lower()))
    text2 = set(re.findall(r'\b\w+\b', text2.lower()))

    intersection = text1 & text2
    union = text1 | text2

    return len(intersection) / len(union) if union else 0.0

def evaluate_models(dataset_sample):
    results = []
    models = {
        'mixtral': "mistralai/Mixtral-8x22B-Instruct-v0.1",
        'nemotron': "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        'typhoon': "scb10x/llama-3-typhoon-v1.5x-70b-instruct-awq"
    }

    for example in tqdm(dataset_sample, desc="Оценка моделей"):
        prompt = example['text']
        reference = example['summary']
        reference_size = len(reference.split())

        for model, model_name in models.items():
            start_time = time.time()
            prediction = eval_model(model_name, prompt, reference_size)
            elapsed = time.time() - start_time

            jaccard = compute_jaccard_similarity(prediction, reference)

            if prediction.strip() and reference.strip():
                emb_pred = bert_model.encode(prediction, convert_to_tensor=True)
                emb_ref = bert_model.encode(reference, convert_to_tensor=True)
                bert_score = util.pytorch_cos_sim(emb_pred, emb_ref).item()
            else:
                bert_score = 0.0

            # Вывод отладочной информации
            print(f"\nМодель: {model}")
            print(f"Текст: {prompt}")
            print(f"Референс: {reference}")
            print(f"Предсказание: {prediction}")
            print(f"Время выполнения: {elapsed:.2f} сек")
            print(f"Jaccard Similarity: {jaccard:.4f}")
            print(f"BERT-Score: {bert_score:.4f}\n")

            results.append({
                'model': model,
                'speed': elapsed,
                'jaccard': jaccard,
                'bert_score': bert_score
            })

    return pd.DataFrame(results)

dataset = datasets.load_dataset("RussianNLP/Mixed-Summarization-Dataset", split='test')
sample = dataset.select(range(100))

results_df = evaluate_models(sample)

aggregated = results_df.groupby('model').agg({
    'speed': 'mean',
    'jaccard': 'mean',
    'bert_score': 'mean'
}).reset_index()

print("\nСводные результаты сравнения моделей:")
print(aggregated.to_markdown(index=False))