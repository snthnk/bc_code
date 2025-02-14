import time
from tqdm import tqdm
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import re
import torch
import pandas as pd

# Список токенов для доступа к API
tokens = []
token_index_hf = 0
client_hf = InferenceClient(provider="hf-inference", api_key=tokens[token_index_hf])
bert_model = SentenceTransformer('ai-forever/ru-en-RoSBERTa')

def compute_jaccard_similarity(text1, text2):
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0

def eval_model_api(model_name, prompt, size):
    global token_index_hf
    system_prompt = f"Сгенерируйте краткое содержание на русском языке объемом примерно {size} слов. Сохраняйте ключевые слова, используйте фразы из текста."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(len(tokens)):
        try:
            completion = client_hf.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=500
            )
            return completion.choices[0].message.content
        except Exception as e:
            if "402" in str(e) or "Payment Required" in str(e):
                token_index_hf = (token_index_hf + 1) % len(tokens)
                client_hf.api_key = tokens[token_index_hf]
            else:
                raise e
    raise Exception("Все токены недоступны")

def eval_model_local(model_name, prompt, size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    system_prompt = f"Сгенерируйте краткое содержание на русском языке объемом примерно {size} слов. Сохраняйте ключевые слова, используйте фразы из текста."

    full_prompt = system_prompt + "\n\n" + prompt

    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)

    generation_kwargs = {
        'max_length': 500,
        'num_return_sequences': 1,
        'temperature': 0.7,
        'top_k': 50,
        'top_p': 0.95,
        'do_sample': True,
        'repetition_penalty': 1.2,
        'length_penalty': 1.0
    }

    output = model.generate(**inputs, **generation_kwargs)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

def evaluate_models(dataset_sample):
    results = []

    models = {
        'phi': "microsoft/Phi-3.5-mini-instruct",
        'gemma': "google/gemma-2b-it"
    }

    for example in tqdm(dataset_sample, desc="Оценка моделей"):
        prompt = example['text']
        reference = example['summary']
        reference_size = len(reference.split())

        for model_key, model_name in models.items():
            print(f"\nТестирование модели {model_key} через API...")
            start_time = time.time()
            prediction_api = eval_model_api(model_name, prompt, reference_size)
            elapsed_api = time.time() - start_time
            jaccard_api = compute_jaccard_similarity(prediction_api, reference)

            with torch.no_grad():
                emb_pred_api = bert_model.encode(prediction_api, convert_to_tensor=True)
                emb_ref_api = bert_model.encode(reference, convert_to_tensor=True)
                bert_score_api = util.pytorch_cos_sim(emb_pred_api, emb_ref_api).item()

            print(f"\nТестирование модели {model_key} локально...")
            start_time = time.time()
            prediction_local = eval_model_local(model_name, prompt, reference_size)
            elapsed_local = time.time() - start_time
            jaccard_local = compute_jaccard_similarity(prediction_local, reference)

            with torch.no_grad():
                emb_pred_local = bert_model.encode(prediction_local, convert_to_tensor=True)
                emb_ref_local = bert_model.encode(reference, convert_to_tensor=True)
                bert_score_local = util.pytorch_cos_sim(emb_pred_local, emb_ref_local).item()

            results.append({
                'model': model_key,
                'method': 'API',
                'speed': elapsed_api,
                'jaccard': jaccard_api,
                'bert_score': abs(bert_score_api)
            })

            results.append({
                'model': model_key,
                'method': 'Local',
                'speed': elapsed_local,
                'jaccard': jaccard_local,
                'bert_score': abs(bert_score_local)
            })

            print(f"API - Время: {elapsed_api:.2f} сек | Jaccard: {jaccard_api:.4f} | BERT: {bert_score_api:.4f}")
            print(f"Local - Время: {elapsed_local:.2f} сек | Jaccard: {jaccard_local:.4f} | BERT: {bert_score_local:.4f}")

    return pd.DataFrame(results)


dataset = datasets.load_dataset("RussianNLP/Mixed-Summarization-Dataset", split='test')
sample = dataset.select(range(100))

results_df = evaluate_models(sample)

aggregated = results_df.groupby(['model', 'method']).agg({
    'speed': 'mean',
    'jaccard': 'mean',
    'bert_score': 'mean'
}).reset_index()

print("\nИтоговые результаты:")
print(aggregated.to_markdown(index=False))
