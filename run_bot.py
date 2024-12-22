from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import json

# Загрузка модели и токенайзера
MODEL_PATH = "poly_encoder_model"
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = TFAutoModel.from_pretrained(MODEL_PATH)

# Загрузка базы вопросов-ответов
with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

def get_best_answer(question):
    # Токенизация вопроса
    inputs = tokenizer(question, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
    question_embedding = model(inputs["input_ids"], inputs["attention_mask"]).last_hidden_state[:, 0, :]  # CLS-токен
    
    # Поиск наиболее похожего ответа
    max_score = -float("inf")
    best_answer = ""
    for item in faq_data:
        answer = item["answer"]
        answer_inputs = tokenizer(answer, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
        answer_embedding = model(answer_inputs["input_ids"], answer_inputs["attention_mask"]).last_hidden_state[:, 0, :]
        
        score = np.dot(question_embedding.numpy(), answer_embedding.numpy().T).item()
        if score > max_score:
            max_score = score
            best_answer = answer
    return best_answer

# Консольный интерфейс
print("Введите вопрос (или 'exit' для выхода):")
while True:
    question = input("> ")
    if question.lower() == "exit":
        break
    print(f"Ответ: {get_best_answer(question)}")
