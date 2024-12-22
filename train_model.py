import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
import os
import json

# Параметры
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "poly_encoder_model"

# 1. Подготовка данных
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Пример файла данных
# [{"question": "Как поступить в университет?", "answer": "Вы можете подать заявку онлайн."}, ...]

# 2. Загрузка модели и токенайзера
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
base_model = TFAutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

# 3. Построение Poly-encoder
def build_poly_encoder():
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="attention_mask")
    
    outputs = base_model(input_ids, attention_mask=attention_mask)
    cls_token = outputs.last_hidden_state[:, 0, :]  # Используем CLS-токен
    
    output = tf.keras.layers.Dense(256, activation="relu")(cls_token)
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

poly_encoder = build_poly_encoder()
poly_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                     loss="cosine_similarity")

# 4. Тренировка
def prepare_data(data):
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    question_encodings = tokenizer(questions, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN, return_tensors="tf")
    answer_encodings = tokenizer(answers, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN, return_tensors="tf")
    
    return (
        {"input_ids": question_encodings["input_ids"], "attention_mask": question_encodings["attention_mask"]},
        {"input_ids": answer_encodings["input_ids"], "attention_mask": answer_encodings["attention_mask"]}
    )

# Загружаем данные
data = load_data("faq_data.json")
train_data = prepare_data(data)

# Тренировка
poly_encoder.fit(train_data[0], train_data[1], batch_size=BATCH_SIZE, epochs=EPOCHS)

# Сохранение модели
poly_encoder.save_pretrained(MODEL_PATH)
