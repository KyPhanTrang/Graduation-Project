import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ==== Tham số ====
DATA_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/data/"
TRUNCATE_LENGTH = 300
TOKENIZER_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/tokenizer.pkl"
MODEL_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/sentiment_model.h5"

# ==== Hàm load dữ liệu ====
def load_data(path, label):
    texts = []
    labels = []
    for fname in os.listdir(path):
        with open(os.path.join(path, fname), encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(label)
    return texts, labels

# ==== Load dữ liệu test ====
pos_test, y_pos_test = load_data(os.path.join(DATA_PATH, "data_test/test/pos"), 1)
neg_test, y_neg_test = load_data(os.path.join(DATA_PATH, "data_test/test/neg"), 0)
test_texts = pos_test + neg_test
test_labels = y_pos_test + y_neg_test

# ==== Load tokenizer ====
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ==== Tiền xử lý văn bản ====
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=TRUNCATE_LENGTH)
test_labels = to_categorical(test_labels)

# ==== Load mô hình ====
model = load_model(MODEL_PATH)

# ==== Đánh giá ====
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"\nAccuracy trên tập test: {accuracy:.4f}")

# ==== Dự đoán mẫu ====
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(test_labels, axis=1)

print("\nMột vài kết quả dự đoán:")
for i in range(5):
    print(f"\nVăn bản: {test_texts[i][:100]}...")
    print(f"  Thật: {true_labels[i]}  |  Dự đoán: {pred_labels[i]}")