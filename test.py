# import os
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

# # ==== Tham số ====
# DATA_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/data/"
# TRUNCATE_LENGTH = 300
# TOKENIZER_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/tokenizer.pkl"
# MODEL_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/sentiment_model.h5"

# # ==== Hàm load dữ liệu ====
# def load_data(path, label):
#     texts = []
#     labels = []
#     for fname in os.listdir(path):
#         with open(os.path.join(path, fname), encoding='utf-8') as f:
#             texts.append(f.read())
#             labels.append(label)
#     return texts, labels

# # ==== Load dữ liệu test ====
# pos_test, y_pos_test = load_data(os.path.join(DATA_PATH, "data_test/test/pos"), 1)
# neg_test, y_neg_test = load_data(os.path.join(DATA_PATH, "data_test/test/neg"), 0)
# test_texts = pos_test + neg_test
# test_labels = y_pos_test + y_neg_test

# # ==== Load tokenizer ====
# with open(TOKENIZER_PATH, "rb") as f:
#     tokenizer = pickle.load(f)

# # ==== Tiền xử lý văn bản ====
# test_sequences = tokenizer.texts_to_sequences(test_texts)
# test_data = pad_sequences(test_sequences, maxlen=TRUNCATE_LENGTH)
# test_labels = to_categorical(test_labels)

# # ==== Load mô hình ====
# model = load_model(MODEL_PATH)

# # ==== Đánh giá ====
# loss, accuracy = model.evaluate(test_data, test_labels)
# print(f"\n🎯 Accuracy trên tập test: {accuracy:.4f}")

# # ==== Dự đoán mẫu ====
# pred_probs = model.predict(test_data)
# pred_labels = np.argmax(pred_probs, axis=1)
# true_labels = np.argmax(test_labels, axis=1)

# print("\n📝 Một vài kết quả dự đoán:")
# for i in range(5):
#     print(f"\nVăn bản: {test_texts[i][:100]}...")
#     print(f"  ✅ Thật: {true_labels[i]}  |  🔮 Dự đoán: {pred_labels[i]}")


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ==== Tham số ====
MODEL_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/sentiment_model.h5"
TOKENIZER_PATH = "C:/Users/admin/Downloads/DATN/Graduation_project/tokenizer.pkl"
TRUNCATE_LENGTH = 300  # phải giống khi train

# ==== Load model & tokenizer ====
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ==== Văn bản cần dự đoán ====
text = "Chúng_mình đi 2 người ăn lẩu , tâm_trạng vui phơi_phới !Đến thì gặp con nhân_viên mặt như có chuyện gì tức_tối muốn đuổi khách lắm ấy , ráng nhịn , kêu cái lẩu và vài que xiên que Đợi mãi không thấy xiên que qua hỏi thì_ra chưa làm = > lý_do : quênKiu nhân_viên cho thêm nước_đá , liếc 1 cái - _ -Ai_đời ăn lẩu tính tiền mì gói riêng , tiền rau riêng , cái bill cả tá tiền lặt_vặtLẩu thịt chả được bao_nhiêu , biết vậy ăn bên quận 8 tốt hơn"

# ==== Tiền xử lý ====
sequence = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequence, maxlen=TRUNCATE_LENGTH)

# ==== Dự đoán ====
prediction = model.predict(padded)
label = np.argmax(prediction)

# ==== In kết quả ====
print(f"Xác suất dự đoán: {prediction}")
print("Kết luận: ", "Tích cực" if label == 1 else "Tiêu cực")
