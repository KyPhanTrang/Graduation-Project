import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Masking
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split


# ==== Tham số ====
DATA_PATH = "C:/Users/admin/Downloads/DATN/data/"
WORD2VEC_PATH = "C:/Users/admin/Downloads/DATN/data/vnw2v.bin"
EMBEDDING_DIM = 300
TRUNCATE_LENGTH = 300
BATCH_SIZE = 50
EPOCHS = 5

# ==== Load dữ liệu văn bản ====
def load_data(path, label):
    texts = []
    labels = []
    for fname in os.listdir(path):
        with open(os.path.join(path, fname), encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(label)
    return texts, labels

pos_train, y_pos = load_data(os.path.join(DATA_PATH, "data_train/train/pos"), 1)
neg_train, y_neg = load_data(os.path.join(DATA_PATH, "data_train/train/neg"), 0)
texts = pos_train + neg_train
labels = y_pos + y_neg
# print(texts[0])

# ==== Token hóa ====
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print(f"Số lượng từ: {len(word_index)}")

# ==== Padding ====
data = pad_sequences(sequences, maxlen=TRUNCATE_LENGTH)
labels = to_categorical(labels)

# ==== Load Word2Vec ====
w2v = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in w2v:
        embedding_matrix[i] = w2v[word]

# ==== Xây dựng mô hình ====
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=TRUNCATE_LENGTH,
                    trainable=False))
model.add(Masking(mask_value=0.0))
model.add(LSTM(200, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

# ==== Train ====
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val))
