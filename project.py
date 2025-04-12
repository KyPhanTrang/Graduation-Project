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

print(texts[0])
