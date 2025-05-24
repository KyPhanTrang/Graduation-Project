import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


class SklearnSentimentTrainer:
    def __init__(self, model_type='svm'):
        assert model_type in ['svm', 'nb'],
        self.model_type = model_type
        self.pipeline = None

    def load_data(self, pos_path, neg_path):
        texts, labels = [], []
        for fname in os.listdir(pos_path):
            with open(os.path.join(pos_path, fname), encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)
        for fname in os.listdir(neg_path):
            with open(os.path.join(neg_path, fname), encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)
        return texts, labels

    def build_pipeline(self):
        vectorizer = TfidfVectorizer(max_features=5000)

        if self.model_type == 'svm':
            classifier = SVC(kernel='linear', probability=True)
        else:
            classifier = MultinomialNB()

        self.pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', classifier)
        ])

    def train(self, texts, labels, test_size=0.2):
        x_train, x_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42)

        self.build_pipeline()
        self.pipeline.fit(x_train, y_train)

        y_pred = self.pipeline.predict(x_val)
        print(classification_report(y_val, y_pred))

    def save_model(self, path='sentiment_sklearn.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, path='sentiment_sklearn.pkl'):
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)
