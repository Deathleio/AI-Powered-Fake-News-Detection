import os
import joblib
import numpy as np
from typing import List, Union, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from src.config import config

def build_vectorizer(max_features: int = 50000, ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        lowercase=True
    )

class FakeNewsPipeline:
    def __init__(self, vectorizer: TfidfVectorizer, classifier):
        self.vectorizer = vectorizer
        self.clf = classifier

    def predict(self, texts: List[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(texts)
        return self.clf.predict(X_vec)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(texts)
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(X_vec)
        elif hasattr(self.clf, 'decision_function'):
            df = self.clf.decision_function(X_vec)
            prob1 = 1.0 / (1.0 + np.exp(-df))
            return np.column_stack([1.0 - prob1, prob1])
        else:
            preds = self.clf.predict(X_vec)
            return np.column_stack([1.0 - preds, preds])

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved pipeline to {filepath}", flush=True)

    @classmethod
    def load(cls, filepath: str) -> "FakeNewsPipeline":
        return joblib.load(filepath)
