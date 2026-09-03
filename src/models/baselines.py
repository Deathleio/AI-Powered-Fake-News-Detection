import os
import joblib
import numpy as np
import scipy.sparse as sp
from typing import List, Union, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import config

def build_word_vectorizer(max_features: int = 40000, ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """Word-level n-gram TF-IDF. Captures entity and topic signals."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=3,
        max_df=0.90,
        stop_words='english',
        lowercase=True,
        analyzer='word'
    )

def build_char_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    """
    Character-level 3-4-gram TF-IDF. Captures stylistic writing patterns:
    - Typos ('acutally', 'mainn') -> strong fake indicator
    - ALL CAPS sequences -> strong fake indicator
    - Unusual punctuation clusters ('!!!', '???') -> fake indicator
    - Journalistic sentence structure -> real indicator
    """
    return TfidfVectorizer(
        ngram_range=(3, 4),
        max_features=max_features,
        sublinear_tf=True,
        min_df=10,
        max_df=0.95,
        lowercase=False,  # Preserve casing so ALL CAPS is captured
        analyzer='char_wb'
    )

# Keep backward-compatible alias
def build_vectorizer(max_features: int = 40000, ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    return build_word_vectorizer(max_features=max_features, ngram_range=ngram_range)


class FakeNewsPipeline:
    """
    Dual-vectorizer pipeline that combines:
    - Word 1-2-gram TF-IDF on full fused text (topic/entity signals)
    - Character 3-4-gram TF-IDF on title only (fast stylistic signals:
      typos, ALL CAPS, punctuation density)

    At inference, pass title_texts= separately for char features.
    """
    def __init__(self, vectorizer, classifier, char_vectorizer=None):
        self.vectorizer = vectorizer
        self.clf = classifier
        self.char_vectorizer = char_vectorizer

    def _get_features(self, texts: List[str], title_texts: Optional[List[str]] = None) -> sp.csr_matrix:
        word_features = self.vectorizer.transform(texts)
        if self.char_vectorizer is not None:
            # Use title_texts for char features if available; fall back to full text
            char_input = title_texts if title_texts is not None else texts
            char_features = self.char_vectorizer.transform(char_input)
            return sp.hstack([word_features, char_features], format='csr')
        return word_features

    def predict(self, texts: List[str], title_texts: Optional[List[str]] = None) -> np.ndarray:
        X = self._get_features(texts, title_texts)
        return self.clf.predict(X)

    def predict_proba(self, texts: List[str], title_texts: Optional[List[str]] = None) -> np.ndarray:
        X = self._get_features(texts, title_texts)
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(X)
        elif hasattr(self.clf, 'decision_function'):
            df = self.clf.decision_function(X)
            prob1 = 1.0 / (1.0 + np.exp(-df))
            return np.column_stack([1.0 - prob1, prob1])
        else:
            preds = self.clf.predict(X)
            return np.column_stack([1.0 - preds, preds])

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved pipeline to {filepath}", flush=True)

    @classmethod
    def load(cls, filepath: str) -> "FakeNewsPipeline":
        instance = joblib.load(filepath)
        if hasattr(instance, 'clf') and not hasattr(instance.clf, 'multi_class'):
            instance.clf.multi_class = 'auto'
        return instance
