import os
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.config import config

class StackingEnsembleModel:
    """
    Combines predictions from multiple diverse base models using soft-voting and meta-learning.
    """
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights
        self.meta_learner: Optional[LogisticRegression] = None

    def fit_meta_learner(self, base_probas: np.ndarray, y_true: np.ndarray):
        """
        base_probas: shape (n_samples, n_models)
        """
        self.meta_learner = LogisticRegression(C=1.0, random_state=config.RANDOM_SEED)
        self.meta_learner.fit(base_probas, y_true)

    def predict_weighted_proba(self, base_probas: np.ndarray) -> np.ndarray:
        """
        Computes weighted average probability across models.
        """
        if self.weights is None:
            return np.mean(base_probas, axis=1)
        w = np.array(self.weights) / np.sum(self.weights)
        return np.sum(base_probas * w, axis=1)

    def predict_meta(self, base_probas: np.ndarray) -> np.ndarray:
        if self.meta_learner is None:
            return self.predict_weighted_proba(base_probas)
        return self.meta_learner.predict_proba(base_probas)[:, 1]

    def save(self, filepath: str):
        joblib.dump({'weights': self.weights, 'meta_learner': self.meta_learner}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "StackingEnsembleModel":
        data = joblib.load(filepath)
        instance = cls(weights=data['weights'])
        instance.meta_learner = data['meta_learner']
        return instance
