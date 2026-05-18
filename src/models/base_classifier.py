"""
models/base_classifier.py
Abstract base class for all classifiers.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    """Abstract base – every concrete classifier must implement these methods."""

    def __init__(self, name: str):
        self.name      = name
        self.model     = None
        self.is_fitted = False

    @abstractmethod
    def build_model(self):
        """Instantiate the underlying scikit-learn estimator."""

    def train(self, X, y) -> None:
        """Fit the model on training data."""
        self.build_model()
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X) -> np.ndarray:
        """Return class predictions."""
        self._assert_fitted()
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probability estimates (where supported)."""
        self._assert_fitted()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        raise NotImplementedError(
            f"{self.name} does not support probability estimates.")

    def _assert_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(f"Call train() before predicting with {self.name}.")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
