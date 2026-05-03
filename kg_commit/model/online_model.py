from __future__ import annotations
import numpy as np
from sklearn.linear_model import SGDClassifier
from kg_commit.core.window import Window
from .base_model import BaseModel


class OnlineModel(BaseModel):
    def __init__(self, classes: list[int] | None = None):
        self.classes = np.array(classes, dtype=int) if classes is not None else None
        self.classifier = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
        self._initialized = False

    def fit(self, windows: list[Window]):
        for window in windows:
            self.update(window)

    def update(self, window: Window):
        X = window.get_features()
        y = window.get_labels()

        if not self._initialized:
            if self.classes is None:
                self.classes = np.unique(y).astype(int)
            self.classifier.partial_fit(X, y, classes=self.classes)
            self._initialized = True
            return

        self.classifier.partial_fit(X, y)

    def predict(self, window: Window):
        X = window.get_features()
        if not self._initialized:
            return np.zeros(X.shape[0], dtype=int)
        return self.classifier.predict(X)