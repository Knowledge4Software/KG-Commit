from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from kg_commit.core.window import Window
from .base_model import BaseModel


class BatchModel(BaseModel):
    def __init__(self, classifier: BaseEstimator | None = None, **classifier_kwargs):
        if classifier is None:
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(max_iter=1000, **classifier_kwargs)
        else:
            self.classifier = classifier

    def fit(self, windows: list[Window]):
        if not windows:
            return

        X = np.vstack([window.get_features() for window in windows])
        y = np.concatenate([window.get_labels() for window in windows])
        self.classifier.fit(X, y)

    def update(self, window: Window):
        raise NotImplementedError("BatchModel does not support online updates")

    def predict(self, window: Window):
        return self.classifier.predict(window.get_features())