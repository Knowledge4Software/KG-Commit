from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from kg_commit.core.window import Window


class Evaluator:
    def __init__(self):
        self.results = []

    def add(self, window: Window, preds):
        self.results.append((window, preds))

    def summarize(self) -> dict:
        if not self.results:
            return {"accuracy": 0.0, "report": "No predictions were made."}

        y_true = np.concatenate([window.get_labels() for window, _ in self.results])
        y_pred = np.concatenate([np.asarray(preds) for _, preds in self.results])

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }