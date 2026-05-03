from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional
from kg_commit.core.window import Window
from .metrics import Metrics


class Evaluator:
    def __init__(self, metrics_computer: Metrics | None = None):
        self.results: List[tuple[Window, np.ndarray]] = []
        self.metrics_computer = metrics_computer or Metrics()

    def add(self, window: Window, preds: np.ndarray):
        """Add predictions for a window."""
        self.results.append((window, preds))

    def evaluate_window(self, window: Window) -> Dict[str, Any]:
        """Evaluate a single window."""
        if not window.has_predictions():
            raise ValueError("Window has no predictions")
        y_true = window.get_labels()
        y_pred = window.get_predictions()
        return self.metrics_computer.compute_all(y_true.tolist(), y_pred.tolist())

    def summarize(self) -> Dict[str, Any]:
        """Summarize all results."""
        if not self.results:
            return {"accuracy": 0.0, "report": "No predictions were made."}

        all_y_true = []
        all_y_pred = []
        for window, preds in self.results:
            all_y_true.extend(window.get_labels().tolist())
            all_y_pred.extend(preds.tolist())

        return self.metrics_computer.compute_all(all_y_true, all_y_pred)