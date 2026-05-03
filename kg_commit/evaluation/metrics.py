from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class Metrics:
    """Class for computing evaluation metrics."""

    @staticmethod
    def accuracy(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate accuracy."""
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def precision(y_true: List[int], y_pred: List[int], average: str = 'binary') -> float:
        """Calculate precision."""
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))

    @staticmethod
    def recall(y_true: List[int], y_pred: List[int], average: str = 'binary') -> float:
        """Calculate recall."""
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))

    @staticmethod
    def f1(y_true: List[int], y_pred: List[int], average: str = 'binary') -> float:
        """Calculate F1 score."""
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))

    @staticmethod
    def classification_report_str(y_true: List[int], y_pred: List[int]) -> str:
        """Get classification report as string."""
        return classification_report(y_true, y_pred, zero_division=0)

    @classmethod
    def compute_all(cls, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        """Compute all metrics."""
        return {
            'accuracy': cls.accuracy(y_true, y_pred),
            'precision': cls.precision(y_true, y_pred),
            'recall': cls.recall(y_true, y_pred),
            'f1': cls.f1(y_true, y_pred),
            'report': cls.classification_report_str(y_true, y_pred),
        }