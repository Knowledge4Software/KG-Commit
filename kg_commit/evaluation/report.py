from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd


class Report(ABC):
    """Abstract base class for generating reports."""

    @abstractmethod
    def build(self, evaluation_results: Dict[str, Any]) -> str:
        """Build a report from evaluation results."""
        pass


class SimpleReport(Report):
    """Simple text report."""

    def build(self, evaluation_results: Dict[str, Any]) -> str:
        """Build a simple text report."""
        report = "Evaluation Report\n"
        report += "=" * 20 + "\n"
        for key, value in evaluation_results.items():
            if isinstance(value, float):
                report += f"{key}: {value:.4f}\n"
            else:
                report += f"{key}: {value}\n"
        return report