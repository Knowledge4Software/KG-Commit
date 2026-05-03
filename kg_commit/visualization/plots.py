from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from kg_commit.core.window import Window


class Plotter(ABC):
    """Abstract base class for plotting utilities."""

    @abstractmethod
    def plot_over_time(self, results: List[Dict[str, Any]]):
        """Plot metrics over time."""
        pass

    @abstractmethod
    def plot_predictions(self, windows: List[Window]):
        """Plot predictions for windows."""
        pass


class SimplePlotter(Plotter):
    """Simple implementation of Plotter using matplotlib."""

    def plot_over_time(self, results: List[Dict[str, Any]]):
        """Plot accuracy over time."""
        if not results:
            print("No results to plot.")
            return

        accuracies = [r.get('accuracy', 0) for r in results]
        plt.figure(figsize=(10, 5))
        plt.plot(accuracies, marker='o')
        plt.title('Accuracy Over Time')
        plt.xlabel('Window Index')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

    def plot_predictions(self, windows: List[Window]):
        """Plot prediction distributions."""
        if not windows:
            print("No windows to plot.")
            return

        pred_counts = []
        for window in windows:
            if window.has_predictions():
                preds = window.get_predictions()
                pred_counts.append((sum(preds == 0), sum(preds == 1)))

        if pred_counts:
            non_buggy = [c[0] for c in pred_counts]
            buggy = [c[1] for c in pred_counts]
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(non_buggy)), non_buggy, label='Non-buggy')
            plt.bar(range(len(buggy)), buggy, bottom=non_buggy, label='Buggy')
            plt.title('Prediction Distribution Over Windows')
            plt.xlabel('Window Index')
            plt.ylabel('Count')
            plt.legend()
            plt.show()