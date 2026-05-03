from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import matplotlib.pyplot as plt
from kg_commit.core.window import Window


class TimelineVisualizer(ABC):
    """Abstract base class for timeline visualizations."""

    @abstractmethod
    def plot(self, windows: List[Window], predictions: List[List[int]]):
        """Plot commit timeline with predictions."""
        pass


class SimpleTimelineVisualizer(TimelineVisualizer):
    """Simple timeline visualizer."""

    def plot(self, windows: List[Window], predictions: List[List[int]]):
        """Plot buggy commits over time."""
        if not windows or not predictions:
            print("No data to plot.")
            return

        times = []
        buggy_counts = []
        for window, preds in zip(windows, predictions):
            times.append(window.start_time)
            buggy_counts.append(sum(preds))

        plt.figure(figsize=(12, 6))
        plt.plot(times, buggy_counts, marker='o')
        plt.title('Buggy Commits Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Buggy Predictions')
        plt.grid(True)
        plt.show()