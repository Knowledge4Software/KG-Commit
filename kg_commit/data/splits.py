from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
from kg_commit.core.window import Window


class TimeWindowSplitter(ABC):
    """Abstract base class for splitting windows temporally."""

    @abstractmethod
    def split(self, windows: List[Window]) -> Tuple[List[Window], List[Window]]:
        """Split windows into train and test sets."""
        pass


class SimpleTimeWindowSplitter(TimeWindowSplitter):
    """Simple temporal split: first 80% train, last 20% test."""

    def split(self, windows: List[Window]) -> Tuple[List[Window], List[Window]]:
        """Perform temporal split."""
        split_idx = int(0.8 * len(windows))
        return windows[:split_idx], windows[split_idx:]