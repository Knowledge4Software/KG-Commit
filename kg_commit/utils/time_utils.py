from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from kg_commit.core.commit import Commit
from kg_commit.core.window import Window
from kg_commit.data.stream import CommitStream


class TimeUtils(ABC):
    """Abstract base class for time-related utilities."""

    @abstractmethod
    def sort_commits(self, commits: List[Commit]) -> List[Commit]:
        """Sort commits by timestamp."""
        pass

    @abstractmethod
    def build_windows(self, commits: List[Commit], window_size: int) -> List[Window]:
        """Build windows from commits."""
        pass


class SimpleTimeUtils(TimeUtils):
    """Simple implementation of time utilities."""

    def sort_commits(self, commits: List[Commit]) -> List[Commit]:
        """Sort commits by timestamp."""
        return sorted(commits, key=lambda c: c.timestamp or 0)

    def build_windows(self, commits: List[Commit], window_size: int) -> List[Window]:
        """Build sliding windows."""
        stream = CommitStream(commits, window_size=window_size, step=window_size)
        return list(stream)