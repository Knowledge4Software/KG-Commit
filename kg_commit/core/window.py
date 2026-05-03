from __future__ import annotations
from typing import List, Dict, Any, Optional
from kg_commit.core.commit import Commit


class Window:
    """
    Core abstraction representing a time-bounded batch of commits.

    This is the ONLY object passed across the pipeline.
    """

    def __init__(self, commits: List[Commit], start_time: float, end_time: float, metadata: Optional[Dict[str, Any]] = None):
        self.commits = commits
        self.start_time = start_time
        self.end_time = end_time
        self.metadata = metadata or {}

        # ML-related fields
        self.X = None
        self.y = None
        self.predictions = None
        self.probabilities = None

    def is_preprocessed(self) -> bool:
        return self.X is not None and self.y is not None

    def has_predictions(self) -> bool:
        return self.predictions is not None

    def set_features(self, X):
        self.X = X

    def set_labels(self, y):
        self.y = y

    def set_predictions(self, preds, probs=None):
        self.predictions = preds
        self.probabilities = probs

    def get_features(self):
        if self.X is None:
            raise ValueError("Window has no features. Run Preprocessor first.")
        return self.X

    def __len__(self) -> int:
        return len(self.commits)

    def size(self) -> int:
        return len(self.commits)

    def get_predictions(self):
        if self.predictions is None:
            raise ValueError("No predictions available.")
        return self.predictions

    def size(self) -> int:
        return len(self.commits)

    def time_span(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (
            f"Window(size={self.size()}, "
            f"time=({self.start_time} → {self.end_time}), "
            f"preprocessed={self.is_preprocessed()})"
        )