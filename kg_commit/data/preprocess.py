from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional
from kg_commit.core.window import Window
from kg_commit.core.commit import Commit


class Preprocessor(ABC):
    @abstractmethod
    def transform_window(self, window: Window) -> Window:
        pass


class SimplePreprocessor(Preprocessor):
    def __init__(self, label_column: str = "buggy", feature_columns: Optional[List[str]] = None, include_text: bool = True):
        self.label_column = label_column
        self.feature_columns = feature_columns  # If None, auto-detect numerical features
        self.include_text = include_text

    def transform_window(self, window: Window) -> Window:
        if not window.commits:
            raise ValueError("Window has no commits to preprocess")

        # Check if commits have the label
        sample_commit = window.commits[0]
        if not hasattr(sample_commit, self.label_column) and self.label_column not in sample_commit.features:
            raise ValueError(f"Label column '{self.label_column}' not found in commits")

        # Determine feature columns
        if self.feature_columns is None:
            # Auto-detect: include all numerical features, and text if include_text
            feature_keys = []
            for key, value in sample_commit.features.items():
                if isinstance(value, (int, float)):
                    feature_keys.append(key)
                elif self.include_text and isinstance(value, str):
                    feature_keys.append(key)
            # Also include non-feature attributes if numerical
            if isinstance(getattr(sample_commit, 'year', None), (int, float)):
                feature_keys.append('year')
            if isinstance(getattr(sample_commit, 'author_date', None), (int, float)):
                feature_keys.append('author_date')
        else:
            feature_keys = self.feature_columns

        # Separate numerical and text features
        numerical_features = [k for k in feature_keys if self._is_numerical(window.commits, k)]
        text_features = [k for k in feature_keys if k not in numerical_features] if self.include_text else []

        # Prepare labels
        y = np.array([self._get_label(commit) for commit in window.commits], dtype=int)

        # Prepare features
        if numerical_features:
            X_num = np.array([[self._get_feature_value(commit, k) for k in numerical_features] for commit in window.commits], dtype=float)
        else:
            X_num = None

        # For text features, we can keep them as list of dicts or something, but for now, since ML models expect numerical, we'll skip text in X
        # But store text separately if needed
        text_data = [{k: self._get_feature_value(commit, k) for k in text_features} for commit in window.commits] if text_features else None

        # For simplicity, set X to numerical features only
        window.set_features(X_num)
        window.set_labels(y)
        # Store text data in metadata if present
        if text_data:
            window.metadata['text_features'] = text_data
            window.metadata['text_feature_keys'] = text_features

        return window

    def _is_numerical(self, commits: List[Commit], key: str) -> bool:
        """Check if a feature key is numerical across commits."""
        for commit in commits[:10]:  # Sample first 10
            value = self._get_feature_value(commit, key)
            if not isinstance(value, (int, float)):
                return False
        return True

    def _get_label(self, commit: Commit) -> int:
        """Get the label value from commit."""
        if hasattr(commit, self.label_column):
            value = getattr(commit, self.label_column)
        else:
            value = commit.features.get(self.label_column)
        if isinstance(value, bool):
            return int(value)
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            raise ValueError(f"Invalid label value: {value}")

    def _get_feature_value(self, commit: Commit, key: str) -> Any:
        """Get feature value from commit."""
        if hasattr(commit, key):
            return getattr(commit, key)
        else:
            return commit.features.get(key, 0.0)