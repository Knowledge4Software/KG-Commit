from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from kg_commit.core.commit import Commit


class FeatureUtils(ABC):
    """Abstract base class for feature utilities."""

    @abstractmethod
    def compute_features(self, commit: Commit) -> Dict[str, Any]:
        """Compute additional features for a commit."""
        pass


class SimpleFeatureUtils(FeatureUtils):
    """Simple feature computation."""

    def compute_features(self, commit: Commit) -> Dict[str, Any]:
        """Compute basic features, e.g., text length if present."""
        features = {}
        if commit.commit_text:
            features['text_length'] = len(commit.commit_text)
        # Add more computations as needed
        return features