from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import pickle
from pathlib import Path
from kg_commit.core.window import Window


class Cache(ABC):
    """Abstract base class for caching."""

    @abstractmethod
    def store_window(self, window: Window, key: str):
        """Store a window in cache."""
        pass

    @abstractmethod
    def retrieve_window(self, key: str) -> Window | None:
        """Retrieve a window from cache."""
        pass


class SimpleCache(Cache):
    """Simple file-based cache."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def store_window(self, window: Window, key: str):
        """Store window as pickle."""
        with open(self.cache_dir / f"{key}.pkl", 'wb') as f:
            pickle.dump(window, f)

    def retrieve_window(self, key: str) -> Window | None:
        """Retrieve window from pickle."""
        path = self.cache_dir / f"{key}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None