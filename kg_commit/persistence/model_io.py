from __future__ import annotations
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
from kg_commit.model.base_model import BaseModel


class ModelIO(ABC):
    """Abstract base class for model I/O."""

    @abstractmethod
    def save_model(self, model: BaseModel, path: str):
        """Save model to file."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> BaseModel:
        """Load model from file."""
        pass


class SimpleModelIO(ModelIO):
    """Simple pickle-based model I/O."""

    def save_model(self, model: BaseModel, path: str):
        """Save model."""
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, path: str) -> BaseModel:
        """Load model."""
        with open(path, 'rb') as f:
            return pickle.load(f)