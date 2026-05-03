from __future__ import annotations
from abc import ABC, abstractmethod
import json
from typing import Any


class Serializer(ABC):
    """Abstract base class for serialization."""

    @abstractmethod
    def save(self, obj: Any, path: str):
        """Serialize and save object."""
        pass

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load and deserialize object."""
        pass


class JSONSerializer(Serializer):
    """JSON-based serializer."""

    def save(self, obj: Any, path: str):
        """Save as JSON."""
        with open(path, 'w') as f:
            json.dump(obj, f, default=str)

    def load(self, path: str) -> Any:
        """Load from JSON."""
        with open(path, 'r') as f:
            return json.load(f)