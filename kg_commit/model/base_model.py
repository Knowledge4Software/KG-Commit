from __future__ import annotations
from abc import ABC, abstractmethod
from kg_commit.core.window import Window


class BaseModel(ABC):
    @abstractmethod
    def fit(self, windows: list[Window]):
        pass

    @abstractmethod
    def update(self, window: Window):
        pass

    @abstractmethod
    def predict(self, window: Window):
        pass