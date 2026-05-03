from __future__ import annotations
from abc import ABC
from kg_commit.core.window import Window


class BaseModel(ABC):
    def fit(self, windows: list[Window]):
        raise NotImplementedError

    def update(self, window: Window):
        raise NotImplementedError

    def predict(self, window: Window):
        raise NotImplementedError