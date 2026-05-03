from __future__ import annotations
from kg_commit.model.base_model import BaseModel
from kg_commit.core.window import Window


class Trainer:

    def __init__(self, model: BaseModel):
        self.model = model

    def train(self, train_windows: list[Window]):
        self.model.fit(train_windows)