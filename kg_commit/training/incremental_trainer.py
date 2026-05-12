from __future__ import annotations
from typing import Iterator, Optional, Tuple
import numpy as np
from kg_commit.data.preprocess import Preprocessor
from kg_commit.data.stream import CommitStream
from kg_commit.evaluation.evaluator import Evaluator
from kg_commit.core.window import Window
from kg_commit.model.base_model import BaseModel


class IncrementalTrainer:
    def __init__(
        self,
        model: BaseModel,
        preprocessor: Preprocessor,
        evaluator: Evaluator,
        graph=None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.evaluator = evaluator
        self.graph = graph  # Optional CommitKnowledgeGraph

    def run(self, stream: CommitStream) -> Iterator[Tuple[Window, np.ndarray]]:
        for window in stream:
            window = self.preprocessor.transform_window(window)
            preds = self.model.predict(window)
            window.set_predictions(preds)
            self.evaluator.add(window, preds)
            self.model.update(window)
            if self.graph is not None:
                self.graph.update_from_window(window)
            yield window, preds