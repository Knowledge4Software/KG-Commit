from __future__ import annotations
from kg_commit.data.preprocess import Preprocessor
from kg_commit.data.stream import CommitStream
from kg_commit.evaluation.evaluator import Evaluator


class IncrementalTrainer:
    def __init__(
        self,
        model,
        preprocessor: Preprocessor,
        evaluator: Evaluator,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.evaluator = evaluator

    def run(self, stream: CommitStream):
        while True:
            window = stream.next_window()
            if window is None:
                break

            window = self.preprocessor.transform_window(window)
            preds = self.model.predict(window)
            window.set_predictions(preds)
            self.evaluator.add(window, preds)
            self.model.update(window)
            yield window, preds