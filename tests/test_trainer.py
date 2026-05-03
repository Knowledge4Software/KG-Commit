import numpy as np
from kg_commit.core.window import Window
from kg_commit.data.preprocess import SimplePreprocessor
from kg_commit.data.stream import CommitStream
from kg_commit.evaluation.evaluator import Evaluator
from kg_commit.training.incremental_trainer import IncrementalTrainer


class DummyModel:
    def __init__(self):
        self.updated_windows = []

    def predict(self, window):
        return np.zeros(window.size(), dtype=int)

    def update(self, window):
        self.updated_windows.append(window)


def test_incremental_trainer_runs_and_updates_model():
    commits = [
        {"commit_time": 0, "buggy": 0, "files_changed": 1},
        {"commit_time": 1, "buggy": 1, "files_changed": 2},
    ]
    stream = CommitStream(commits, window_size=1, step=1)
    trainer = IncrementalTrainer(model=DummyModel(), preprocessor=SimplePreprocessor(), evaluator=Evaluator())

    windows = list(trainer.run(stream))

    assert len(windows) == 2
    first_window, first_preds = windows[0]
    assert first_window.is_preprocessed()
    assert first_preds.tolist() == [0]
    assert windows[1][0].get_labels().tolist() == [1]
