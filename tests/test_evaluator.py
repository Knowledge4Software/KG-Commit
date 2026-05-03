import numpy as np
from kg_commit.core.window import Window
from kg_commit.evaluation.evaluator import Evaluator


def test_evaluator_computes_accuracy_and_report():
    window = Window([
        {"buggy": 0, "files_changed": 2},
        {"buggy": 1, "files_changed": 4},
    ], start_time=0, end_time=1)

    window.set_features(np.array([[2.0], [4.0]]))
    window.set_labels(np.array([0, 1]))

    evaluator = Evaluator()
    evaluator.add(window, np.array([0, 1]))
    summary = evaluator.summarize()

    assert summary["accuracy"] == 1.0
    assert "precision" in summary["report"]


def test_evaluator_returns_zero_when_no_results():
    evaluator = Evaluator()
    summary = evaluator.summarize()

    assert summary["accuracy"] == 0.0
    assert "No predictions were made." in summary["report"]
