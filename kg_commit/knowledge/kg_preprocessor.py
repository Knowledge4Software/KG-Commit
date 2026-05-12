from __future__ import annotations
import numpy as np
from kg_commit.core.window import Window
from kg_commit.data.preprocess import SimplePreprocessor
from kg_commit.knowledge.graph import CommitKnowledgeGraph


class KGPreprocessor(SimplePreprocessor):
    """
    Extends SimplePreprocessor by appending graph-derived features to X.

    Graph features are read BEFORE prediction; the graph is updated
    by IncrementalTrainer AFTER model.update() to avoid label leakage.
    """

    def __init__(self, graph: CommitKnowledgeGraph, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph

    def transform_window(self, window: Window) -> Window:
        window = super().transform_window(window)

        kg_rows = np.array(
            [list(self.graph.get_features(c).values()) for c in window.commits],
            dtype=float,
        )

        if window.X is not None:
            window.set_features(np.hstack([window.X, kg_rows]))
        else:
            window.set_features(kg_rows)

        return window
