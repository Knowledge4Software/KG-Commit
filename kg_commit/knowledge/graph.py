from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
from kg_commit.core.commit import Commit
from kg_commit.core.window import Window


class CommitKnowledgeGraph:
    """
    Incremental knowledge graph over commits.

    Nodes: Commit, Project
    Edges: (Commit) --in_project--> (Project)
           (Commit_i) --precedes--> (Commit_j)  within the same project

    Updated AFTER prediction to avoid label leakage.
    """

    def __init__(self, recent_window: int = 10):
        self.recent_window = recent_window
        # Per-project ordered history: list of (commit_id, buggy, la, ld)
        self._project_history: Dict[str, List[dict]] = defaultdict(list)

    def get_features(self, commit: Commit) -> Dict[str, float]:
        """Return graph-derived features for a commit. Call BEFORE prediction."""
        history = self._project_history[commit.project]
        n = len(history)

        if n == 0:
            return {
                "kg_project_commit_count": 0.0,
                "kg_project_bug_rate": 0.0,
                "kg_project_recent_bug_rate": 0.0,
                "kg_project_avg_la": 0.0,
                "kg_project_avg_ld": 0.0,
            }

        bug_rate = sum(1 for h in history if h["buggy"]) / n
        recent = history[-self.recent_window:]
        recent_bug_rate = sum(1 for h in recent if h["buggy"]) / len(recent)
        avg_la = sum(h["la"] for h in history) / n
        avg_ld = sum(h["ld"] for h in history) / n

        return {
            "kg_project_commit_count": float(n),
            "kg_project_bug_rate": bug_rate,
            "kg_project_recent_bug_rate": recent_bug_rate,
            "kg_project_avg_la": avg_la,
            "kg_project_avg_ld": avg_ld,
        }

    def add_commit(self, commit: Commit) -> None:
        """Record a commit into the graph. Call AFTER prediction."""
        self._project_history[commit.project].append({
            "commit_id": commit.commit_id,
            "buggy": bool(commit.buggy),
            "la": float(commit.features.get("la", 0)),
            "ld": float(commit.features.get("ld", 0)),
        })

    def update_from_window(self, window: Window) -> None:
        """Add all commits in a window to the graph. Call AFTER model.update()."""
        for commit in window.commits:
            self.add_commit(commit)