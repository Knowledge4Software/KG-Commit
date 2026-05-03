from __future__ import annotations
from typing import List
from dataclasses import dataclass
from .commit import Commit


@dataclass
class Project:
    """
    Represents a software project with its commits.
    """
    name: str
    commits: List[Commit] = None

    def __post_init__(self):
        if self.commits is None:
            self.commits = []

    def add_commit(self, commit: Commit):
        """Add a commit to the project."""
        self.commits.append(commit)

    def get_commits(self) -> List[Commit]:
        """Get all commits."""
        return self.commits

    def sort_commits_by_time(self):
        """Sort commits by timestamp."""
        self.commits.sort(key=lambda c: c.timestamp or 0)

    def get_commit_by_id(self, commit_id: str) -> Commit | None:
        """Get a commit by its ID."""
        for commit in self.commits:
            if commit.commit_id == commit_id:
                return commit
        return None

    def __len__(self) -> int:
        return len(self.commits)