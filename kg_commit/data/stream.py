from __future__ import annotations
from typing import List, Optional
from kg_commit.core.window import Window
from kg_commit.core.commit import Commit


class CommitStream:
    def __init__(
        self,
        commits: List[Commit],
        window_size: int = 50,
        step: int = 50,
    ):
        self.commits = commits
        self.window_size = window_size
        self.step = step
        self.cursor = 0

    def next_window(self) -> Optional[Window]:
        if self.cursor >= len(self.commits):
            return None

        window_commits = self.commits[self.cursor : self.cursor + self.window_size]
        self.cursor += self.step

        start_time = window_commits[0].timestamp or self.cursor - self.window_size
        end_time = window_commits[-1].timestamp or self.cursor

        return Window(window_commits, start_time, end_time)

    def __iter__(self):
        while True:
            window = self.next_window()
            if window is None:
                break
            yield window