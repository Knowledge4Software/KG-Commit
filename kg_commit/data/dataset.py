from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import csv
from typing import List, Dict, Any
from kg_commit.core.commit import Commit
from kg_commit.core.project import Project


class CommitDataset(ABC):
    @abstractmethod
    def load(self) -> List[Commit]:
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        pass


class CSVCommitDataset(CommitDataset):
    def __init__(
        self,
        source: str | Path,
        label_column: str = "buggy",
        timestamp_column: str = "author_date",
    ):
        self.source = Path(source)
        self.label_column = label_column
        self.timestamp_column = timestamp_column
        self._metadata: Dict[str, Any] = {}

    def load(self) -> List[Commit]:
        if not self.source.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.source}")

        commits = []
        with open(self.source, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert buggy and fix to bool if present
                if 'buggy' in row:
                    row['buggy'] = row['buggy'].lower() == 'true'
                if 'fix' in row:
                    row['fix'] = row['fix'].lower() == 'true'
                # Convert year to int if present
                if 'year' in row and row['year']:
                    try:
                        row['year'] = int(row['year'])
                    except ValueError:
                        pass
                # Convert author_date to int if present
                if 'author_date' in row and row['author_date']:
                    try:
                        row['author_date'] = int(float(row['author_date']))
                    except ValueError:
                        pass
                commit = Commit.from_dict(row)
                commits.append(commit)

        # Sort by timestamp
        commits.sort(key=lambda c: c.timestamp or 0)

        self._metadata = {
            "rows": len(commits),
            "columns": list(commits[0].to_dict().keys()) if commits else [],
        }
        return commits

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)