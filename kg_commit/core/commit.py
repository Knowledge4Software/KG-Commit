from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime


class Commit:
    """
    Represents a single commit with its metadata and features.
    """

    def __init__(
        self,
        commit_id: str,
        project: str,
        buggy: Optional[bool] = None,
        fix: Optional[bool] = None,
        year: Optional[int] = None,
        author_date: Optional[int] = None,
        commit_text: Optional[str] = None,
        **features: Any
    ):
        self.commit_id = commit_id
        self.project = project
        self.buggy = buggy
        self.fix = fix
        self.year = year
        self.author_date = author_date
        self.commit_text = commit_text
        self.features: Dict[str, Any] = features
        self._timestamp: Optional[float] = None

    @property
    def timestamp(self) -> Optional[float]:
        """Get the timestamp for sorting, preferring author_date."""
        if self._timestamp is None and self.author_date is not None:
            self._timestamp = float(self.author_date)
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: float):
        self._timestamp = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            'commit_id': self.commit_id,
            'project': self.project,
            'buggy': self.buggy,
            'fix': self.fix,
            'year': self.year,
            'author_date': self.author_date,
            'commit_text': self.commit_text,
        }
        data.update(self.features)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Commit:
        """Create Commit from dictionary."""
        return cls(
            commit_id=data.get('commit_id', ''),
            project=data.get('project', ''),
            buggy=data.get('buggy'),
            fix=data.get('fix'),
            year=data.get('year'),
            author_date=data.get('author_date'),
            commit_text=data.get('commit_text'),
            **{k: v for k, v in data.items() if k not in ['commit_id', 'project', 'buggy', 'fix', 'year', 'author_date', 'commit_text']}
        )

    def __repr__(self) -> str:
        return f"Commit(id={self.commit_id}, project={self.project}, buggy={self.buggy})"