from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseCommitParser(ABC):
    """Abstract Base Class defining the standard interface for all commit parsers."""

    @abstractmethod
    def parse(self, commit_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parses a commit payload and returns the processed entity dictionary."""
        pass

class IdentityCommitParser(BaseCommitParser):
    """
    A parser that performs an identity transformation, returning the raw 
    commit payload completely unfiltered and intact.
    """

    def parse(self, commit_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the incoming commit payload exactly as-is."""
        return commit_payload


class FilteredCommitParser(BaseCommitParser):
    """
    A parser that filters the commit payload dictionary down to a selective schema.
    Defaults to core metadata and file tracking arrays relevant to the knowledge graph.
    """

    def __init__(self, allowed_keys: Optional[List[str]] = None):
        """
        Args:
            allowed_keys: Optional custom list of keys. If None, uses the standard
                          JIT defect prediction base schema.
        """
        # Fallback to your default production schema if no list is passed
        self.allowed_keys = allowed_keys if allowed_keys is not None else [
            "commit_id",
            "project",
            "containing_branches",
            "author_name",
            "author_email",
            "committed_datetime",
            "files_added_list",
            "files_deleted_list",
            "files_modified_list",
            "files_renamed_list",
            "files_copied_list",
        ]

    def parse(self, commit_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Filters the incoming payload, returning only the configured schema fields."""
        return {
            key: commit_payload[key] 
            for key in self.allowed_keys 
            if key in commit_payload
        }