import os
import csv
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, List, Optional
from git import Repo, Commit


class BaseDatasetAdapter(ABC):
    """
    Abstract Base Class for converting specific dataset files (CSV, JSON, etc.)
    into a standardized metadata dictionary format.
    """
    
    @abstractmethod
    def stream_records(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Yields standard metadata dictionaries from the source file.
        Each yielded dict must consistently contain at least:
        ['commit_id', 'project', 'buggy', 'fix', 'year', 'author_date']
        """
        pass


class JITDatasetAdapter(BaseDatasetAdapter):
    """
    Adapter specifically tailored for the JIT dataset format containing columns like
    commit_id, project, buggy, fix, year, author_date, etc.
    """
    
    def __init__(self, target_columns: Optional[List[str]] = None):
        # Default columns we care about, but customizable if needed
        self.target_columns = target_columns or [
            "commit_id", "project", "buggy", "fix", "year", "author_date"
        ]

    def stream_records(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
            
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Basic validation to ensure the file has what we need
            if reader.fieldnames:
                missing = [col for col in self.target_columns if col not in reader.fieldnames]
                if missing:
                    raise KeyError(f"Source CSV missing required JIT columns: {missing}")
            
            for row in reader:
                # Filter row to keep only the target columns and yield
                yield {col: row[col] for col in self.target_columns}


class CommitDataLoader:
    """
    Handles loading raw commit details from local Git repositories
    and works with adapters to process tabular JIT metadata records.
    """

    def __init__(self, repo_map: Dict[str, str]):
        """
        Args:
            repo_map: Mapping of project names to local paths.
                      Example: {"apache/groovy": "E:\\repos\\groovy"}
        """
        self.repo_map = repo_map
        self._cached_repos: Dict[str, Repo] = {}

    def _get_repo(self, project_name: str) -> Repo:
        if project_name not in self._cached_repos:
            repo_path = self.repo_map.get(project_name)
            if not repo_path or not os.path.exists(repo_path):
                raise FileNotFoundError(
                    f"Repository path for '{project_name}' not found or invalid: {repo_path}"
                )
            self._cached_repos[project_name] = Repo(repo_path)
        return self._cached_repos[project_name]

    def fetch_commit_data(self, project: str, commit_id: str) -> Optional[Dict[str, Any]]:
        """Fetches commit textual information directly from Git."""
        try:
            repo = self._get_repo(project)
            commit: Commit = repo.commit(commit_id)
            
            commit_text = commit.message
            diff_text = ""
            if commit.parents:
                diff_index = commit.parents[0].diff(commit, create_patch=True)
                diff_text = "\n".join([d.diff.decode('utf-8', errors='ignore') for d in diff_index])

            return {
                "commit_id": commit_id,
                "project": project,
                "message": commit_text,
                "diff": diff_text,
                "author": commit.author.name,
                "authored_datetime": commit.authored_datetime
            }
            
        except Exception as e:
            print(f"Error fetching commit {commit_id} for {project}: {e}")
            return None