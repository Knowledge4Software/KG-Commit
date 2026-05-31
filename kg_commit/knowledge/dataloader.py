import os
import csv
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, List, Optional
from git import Repo, Commit
from pathlib import Path
import re


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
        """
        Fetches an exhaustive profile of commit data, topology, structural metrics,
        all unique linked issues found in text/trailers, and active branches.
        """
        try:
            repo = self._get_repo(project)
            commit = repo.commit(commit_id)
            
            # 1. Gather text representations
            commit_text = commit.message
            
            diff_text = ""
            if commit.parents:
                diff_index = commit.parents[0].diff(commit, create_patch=True)
                diff_text = "\n".join([d.diff.decode('utf-8', errors='ignore') for d in diff_index])

            # 2. Extract modification stats
            stats = commit.stats
            files_added, files_deleted, files_modified = [], [], []
            file_extensions = set()
            max_directory_depth = 0

            for filepath, f_stats in stats.files.items():
                ext = os.path.splitext(filepath)[1].lower()
                if ext:
                    file_extensions.add(ext)
                
                depth = len(Path(filepath).parts) - 1
                if depth > max_directory_depth:
                    max_directory_depth = depth

                if f_stats.get("deletions") == 0 and f_stats.get("insertions") > 0:
                    files_added.append(filepath)
                elif f_stats.get("insertions") == 0 and f_stats.get("deletions") > 0:
                    files_deleted.append(filepath)
                else:
                    files_modified.append(filepath)

            # 3. Comprehensive Issue Linking Extraction
            # Merge the main message with metadata trailers (e.g., 'Fixes: #102') if they exist
            full_text_context = commit_text
            if hasattr(commit, 'trailers') and commit.trailers:
                full_text_context += "\n" + "\n".join(f"{k}: {v}" for k, v in commit.trailers.items())
                
            # Match patterns like GROOVY-3294, #123, or GH-55 using your clean pattern
            issue_pattern = re.compile(r'\b([A-Z]+-\d+|#\d+|GH-\d+)\b')
            
            # Find all unique keys, deduplicate via set, and sort them deterministically
            linked_issues = sorted(list(set(issue_pattern.findall(full_text_context))))

            # 4. Tracking the branches containing this commit
            try:
                containing_branches = [
                    b.strip().replace("* ", "") 
                    for b in repo.git.branch("--contains", commit_id).split("\n") 
                    if b.strip()
                ]
            except Exception:
                containing_branches = []

            return {
                "commit_id": commit_id,
                "project": project,
                "message": commit_text,
                "diff": diff_text,
                
                # Refactored Lineage Tuple Length
                "parents": [p.hexsha for p in commit.parents],
                "parents_length": len(commit.parents),
                
                # Context Extractions (Now returns a clean list of strings or an empty list)
                "linked_issues": linked_issues,
                "containing_branches": containing_branches,
                
                # Identity & Timestamps
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "authored_timestamp": commit.authored_date,
                "authored_datetime": commit.authored_datetime.isoformat(),
                "committer_name": commit.committer.name,
                "committed_datetime": commit.committed_datetime.isoformat(),
                
                # JIT Volumetric Data
                "lines_added": stats.total.get("insertions", 0),
                "lines_deleted": stats.total.get("deletions", 0),
                "files_changed_count": stats.total.get("files", 0),
                
                # Structural Analysis arrays
                "files_added_list": files_added,
                "files_deleted_list": files_deleted,
                "files_modified_list": files_modified,
                "modified_extensions": list(file_extensions),
                "max_directory_depth": max_directory_depth,
                "commit_size_bytes": commit.size
            }
            
        except Exception as e:
            print(f"Error fetching commit {commit_id} for {project}: {e}")
            return None