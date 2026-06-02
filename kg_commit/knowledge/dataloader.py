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
        Fetches commit data, topology, and structural metrics, filtered 
        exclusively for Java source code files (.java).
        """
        try:
            repo = self._get_repo(project)
            commit = repo.commit(commit_id)
            
            # Initialize storage for all file change types
            files_added, files_deleted, files_modified = [], [], []
            files_renamed_details, files_copied = [], []
            
            max_directory_depth = 0
            java_insertions = 0
            java_deletions = 0
            diff_text = ""

            # 1. Structural Analysis using Diff
            if commit.parents:
                diff_index = commit.parents[0].diff(commit, create_patch=True)
                diff_lines = []
                
                for d in diff_index:
                    # Resolve filepath (b_path is current, a_path is previous)
                    filepath = d.b_path if d.b_path else d.a_path
                    if not filepath or not filepath.lower().endswith('.java'):
                        continue
                    
                    # Capture Patch Diff Text
                    diff_lines.append(d.diff.decode('utf-8', errors='ignore'))
                    
                    # Categorize by change_type
                    if d.change_type == 'A': 
                        files_added.append(filepath)
                    elif d.change_type == 'D': 
                        files_deleted.append(filepath)
                    elif d.change_type == 'R': 
                        # Store as tuple (old_path, new_path) for renamed files
                        files_renamed_details.append((d.a_path, d.b_path))
                    elif d.change_type == 'C': 
                        files_copied.append(filepath)
                    else: 
                        files_modified.append(filepath) # Includes 'M'
                    
                    # Volumetric stats from reliable commit.stats
                    f_stats = commit.stats.files.get(filepath, {})
                    java_insertions += f_stats.get("insertions", 0)
                    java_deletions += f_stats.get("deletions", 0)
                    
                    depth = len(Path(filepath).parts) - 1
                    if depth > max_directory_depth:
                        max_directory_depth = depth
                
                diff_text = "\n".join(diff_lines)

            # Skip if no .java source blocks were affected
            total_java_files = (len(files_added) + len(files_deleted) + len(files_modified) + 
                            len(files_renamed_details) + len(files_copied))
            if total_java_files == 0:
                return None

            # 2. Issue Linking Context Extraction
            full_text_context = commit.message
            if hasattr(commit, 'trailers') and commit.trailers:
                full_text_context += "\n" + "\n".join(f"{k}: {v}" for k, v in commit.trailers.items())
                
            issue_pattern = re.compile(r'\b([A-Z]+-\d+|#\d+|GH-\d+)\b')
            linked_issues = sorted(list(set(issue_pattern.findall(full_text_context))))

            # 3. Branch Tracking
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
                "message": commit.message,
                "diff": diff_text,
                
                "parents": [p.hexsha for p in commit.parents],
                "parents_length": len(commit.parents),
                
                "linked_issues": linked_issues,
                "containing_branches": containing_branches,
                
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "authored_timestamp": commit.authored_date,
                "authored_datetime": commit.authored_datetime.isoformat(),
                "committer_name": commit.committer.name,
                "committed_datetime": commit.committed_datetime.isoformat(),
                
                # Java-Specific JIT Metrics
                "lines_added": java_insertions,
                "lines_deleted": java_deletions,
                "files_changed_count": total_java_files,
                
                "files_added_list": files_added,
                "files_deleted_list": files_deleted,
                "files_modified_list": files_modified,
                "files_renamed_details": files_renamed_details, # Contains (a_path, b_path)
                "files_copied_list": files_copied,
                
                "max_directory_depth": max_directory_depth,
                "commit_size_bytes": commit.size
            }
            
        except Exception as e:
            print(f"Error fetching commit {commit_id} for {project}: {e}")
            return None