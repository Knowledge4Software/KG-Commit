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
    
    def fetch_all_commits_fast(self, project: str, limit: int = -1) -> Generator[Dict[str, Any], None, None]:
        """
        High-speed chronological commit streaming. Explodes and categorizes file 
        changes by explicit Git status types (A, D, R, C, M) using numstat + summary logs.
        """
        import datetime
        repo = self._get_repo(project)
        
        # 1. Map branch alignments up front using local branch tracking heads only
        commit_branch_map = {}
        for branch in repo.branches:
            try:
                for c in repo.iter_commits(branch.name):
                    commit_branch_map.setdefault(c.hexsha, set()).add(branch.name)
            except Exception:
                continue

        # 2. Extract logs with 7-token metadata header format AND summary stats
        delimiter = "||--NEXT_COMMIT--||"
        log_format = f"{delimiter}%H|%aN|%aE|%cN|%at|%ct|%B"
        
        # Adding --summary gives us explicit 'create mode', 'delete mode', 'rename', and 'copy' lines
        args = ["--reverse", f"--format={log_format}", "--numstat", "--summary"]
        if limit > 0:
            args.append(f"-n {limit}")
            
        raw_log_stream = repo.git.log(*args)
        raw_commits = raw_log_stream.split(delimiter)

        issue_pattern = re.compile(r'\b([A-Z]+-\d+|#\d+|GH-\d+)\b')

        for raw_block in raw_commits:
            if not raw_block.strip():
                continue
                
            lines = raw_block.strip().split("\n")
            header = lines[0].split("|")
            
            if len(header) < 7:
                continue
                
            commit_id = header[0]
            author_name = header[1]
            author_email = header[2]
            committer_name = header[3]
            authored_ts = int(header[4])
            committed_ts = int(header[5])
            
            authored_dt = datetime.datetime.fromtimestamp(authored_ts, datetime.timezone.utc).isoformat()
            committed_dt = datetime.datetime.fromtimestamp(committed_ts, datetime.timezone.utc).isoformat()

            # Separate message body paragraphs from the log data blocks
            message_lines = [header[6]]
            data_lines = []
            
            for line in lines[1:]:
                # Data lines are either tab-delimited numstats or space-delimited summaries
                if "\t" in line or line.strip().startswith(('create mode', 'delete mode', 'rename', 'copy')):
                    data_lines.append(line.strip())
                else:
                    message_lines.append(line)
                    
            message_body = "\n".join(message_lines).strip()

            # Tracking structures for files and line deltas
            files_added, files_deleted, files_modified = [], [], []
            files_renamed_list, files_copied_list = [], []
            
            java_insertions = 0
            java_deletions = 0
            max_directory_depth = 0
            
            # Sub-pass 1: Parse all raw file paths and line tallies from numstats
            numstat_map = {} # Maps target path -> (added, deleted)
            for line in data_lines:
                if "\t" in line:
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    numstat_map[parts[2]] = (parts[0], parts[1])

            # Sub-pass 2: Determine explicit Change Types using the summary strings
            # We track processed files to know who is left over as a Modification ('M')
            processed_raw_paths = set()

            for line in data_lines:
                # Catch explicit Create ('A') Action
                if line.startswith('create mode'):
                    # format: "create mode 100644 path/to/file.java"
                    filepath = line.split(' ', 3)[-1]
                    if filepath.lower().endswith('.java') and filepath in numstat_map:
                        files_added.append(filepath)
                        processed_raw_paths.add(filepath)

                # Catch explicit Delete ('D') Action
                elif line.startswith('delete mode'):
                    # format: "delete mode 100644 path/to/file.java"
                    filepath = line.split(' ', 3)[-1]
                    if filepath.lower().endswith('.java') and filepath in numstat_map:
                        files_deleted.append(filepath)
                        processed_raw_paths.add(filepath)

                # Catch explicit Rename ('R') Action
                elif line.startswith('rename '):
                    # format: "rename old_path => new_path (90%)"
                    raw_path_block = line.split(' ', 1)[1].rsplit(' (', 1)[0]
                    if raw_path_block in numstat_map:
                        processed_raw_paths.add(raw_path_block)
                        
                        # Unpack git's brace notation if renaming inside the same tree package
                        if " => " in raw_path_block:
                            match = re.search(r'\{(.*?) => (.*?)\}', raw_path_block)
                            if match:
                                old_part, new_part = match.group(1), match.group(2)
                                old_path = raw_path_block.replace(match.group(0), old_part).replace("//", "/")
                                new_path = raw_path_block.replace(match.group(0), new_part).replace("//", "/")
                            else:
                                r_parts = raw_path_block.split(" => ")
                                old_path, new_path = r_parts[0], r_parts[1]
                            
                            if new_path.lower().endswith('.java'):
                                files_renamed_list.append((old_path, new_path))

                # Catch explicit Copy ('C') Action
                elif line.startswith('copy '):
                    # format: "copy old_path => new_path (90%)"
                    raw_path_block = line.split(' ', 1)[1].rsplit(' (', 1)[0]
                    if raw_path_block in numstat_map:
                        processed_raw_paths.add(raw_path_block)
                        
                        if " => " in raw_path_block:
                            match = re.search(r'\{(.*?) => (.*?)\}', raw_path_block)
                            new_path = raw_path_block.replace(match.group(0), match.group(2)).replace("//", "/") if match else raw_path_block.split(" => ")[1]
                            if new_path.lower().endswith('.java'):
                                files_copied_list.append(new_path)

            # Sub-pass 3: Everything left inside our numstat map is a pure Modification ('M')
            for raw_filepath, (added_str, deleted_str) in numstat_map.items():
                if raw_filepath in processed_raw_paths:
                    continue # Already categorized by summary rules above
                
                # Check extension on the raw_filepath (or unpack if it's an inline modification/rename Git compressed)
                clean_path = raw_filepath
                if " => " in raw_filepath:
                    match = re.search(r'\{(.*?) => (.*?)\}', raw_filepath)
                    clean_path = raw_filepath.replace(match.group(0), match.group(2)).replace("//", "/") if match else raw_filepath.split(" => ")[1]

                if not clean_path.lower().endswith('.java'):
                    continue

                # Add up lines metrics
                add_val = int(added_str) if added_str.isdigit() else 0
                del_val = int(deleted_str) if deleted_str.isdigit() else 0
                java_insertions += add_val
                java_deletions += del_val

                # Track tree depth metrics
                depth = len(Path(clean_path).parts) - 1
                if depth > max_directory_depth:
                    max_directory_depth = depth

                # If it wasn't explicitly tagged as A, D, R, or C by the summary, it is a Modification ('M')
                files_modified.append(clean_path)

            # Sum total java files touched in this block
            total_java_files = (len(files_added) + len(files_deleted) + len(files_modified) + 
                                len(files_renamed_list) + len(files_copied_list))
            
            if total_java_files == 0:
                continue

            linked_issues = sorted(list(set(issue_pattern.findall(message_body))))
            containing_branches = list(commit_branch_map.get(commit_id, ["main"]))

            yield {
                "commit_id": commit_id,
                "project": project,
                "message": message_body,
                "diff": "", 
                
                "parents": [], 
                "parents_length": 0,
                
                "linked_issues": linked_issues,
                "containing_branches": containing_branches,
                
                "author_name": author_name,
                "author_email": author_email,
                "authored_timestamp": authored_ts,
                "authored_datetime": authored_dt, 
                "committer_name": committer_name,
                "committed_datetime": committed_dt, 
                
                "lines_added": java_insertions,
                "lines_deleted": java_deletions,
                "files_changed_count": total_java_files,
                
                "files_added_list": files_added,
                "files_deleted_list": files_deleted,
                "files_modified_list": files_modified,
                "files_renamed_list": files_renamed_list,
                "files_copied_list": files_copied_list,
                
                "max_directory_depth": max_directory_depth,
                "commit_size_bytes": 0
            }
    
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
            files_renamed_list, files_copied = [], []
            
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
                        files_renamed_list.append((d.a_path, d.b_path))
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
                            len(files_renamed_list) + len(files_copied))
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
                "files_renamed_list": files_renamed_list, # Contains (a_path, b_path)
                "files_copied_list": files_copied,
                
                "max_directory_depth": max_directory_depth,
                "commit_size_bytes": commit.size
            }
            
        except Exception as e:
            print(f"Error fetching commit {commit_id} for {project}: {e}")
            return None