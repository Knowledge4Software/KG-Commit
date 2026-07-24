import os
import csv
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, List, Optional
from git import Repo, Commit
from pathlib import Path
import re
import datetime
import subprocess
        
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
    
    def fetch_all_commits_fast(self, project: str, limit: int = -1,
                               only_commits=None) -> Generator[Dict[str, Any], None, None]:
        """
        High-speed chronological commit streaming with complete inline code diff collection.
        Processes the Git stream entirely in byte space to guarantee immunity to encoding failures.

        only_commits : optional iterable of commit SHAs. When given, ONLY those
            commits are streamed (via `git log --no-walk`, still chronological
            via --reverse), instead of the repo's full history. Used by the
            multi-repository ingestion so a large secondary repo (e.g. the hadoop
            monorepo) contributes only its labelled commits, not all ~28k. See
            docs/Critical_notes.tex.
        """

        repo = self._get_repo(project)

        # 1. Map branch alignments up front using local branch tracking heads only
        commit_branch_map = {}
        for branch in repo.branches:
            try:
                for c in repo.iter_commits(branch.name):
                    commit_branch_map.setdefault(c.hexsha, set()).add(branch.name)
            except Exception:
                continue

        # 1. Update your formatting string to include %P right after %H
        delimiter = b"||--NEXT_COMMIT--||"
        log_format = "||--NEXT_COMMIT--||%H|%P|%aN|%aE|%cN|%at|%ct|%B"

        # Keep your command array exactly the same
        cmd = ["git", "-C", repo.working_dir, "log", "--reverse", f"--format={log_format}", "--numstat", "--summary", "-p"]
        # Windows fix: some repos (e.g. hbase) contain files with a ':' in their
        # name (HBASE-18070-ROOT_hbase:meta_Region_Replicas.pdf). ':' is illegal
        # in Windows paths, so `git log -p` cannot materialize that blob's
        # temp-file and dies with "fatal: unable to create temp-file: Invalid
        # argument", aborting the whole stream. These are binary files that carry
        # no signal for the KG (only .java text is parsed), so exclude any
        # colon-named path from the diff. Trailing pathspec must come last.
        colon_exclude = ["--", ".", ":(exclude,glob)**/*:*"]
        stdin_bytes = None
        if only_commits:
            # Restrict to an explicit SHA set: --no-walk stops history traversal
            # so we get exactly these commits (each still diffed vs its parent by
            # the -p/--numstat machinery). --reverse orders them chronologically.
            # The SHAs are passed via --stdin (one per line), NOT on the command
            # line: a large allow-list (e.g. 796 SHAs ~ 32 KB) exceeds Windows'
            # command-line length limit and raises WinError 206. See
            # docs/Critical_notes.tex.
            allow = [s for s in dict.fromkeys(only_commits)]   # de-dup, keep order
            cmd += ["--no-walk", "--stdin"]
            stdin_bytes = ("\n".join(allow) + "\n").encode("utf-8")
        if limit > 0:
            cmd.extend(["-n", str(limit)])
        cmd += colon_exclude   # trailing pathspec MUST be last (excludes ':'-named files)

        # NATIVE BYTES FIX: Directly stream stdout into bytes, completely bypassing encoding layers
        process = subprocess.run(cmd, input=stdin_bytes,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if process.returncode != 0:
            raise RuntimeError(f"Git command failed: {process.stderr.decode('utf-8', errors='replace')}")
            
        raw_log_stream_bytes = process.stdout
        raw_commits = raw_log_stream_bytes.split(delimiter)

        issue_pattern = re.compile(r'\b([A-Z]+-\d+|#\d+|GH-\d+)\b')

        for raw_block in raw_commits:
            if not raw_block.strip():
                continue
                
            lines = raw_block.strip().split(b"\n")
            header = lines[0].split(b"|")
            
            # Since we added %P, we now expect at least 8 elements in the header
            if len(header) < 8:
                continue
                
            commit_id = header[0].decode('utf-8', errors='replace')
            
            # UNPACK PARENTS: Extract space-separated parent hashes into a clean list of strings
            raw_parents = header[1].decode('utf-8', errors='replace').strip()
            parents_list = raw_parents.split(" ") if raw_parents else []
            
            # Shift your remaining indexes down by 1
            author_name = header[2].decode('utf-8', errors='replace')
            author_email = header[3].decode('utf-8', errors='replace')
            committer_name = header[4].decode('utf-8', errors='replace')
            
            try:
                authored_ts = int(header[5])
                committed_ts = int(header[6])
            except ValueError:
                continue

            authored_dt = datetime.datetime.fromtimestamp(authored_ts, datetime.timezone.utc).isoformat()
            committed_dt = datetime.datetime.fromtimestamp(committed_ts, datetime.timezone.utc).isoformat()

            # FIX: Message body comes strictly and exclusively from the %B token inside the header array
            message_body = header[7].decode('utf-8', errors='replace').strip()
            
            data_lines = []
            diff_lines = []
            
            # State tracker to separate file alterations metadata from raw patches/context lines
            in_diff = False
            
            for line in lines[1:]:
                # Once we cross into the diff block, everything trailing belongs to the patch
                if in_diff:
                    diff_lines.append(line)
                # Detect the line boundary indicating a patch block has begun
                elif line.startswith(b'diff --git'):
                    in_diff = True
                    diff_lines.append(line)
                # Everything else before the patch is strictly numstat or summary lines
                else:
                    # Ignore empty spacing or lone trailing newline noise padding
                    if line.strip():
                        data_lines.append(line.strip())
                    
            # Safely transform the patch byte array back into a real string 
            actual_diff_patch = b"\n".join(diff_lines).decode('utf-8', errors='replace').strip()

            # Tracking structures for files and line deltas
            files_added, files_deleted, files_modified = [], [], []
            files_renamed_list, files_copied_list = [], []
            
            java_insertions = 0
            java_deletions = 0
            max_directory_depth = 0
            
            # Sub-pass 1: Parse all raw file paths and line tallies from numstats
            numstat_map = {} 
            for line_bytes in data_lines:
                if b"\t" in line_bytes:
                    parts = line_bytes.split(b"\t")
                    if len(parts) < 3:
                        continue
                    # Safely store path references as text
                    path_str = parts[2].decode('utf-8', errors='replace')
                    numstat_map[path_str] = (parts[0].decode('utf-8'), parts[1].decode('utf-8'))

            # Convert our binary data lines into safely isolated string sequences for parsing
            data_strings = [line.decode('utf-8', errors='replace') for line in data_lines]

            # Sub-pass 2: Determine explicit Change Types using the summary strings
            processed_raw_paths = set()

            for line in data_strings:
                if line.startswith('create mode'):
                    filepath = line.split(' ', 3)[-1]
                    if filepath.lower().endswith('.java') and filepath in numstat_map:
                        files_added.append(filepath)
                        processed_raw_paths.add(filepath)

                elif line.startswith('delete mode'):
                    filepath = line.split(' ', 3)[-1]
                    if filepath.lower().endswith('.java') and filepath in numstat_map:
                        files_deleted.append(filepath)
                        processed_raw_paths.add(filepath)

                elif line.startswith('rename '):
                    raw_path_block = line.split(' ', 1)[1].rsplit(' (', 1)[0]
                    if raw_path_block in numstat_map:
                        processed_raw_paths.add(raw_path_block)
                        
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

                elif line.startswith('copy '):
                    raw_path_block = line.split(' ', 1)[1].rsplit(' (', 1)[0]
                    if raw_path_block in numstat_map:
                        processed_raw_paths.add(raw_path_block)
                        
                        if " => " in raw_path_block:
                            match = re.search(r'\{(.*?) => (.*?)\}', raw_path_block)
                            new_path = raw_path_block.replace(match.group(0), match.group(2)).replace("//", "/") if match else raw_path_block.split(" => ")[1]
                            if new_path.lower().endswith('.java'):
                                files_copied_list.append(new_path)

            # Sub-pass 3: Modifications ('M')
            for raw_filepath, (added_str, deleted_str) in numstat_map.items():
                if raw_filepath in processed_raw_paths:
                    continue 
                
                clean_path = raw_filepath
                if " => " in raw_filepath:
                    match = re.search(r'\{(.*?) => (.*?)\}', raw_filepath)
                    clean_path = raw_filepath.replace(match.group(0), match.group(2)).replace("//", "/") if match else raw_filepath.split(" => ")[1]

                if not clean_path.lower().endswith('.java'):
                    continue

                add_val = int(added_str) if added_str.isdigit() else 0
                del_val = int(deleted_str) if deleted_str.isdigit() else 0
                java_insertions += add_val
                java_deletions += del_val

                depth = len(Path(clean_path).parts) - 1
                if depth > max_directory_depth:
                    max_directory_depth = depth

                files_modified.append(clean_path)

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
                "diff": actual_diff_patch,
                
                "parents": parents_list, 
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
                "commit_size_bytes": len(actual_diff_patch.encode('utf-8'))
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