"""
fetch_issues.py
---------------
Fetches JIRA issue texts for all commits in the ApacheJIT CSV.

Phase 1 – LOCAL git repos (no network):
    Read commit messages from cloned repos using GitPython.
    Extract JIRA issue IDs (e.g. GROOVY-1234) from each message.

Phase 2 – Apache JIRA REST API (network, but only unique issues):
    Fetch title + description for each unique issue ID.
    Typically a few thousand unique issues, not 106k API calls.

Both phases cache results to disk so interrupted runs resume cheaply.

Usage:
    python scripts/fetch_issues.py \\
        --csv       data/apachejit/apachejit_total.csv \\
        --repos     ./repos/apache/ \\
        --out       data/apachejit/issue_texts.csv \\
        --map       data/apachejit/commit_issue_map.csv \\
        --workers   4

Output:
    issue_texts.csv       – issue_id, title, description, status, created, updated
    commit_issue_map.csv  – commit_id, project, issue_id  (many-to-many)
"""

import os
import re
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from tqdm import tqdm
try:
    from git import Repo, InvalidGitRepositoryError, BadName
except ModuleNotFoundError:
    # GitPython is only needed for Phase 1 (local repos).
    # Phase 2 (JIRA API) runs without it.
    Repo = InvalidGitRepositoryError = BadName = None

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── JIRA key prefixes per GitHub project slug ─────────────────────────────────
JIRA_KEY_MAP = {
    "apache/groovy":           ["GROOVY"],
    "apache/activemq":         ["AMQ", "ACTIVEMQ"],
    "apache/camel":            ["CAMEL"],
    "apache/hbase":            ["HBASE"],
    "apache/zookeeper":        ["ZOOKEEPER"],
    "apache/hive":             ["HIVE"],
    "apache/cassandra":        ["CASSANDRA"],
    "apache/hadoop-hdfs":      ["HDFS", "HADOOP"],
    "apache/hadoop-mapreduce": ["MAPREDUCE", "HADOOP"],
    "apache/hadoop":           ["HADOOP"],
    "apache/flink":            ["FLINK"],
    "apache/spark":            ["SPARK"],
    "apache/kafka":            ["KAFKA"],
    "apache/zeppelin":         ["ZEPPELIN"],
    "apache/ignite":           ["IGNITE"],
}

_JIRA_RE  = re.compile(r'\b([A-Z][A-Z0-9]+-\d+)\b')
JIRA_BASE = "https://issues.apache.org/jira/rest/api/2"


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_json_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json_cache(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ── Phase 1: extract commit messages from local repos ─────────────────────────

def build_repo_mapping(repo_base: str) -> dict:
    """
    Scan repo_base for subdirectories that contain a .git folder.
    Returns dict mapping both 'apache/groovy' and 'groovy' → local path.
    """
    mapping = {}
    base = Path(repo_base)
    if not base.exists():
        log.error("Repo base '%s' does not exist.", repo_base)
        return mapping
    for folder in base.iterdir():
        if folder.is_dir() and (folder / ".git").is_dir():
            name = folder.name
            mapping[name]              = str(folder)
            mapping[f"apache/{name}"]  = str(folder)
    log.info("Found %d local repositories under %s", len(mapping) // 2, repo_base)
    return mapping


def extract_commit_messages_locally(df, repo_mapping, cache_path):
    """
    For each commit in df, read the commit message from the local git repo.
    Returns dict: commit_id -> message_str
    """
    cache   = _load_json_cache(cache_path)
    missing = df[~df.commit_id.isin(cache)].copy()
    log.info("Phase 1 (local): %d messages cached, %d to extract",
             len(cache), len(missing))

    if missing.empty:
        return cache

    repo_objects = {}   # path -> Repo instance (cached)
    save_counter = 0
    SAVE_EVERY   = 10_000

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="Reading commit messages"):
        project   = row["project"]
        commit_id = row["commit_id"]

        repo_path = repo_mapping.get(project) or repo_mapping.get(project.split("/")[-1])
        if not repo_path:
            cache[commit_id] = ""
            continue

        if repo_path not in repo_objects:
            try:
                repo_objects[repo_path] = Repo(repo_path)
            except InvalidGitRepositoryError:
                log.warning("Cannot open repo at %s", repo_path)
                cache[commit_id] = ""
                continue

        try:
            commit = repo_objects[repo_path].commit(commit_id)
            cache[commit_id] = commit.message or ""
        except (BadName, Exception):
            cache[commit_id] = ""

        save_counter += 1
        if save_counter >= SAVE_EVERY:
            _save_json_cache(cache_path, cache)
            save_counter = 0

    _save_json_cache(cache_path, cache)
    log.info("Phase 1 done. %d messages extracted.", len(cache))
    return cache


# ── Phase 1b: parse JIRA issue IDs from messages ─────────────────────────────

def extract_issue_ids(df, message_map):
    """
    Returns list of dicts {commit_id, project, issue_id}.
    Filters issue IDs to only those matching the project's known JIRA prefixes.
    """
    rows = []
    for _, row in df.iterrows():
        msg    = message_map.get(row["commit_id"], "")
        valid  = set(JIRA_KEY_MAP.get(row["project"], []))
        for m in _JIRA_RE.finditer(msg):
            key = m.group(1)
            if any(key.startswith(prefix) for prefix in valid):
                rows.append({
                    "commit_id": row["commit_id"],
                    "project":   row["project"],
                    "issue_id":  key,
                })
    return rows


# ── Phase 2: fetch JIRA issue details (network) ───────────────────────────────

def _get(url, session, retries=5, backoff=2.0):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", backoff * (2 ** attempt)))
                log.warning("HTTP %d – sleeping %.1fs", r.status_code, wait)
                time.sleep(wait)
            else:
                log.error("Unexpected HTTP %d for %s", r.status_code, url)
                return None
        except requests.RequestException as e:
            log.warning("Request error: %s – retry %d/%d", e, attempt + 1, retries)
            time.sleep(backoff * (2 ** attempt))
    log.error("Giving up on %s after %d attempts", url, retries)
    return None


def fetch_jira_issues(issue_ids, cache_path, workers=4):
    """
    Fetch title + description for each unique JIRA key.
    Returns dict: issue_key -> {title, description, status, created, updated}
    """
    cache   = _load_json_cache(cache_path)
    missing = [k for k in issue_ids if k not in cache]
    log.info("Phase 2 (JIRA API): %d issues cached, %d to fetch",
             len(cache), len(missing))

    if not missing:
        return cache

    session = requests.Session()
    session.headers["Accept"] = "application/json"

    lock       = threading.Lock()
    save_every = 200

    def fetch_one(key):
        url  = f"{JIRA_BASE}/issue/{key}?fields=summary,description,status,created,updated"
        data = _get(url, session)
        if data is None:
            return key, None
        fields = data.get("fields", {})
        return key, {
            "title":       fields.get("summary", ""),
            "description": fields.get("description", "") or "",
            "status":      (fields.get("status") or {}).get("name", ""),
            "created":     fields.get("created", ""),
            "updated":     fields.get("updated", ""),
        }

    with tqdm(total=len(missing), desc="JIRA issues") as bar:
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(fetch_one, k): k for k in missing}
            for i, fut in enumerate(as_completed(futures)):
                key, info = fut.result()
                with lock:
                    if info is not None:
                        cache[key] = info
                bar.update(1)
                if (i + 1) % save_every == 0:
                    with lock:
                        _save_json_cache(cache_path, dict(cache))

    _save_json_cache(cache_path, cache)
    return cache


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="data/apachejit/apachejit_total.csv")
    parser.add_argument("--repos",   default="./repos/apache/",
                        help="Root folder of cloned git repos")
    parser.add_argument("--out",     default="data/apachejit/issue_texts.csv")
    parser.add_argument("--map",     default="data/apachejit/commit_issue_map.csv")
    parser.add_argument("--cache",   default=".cache/fetch_issues")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel threads for JIRA API fetches")
    parser.add_argument("--project", default=None,
                        help="Limit to one project, e.g. apache/groovy")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                        help="1 = local git only (server, no network); "
                             "2 = JIRA API only (local, needs VPN); "
                             "all = both (default)")
    args = parser.parse_args()

    cache_dir = Path(args.cache)

    # ── Phase 1 (server side, no network) ─────────────────────────────────────
    if args.phase in ("1", "all"):
        df = pd.read_csv(args.csv)
        if args.project:
            df = df[df.project == args.project]
            log.info("Filtered to '%s': %d commits", args.project, len(df))
        else:
            log.info("Loaded %d commits across %d projects",
                     len(df), df.project.nunique())

        repo_mapping = build_repo_mapping(args.repos)
        msg_cache    = extract_commit_messages_locally(
            df, repo_mapping, cache_dir / "commit_messages.json"
        )

        mapping_rows = extract_issue_ids(df, msg_cache)
        mapping_df   = pd.DataFrame(mapping_rows)

        if mapping_df.empty:
            log.warning("No JIRA issue IDs found. Check that repos are cloned.")
            return

        log.info("Found %d unique JIRA issues across %d commits",
                 mapping_df.issue_id.nunique(), mapping_df.commit_id.nunique())

        Path(args.map).parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(args.map, index=False)
        log.info("Commit–issue map → %s", args.map)

        if args.phase == "1":
            log.info("Phase 1 complete. Download '%s' and run --phase 2 locally.",
                     args.map)
            return

    # ── Phase 2 (local side, needs VPN) ───────────────────────────────────────
    # Re-read the map so phase 2 can run standalone from just the CSV.
    mapping_df    = pd.read_csv(args.map)
    unique_issues = mapping_df.issue_id.unique()
    log.info("Phase 2: %d unique JIRA issues to fetch", len(unique_issues))

    issue_data = fetch_jira_issues(
        unique_issues, cache_dir / "jira_issues.json", workers=args.workers
    )

    # Write output
    out_rows = [
        {
            "issue_id":    key,
            "jira_key":    key,
            "jira_project": key.split("-")[0],
            "title":       info["title"],
            "description": info["description"],
            "status":      info["status"],
            "created":     info["created"],
            "updated":     info["updated"],
        }
        for key, info in issue_data.items()
    ]
    out_df = pd.DataFrame(out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    log.info("Issue texts → %s  (%d issues)", args.out, len(out_df))


if __name__ == "__main__":
    main()
