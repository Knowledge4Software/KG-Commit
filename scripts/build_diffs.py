"""
build_diffs.py
--------------
Rebuild the `diff_text` column for every commit in the ApacheJIT CSV using
ONLY locally cloned git repositories — no GitHub / network calls.

Why this exists
---------------
The old extraction cell appended only `d.diff` (the hunk body). With
`create_patch=True`, GitPython keeps the file path in the *Diff object's
attributes* (`d.a_path` / `d.b_path` / `d.new_file` …), NOT inside `d.diff`.
So every `diff --git` / `--- a/` / `+++ b/` header was dropped, leaving
hunk-only text where `parse_files` returns nothing and the FILE / DIR /
AUTHOR tiers of the knowledge graph collapse.

This script keeps the SAME logic as that working cell (same repo discovery,
same `parent.diff(commit, create_patch=True)` direction, same in-memory
DataFrame + periodic save + resume-by-empty-cell), with one fix: it
reconstructs the headers from the path attributes and prepends a synthetic
`git show` header so author + message are recoverable too.

    commit <sha>
    Author: <name> <email>
    Date:   <iso date>

        <commit message, indented 4 spaces>

    --- a/path/File.java
    +++ b/path/File.java
    @@ -.. +.. @@ <section heading>
    ...hunks...

Resumable: rows whose `diff_text` is already non-empty are skipped, so an
interrupted run continues cheaply (re-run the same command).

Usage (server, in a notebook):
    import sys, subprocess
    subprocess.run([sys.executable, "-u", "scripts/build_diffs.py",
                    "--csv",   "data/apachejit/apachejit_total.csv",
                    "--repos", "./repos/apache/",
                    "--out",   "data/apachejit/apachejit_with_diffs_rebuilt.csv"])
"""

import os
import argparse
import logging

import pandas as pd
from tqdm import tqdm
from git import Repo, NULL_TREE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------- 1. Build dynamic project → path mapping ----------
def build_repo_mapping(local_repo_base: str) -> dict:
    """
    Look inside local_repo_base for folders that contain a .git directory.
    Maps both 'apache/groovy' and plain 'groovy' → local path.
    (Same discovery logic as the original working cell.)
    """
    repo_mapping = {}
    if os.path.exists(local_repo_base):
        for folder in sorted(os.listdir(local_repo_base)):
            full_path = os.path.join(local_repo_base, folder)
            if os.path.isdir(full_path) and \
               os.path.isdir(os.path.join(full_path, ".git")):
                repo_mapping[folder] = full_path
                repo_mapping[f"apache/{folder}"] = full_path
    log.info("Found %d local repositories under %s",
             len(repo_mapping) // 2, local_repo_base)
    return repo_mapping


# ---------- 2. Helper: compute FULL diff for a commit locally ----------
def _git_show_header(commit) -> str:
    """Synthetic 'git show' header so parse_author / parse_issues work."""
    msg = (commit.message or "").rstrip()
    indented = "\n".join("    " + ln for ln in msg.splitlines())
    return (
        f"commit {commit.hexsha}\n"
        f"Author: {commit.author.name} <{commit.author.email}>\n"
        f"Date:   {commit.authored_datetime.isoformat()}\n"
        f"\n{indented}\n\n"
    )


def compute_diff(repo, commit_sha: str) -> str:
    """
    Unified diff of a commit vs its first parent (or the empty tree for a
    root commit), WITH `--- a/ +++ b/` headers + a git-show header.

    Same direction as the original cell — parent.diff(commit) — just with
    the path attributes (dropped before) written back into the text.
    """
    try:
        commit = repo.commit(commit_sha)
    except Exception:                              # noqa: BLE001
        return ""

    parent = commit.parents[0] if commit.parents else NULL_TREE
    try:
        diffs = parent.diff(commit, create_patch=True)
    except Exception:                              # noqa: BLE001
        return _git_show_header(commit)

    parts = []
    for d in diffs:
        a = d.a_path or "/dev/null"
        b = d.b_path or "/dev/null"
        try:
            body = d.diff.decode("utf-8", "ignore") if d.diff else ""
        except Exception:                          # noqa: BLE001
            body = str(d.diff)
        parts.append(f"--- a/{a}\n+++ b/{b}\n{body}")

    return _git_show_header(commit) + "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   default="data/apachejit/apachejit_total.csv",
                    help="Input CSV (commit_id, project, … features)")
    ap.add_argument("--repos", default="./repos/apache/",
                    help="Root folder of cloned repos")
    ap.add_argument("--out",
                    default="data/apachejit/apachejit_with_diffs_rebuilt.csv",
                    help="Output CSV (all input columns + diff_text)")
    ap.add_argument("--save-every", type=int, default=10000,
                    help="Flush to disk after this many new diffs")
    ap.add_argument("--project", default=None,
                    help="Limit to one project, e.g. apache/groovy")
    args = ap.parse_args()

    repo_mapping = build_repo_mapping(args.repos)

    # ---------- Load the output CSV (create if missing) ----------
    if os.path.exists(args.out):
        df_saved = pd.read_csv(args.out)
        log.info("Loaded existing output with %d rows.", len(df_saved))
        if "diff_text" not in df_saved.columns:
            df_saved["diff_text"] = ""
    else:
        df_saved = pd.read_csv(args.csv)
        df_saved["diff_text"] = ""
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df_saved.to_csv(args.out, index=False)
        log.info("Created fresh output from %s (%d rows).",
                 args.csv, len(df_saved))

    if args.project:
        scope = df_saved.project == args.project
    else:
        scope = pd.Series(True, index=df_saved.index)

    # ---------- Identify missing diffs ----------
    missing_mask = scope & (
        df_saved["diff_text"].isna() | (df_saved["diff_text"] == "")
    )
    missing_indices = df_saved.index[missing_mask].tolist()
    log.info("Commits still missing diffs: %d", len(missing_indices))

    # ---------- Process them ----------
    repo_objects = {}
    save_counter = 0

    for idx in tqdm(missing_indices, desc="Extracting local diffs"):
        row = df_saved.loc[idx]
        project = row["project"]
        commit_id = row["commit_id"]

        repo_path = repo_mapping.get(project) or \
            repo_mapping.get(str(project).split("/")[-1])
        if not repo_path:
            df_saved.at[idx, "diff_text"] = ""
            continue

        if repo_path not in repo_objects:
            try:
                repo_objects[repo_path] = Repo(repo_path)
            except Exception as e:                 # noqa: BLE001
                log.warning("Error opening repo %s: %s", repo_path, e)
                df_saved.at[idx, "diff_text"] = ""
                continue

        try:
            diff = compute_diff(repo_objects[repo_path], commit_id)
        except Exception:                          # noqa: BLE001
            diff = ""
        df_saved.at[idx, "diff_text"] = diff

        save_counter += 1
        if save_counter >= args.save_every:
            df_saved.to_csv(args.out, index=False)
            log.info("Saved %d new diffs (total rows: %d)",
                     save_counter, len(df_saved))
            save_counter = 0

    df_saved.to_csv(args.out, index=False)
    log.info("Done. CSV saved with %d rows → %s", len(df_saved), args.out)


if __name__ == "__main__":
    main()
