"""
Suggest a BASE_COMMIT for the active project (KGC_PROJECT).

The base commit seeds the AST/structural base snapshot: it should be the first
commit at which the project has a substantial body of .java files (the "start
from zero" point). This helper picks the earliest labelled commit that both
checks out and yields > 0 .java files under the cloned repo, and prints a
paste-ready YAML value -- it does NOT edit projects.yaml (you paste the SHA in,
so the choice is explicit and auditable). This generalises the original
groovy-specific find_valid_commits.py to any project, with NO Neo4j dependency
(the base does not exist in the DB yet for a fresh project).

Run:
    (PowerShell)  $env:KGC_PROJECT = 'zookeeper'; python find_base_commit.py
    (bash)        KGC_PROJECT=zookeeper python find_base_commit.py

Options:
    --min-java N   require at least N .java files at the candidate (default 20)
    --scan N       scan the first N labelled commits by author_date (default 40)
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (PROJECT, REPO_PATH, CSV_PATH, summary)


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO_PATH), *args],
                          capture_output=True, text=True)


def _java_count_at(sha: str) -> int:
    """Number of .java files present in the tree at `sha` (0 if it won't check out)."""
    r = _git("ls-tree", "-r", "--name-only", sha)
    if r.returncode != 0:
        return -1
    return sum(1 for p in r.stdout.splitlines() if p.endswith(".java"))


def _labelled_commits_by_date():
    """(commit_id, author_date) for the project's labelled commits, earliest first."""
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r.get("commit_id") or r.get("commit") or ""
            ad = r.get("author_date") or r.get("year") or ""
            if cid:
                rows.append((cid.strip(), ad))
    rows.sort(key=lambda t: t[1])          # by author_date ascending
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-java", type=int, default=20)
    ap.add_argument("--scan", type=int, default=40)
    args = ap.parse_args()

    if not REPO_PATH.exists():
        sys.exit(f"Repo not cloned: {REPO_PATH}\n"
                 f"Clone it there first (read in place, never copied).")

    print(summary())
    rows = _labelled_commits_by_date()
    print(f"Scanning the first {min(args.scan, len(rows))} labelled commits of "
          f"{PROJECT} (need >= {args.min_java} .java files)...\n")

    chosen = None
    for cid, ad in rows[:args.scan]:
        n = _java_count_at(cid)
        flag = "" if n < 0 else f"{n} .java"
        print(f"  {cid[:12]}  {ad[:10]}  {flag if n >= 0 else '(no checkout)'}")
        if n >= args.min_java and chosen is None:
            chosen = (cid, ad, n)
            # keep printing a few more for context, then stop
    print()
    if chosen is None:
        print("No commit in the scan window has enough .java files. "
              "Increase --scan or lower --min-java, or verify the clone.")
        return

    cid, ad, n = chosen
    print("=" * 64)
    print(f"Suggested base_commit for '{PROJECT}': {cid}")
    print(f"  (author_date {ad[:10]}, {n} .java files at that commit)")
    print("=" * 64)
    print("\nPaste this into config/projects.yaml under the project's entry:\n")
    print(f'    base_commit: "{cid}"\n')


if __name__ == "__main__":
    main()
