"""
Preflight for the final run: assert every project is fully provisioned.
=======================================================================

Two failures during the first launch attempt were both "the project is not actually
provisioned here" problems that only surfaced after compute had been spent:

  * the Neo4j dump could not be opened (version gap), and
  * six projects had no label/diff CSV, which the CSTG stage needs.

A missing git repo is worse still: scalability/time_build_replay.py shells out to
`git`, so it degrades SILENTLY to empty build-cost timings instead of failing.

This script checks all four inputs for all eleven projects up front, plus the Neo4j
version actually serving, and exits non-zero if anything is missing. Run it before
starting the sweep.

Run:  python preflight_final_run.py
      python preflight_final_run.py --project zookeeper
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))
import _kgc_paths  # noqa: E402,F401
from protocol import PROJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data" / "apachejit" / "projects"
REPOS = ROOT / "repos" / "apache"


def check(project):
    """Return (ok, [problems]) for one project."""
    bad = []
    dump = OUTPUTS / project / "neo4j_dump" / f"{project}.dump"
    if not dump.exists():
        bad.append("dump missing")
    elif dump.stat().st_size < 1_000_000:
        bad.append(f"dump suspiciously small ({dump.stat().st_size} B)")

    csv = DATA / f"apache_{project}.csv"
    diff = DATA / f"apache_{project}_diff.csv"
    if not csv.exists():
        bad.append("label CSV missing")
    if not diff.exists():
        bad.append("diff CSV missing  <-- CSTG cannot run")

    repo = REPOS / project
    if not repo.is_dir():
        bad.append("repo missing  <-- build-cost timings would be SILENTLY empty")
    else:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True)
        if r.returncode != 0:
            bad.append("repo present but git HEAD does not resolve")
    return (not bad), bad


def neo4j_version():
    container = os.environ.get("KGC_NEO4J_CONTAINER", "")
    if not container:
        return None
    r = subprocess.run(["docker", "logs", "--tail", "400", container],
                       capture_output=True, text=True)
    for line in reversed((r.stdout + r.stderr).splitlines()):
        if "======== Neo4j" in line:
            return line.split("========")[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    a = ap.parse_args()

    projects = [a.project] if a.project else PROJECTS
    print(f"outputs : {OUTPUTS}")
    print(f"data    : {DATA}")
    print(f"repos   : {REPOS}")
    v = neo4j_version()
    if v:
        print(f"neo4j   : {v}")
    print("=" * 68)

    n_bad = 0
    for p in projects:
        ok, bad = check(p)
        if ok:
            print(f"  [ok  ] {p}")
        else:
            n_bad += 1
            print(f"  [FAIL] {p}")
            for b in bad:
                print(f"           - {b}")

    print("=" * 68)
    if n_bad:
        print(f"{n_bad}/{len(projects)} project(s) NOT ready. Fix before running.")
        sys.exit(1)
    print(f"all {len(projects)} project(s) fully provisioned.")


if __name__ == "__main__":
    main()
