"""
Clone the ApacheJIT project repositories the KG-Commit build reads in place.
==========================================================================

The build replays each labelled commit by SHA (git show/diff), so it needs the
FULL commit history -- shallow clones would leave labelled SHAs unresolvable.
This helper clones one or more apache/<project> repos to repos/apache/<project>/,
skipping any that are already cloned, and reports disk headroom before each clone
so a large clone cannot silently fill the disk.

Usage:
    python clone_repos.py spark hive camel        # clone these
    python clone_repos.py --all-missing           # every ApacheJIT project not yet cloned
    python clone_repos.py spark --min-free-gb 5    # abort if < 5 GB would remain

Notes:
  * Reads nothing from Neo4j; pure network + disk. Safe to run alongside a build,
    but watch disk space (each of camel/hive is multi-GB).
  * project_key format is apache/<name>; the clone target is repos/apache/<name>.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent
REPOS = PROJECT_ROOT / "repos" / "apache"
LABELS = PROJECT_ROOT / "data" / "apachejit" / "projects"


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _is_cloned(dest: Path) -> bool:
    """A repo is 'cloned' only if HEAD actually resolves -- git creates .git/
    early during a clone, so `.git` existence alone gives a false positive while a
    clone is still in flight."""
    if not (dest / ".git").exists():
        return False
    r = subprocess.run(["git", "-C", str(dest), "rev-parse", "--verify", "HEAD"],
                       capture_output=True)
    return r.returncode == 0


def all_apachejit_projects():
    """Project names that have an ApacheJIT label CSV (apache_<name>.csv)."""
    names = []
    for p in sorted(LABELS.glob("apache_*.csv")):
        n = p.stem[len("apache_"):]
        if not n.endswith("_diff"):
            names.append(n)
    return names


def clone_one(name: str, min_free_gb: float) -> str:
    dest = REPOS / name
    if _is_cloned(dest):
        return f"{name}: already cloned -> {dest}"
    free = _free_gb(PROJECT_ROOT)
    print(f"  disk free before {name}: {free:.1f} GB")
    if free < min_free_gb:
        return (f"{name}: SKIPPED -- only {free:.1f} GB free (< --min-free-gb "
                f"{min_free_gb}). Free space or lower the threshold.")
    REPOS.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/apache/{name}.git"
    print(f"  cloning {url} -> {dest} (full history) ...", flush=True)
    r = subprocess.run(["git", "clone", url, str(dest)])
    if r.returncode != 0:
        return f"{name}: CLONE FAILED (git exit {r.returncode})."
    return f"{name}: cloned -> {dest}  (free now {_free_gb(PROJECT_ROOT):.1f} GB)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projects", nargs="*", help="project names, e.g. spark hive camel")
    ap.add_argument("--all-missing", action="store_true",
                    help="clone every ApacheJIT project not yet under repos/apache/")
    ap.add_argument("--min-free-gb", type=float, default=3.0,
                    help="abort a clone if fewer than this many GB would remain")
    args = ap.parse_args()

    if args.all_missing:
        targets = [n for n in all_apachejit_projects()
                   if not _is_cloned(REPOS / n)]
    else:
        targets = args.projects
    if not targets:
        sys.exit("Nothing to clone. Pass project names or --all-missing.")

    print(f"Targets: {', '.join(targets)}")
    print(f"Disk free: {_free_gb(PROJECT_ROOT):.1f} GB  |  min-free guard: {args.min_free_gb} GB\n")
    for name in targets:
        # verify it is a known ApacheJIT project (has a label CSV)
        if not (LABELS / f"apache_{name}.csv").exists():
            print(f"{name}: no ApacheJIT label CSV ({LABELS / f'apache_{name}.csv'}); skipping.")
            continue
        print(clone_one(name, args.min_free_gb))
        print()


if __name__ == "__main__":
    main()
