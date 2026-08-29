"""
De-duplicate per-commit timing logs in kgcommit_repro/logs/<project>/.

WHY THEY HAVE DUPLICATES
------------------------
The online build engines open their --timing-log in APPEND mode. When a build is
resumed (or a range is re-run), the engine re-processes commits it has already
logged and appends a second set of rows for the same `idx`. Nothing ever truncates
or de-duplicates the file, so a project resumed N times can carry up to N rows per
commit.

The duplicate rows are NOT identical:
  * `wall_ms` differs (a genuinely re-measured wall-clock time), and
  * GumTree's matching fields (`moves`, `updates`, `matched`, and +/-1 on
    `adds`/`removes`) can differ slightly, because the tree-matching heuristic is
    not bit-deterministic across runs.
File-count columns (`A`/`D`) never differ, confirming the passes saw the same
commits and the same diffs -- only the edit script's tie-breaking varied.

POLICY
------
Keep the LAST occurrence of each `idx`. The final pass is the one whose graph state
corresponds to the finished build, so its deltas are the ones consistent with the
Neo4j graph the results were computed from.

The original file is preserved alongside as <name>.orig-<stamp>.csv unless
--no-backup is passed.

Usage:
    python dedup_timing_logs.py --dry-run          # report only
    python dedup_timing_logs.py                    # de-dup every project
    python dedup_timing_logs.py --project hive     # one project
"""
import argparse
import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parent / "logs"
DELTA = ["A", "M", "D", "adds", "removes", "updates", "moves", "matched", "boot"]


def analyse(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return None
    by = defaultdict(list)
    for r in rows:
        by[int(r["idx"])].append(r)
    dups = {k: v for k, v in by.items() if len(v) > 1}
    delta_diff = 0
    for v in dups.values():
        if any(c in v[0] and len({r[c] for r in v}) > 1 for c in DELTA):
            delta_diff += 1
    return {"rows": len(rows), "unique": len(by), "dups": len(dups),
            "delta_diff": delta_diff, "by": by, "fields": list(rows[0].keys())}


def dedup(path, backup=True):
    a = analyse(path)
    if a is None or a["dups"] == 0:
        return a, False
    if backup:
        stamp = time.strftime("%Y%m%d")
        shutil.copy2(path, path.with_name(f"{path.stem}.orig-{stamp}.csv"))
    out = [a["by"][k][-1] for k in sorted(a["by"])]      # LAST wins
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=a["fields"])
        w.writeheader()
        w.writerows(out)
    return a, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    projects = [args.project] if args.project else \
        sorted(p.name for p in LOGS.iterdir() if p.is_dir())

    total_removed = 0
    for proj in projects:
        d = LOGS / proj
        files = sorted(f for f in d.glob("*_timing.csv") if ".orig-" not in f.name)
        if not files:
            continue
        print(f"\n### {proj}")
        for f in files:
            a = analyse(f)
            if a is None:
                print(f"   {f.name:<22} empty")
                continue
            if a["dups"] == 0:
                print(f"   {f.name:<22} clean ({a['rows']} rows)")
                continue
            removed = a["rows"] - a["unique"]
            note = f"  [{a['delta_diff']} groups differ in deltas]" if a["delta_diff"] else ""
            if args.dry_run:
                print(f"   {f.name:<22} {a['rows']} -> {a['unique']}  (-{removed}){note}")
            else:
                _, done = dedup(f, backup=not args.no_backup)
                print(f"   {f.name:<22} {a['rows']} -> {a['unique']}  (-{removed}){note}"
                      f"{'  DEDUPED' if done else ''}")
                total_removed += removed

    if not args.dry_run and total_removed:
        print(f"\nremoved {total_removed} duplicate rows in total")
    elif args.dry_run:
        print("\n(dry run -- nothing written)")


if __name__ == "__main__":
    main()
