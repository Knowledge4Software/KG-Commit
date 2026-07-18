"""
Live progress inspector for the KG-Commit pipeline (per project, KGC_PROJECT).

Shows, at a glance, where the active project's run currently is:

  * BUILD -- per online-growth layer (AST, CFG, DFG, PDG, SEQ): commits done /
    total, a progress bar, a live commit-rate and ETA (sampled from the per-commit
    timing CSV -- the most accurate signal, unaffected by buffered stdout), the
    running wall-time for that layer, and which layer is currently active.
  * GRAPH -- the Neo4j node/edge tallies landed so far (one read-only round-trip).
  * PIPELINE -- which result artifacts / scalability JSON / notebooks already exist
    under outputs/<project>/, so you can see how far past the build the run is.

It reads the per-commit timing logs the build driver writes to logs/<project>/,
the per-layer checkpoints under outputs/<project>/, and (optionally) Neo4j. It is
strictly read-only and safe to run at any time against a live build.

Run:
    KGC_PROJECT=kafka python check_progress.py            # one snapshot
    KGC_PROJECT=kafka python check_progress.py --watch    # refresh every 15s
    KGC_PROJECT=kafka python check_progress.py --watch --interval 30 --no-neo4j
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (PROJECT, OUT, SCAL_OUT, TIMING_DIR, CKPT_PATH,
                                    ckpt_path, CSV_PATH, NEO4J_URI, NEO4J_AUTH)

LAYERS = [("ast", "ASTNode"), ("cfg", "CFGNode"), ("dfg", "DFGNode"),
          ("pdg", "PDGNode"), ("seq", "SEQNode")]


def n_target():
    """Labelled-commit count = the index a completed online layer reaches."""
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def _timing_path(kind):
    return TIMING_DIR / f"{kind}_timing.csv"


def _ckpt_index(kind):
    p = CKPT_PATH if kind == "ast" else ckpt_path(kind)
    try:
        return json.loads(p.read_text()).get("next_index")
    except Exception:
        return None


def _timing_rows(kind):
    """(n_rows, last_idx, total_wall_s, recent_rate_cps, mtime) from the timing CSV."""
    p = _timing_path(kind)
    if not p.exists():
        return 0, None, 0.0, 0.0, None
    try:
        rows = p.read_text(errors="replace").splitlines()
        data = rows[1:] if rows and rows[0].startswith("idx") else rows
        n = len(data)
        if n == 0:
            return 0, None, 0.0, 0.0, p.stat().st_mtime
        def wall(line):
            try:
                return float(line.rsplit(",", 1)[1])
            except Exception:
                return 0.0
        def idx(line):
            try:
                return int(line.split(",", 1)[0])
            except Exception:
                return None
        total_ms = sum(wall(l) for l in data)
        recent = data[-100:]
        recent_ms = sum(wall(l) for l in recent) / max(len(recent), 1)
        rate = 1000.0 / recent_ms if recent_ms > 0 else 0.0     # commits/sec
        last = idx(data[-1])
        return n, last, total_ms / 1000.0, rate, p.stat().st_mtime
    except Exception:
        return 0, None, 0.0, 0.0, None


def _bar(done, total, width=32):
    if not total:
        return "[" + "?" * width + "]"
    f = min(max(done / total, 0.0), 1.0)
    k = int(f * width)
    return "[" + "#" * k + "-" * (width - k) + f"] {100*f:5.1f}%"


def _fmt_dur(s):
    s = int(s)
    h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return (f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s")


def build_status(target):
    print(f"  BUILD  (target = {target:,} labelled commits)")
    active = None
    for kind, _ in LAYERS:
        n, last, wall_s, rate, mtime = _timing_rows(kind)
        ck = _ckpt_index(kind)
        done = max([x for x in (n, last, ck) if x is not None], default=0)
        complete = target and done >= target
        fresh = mtime and (time.time() - mtime < 180)     # written recently
        state = "done " if complete else ("LIVE " if fresh else "     ")
        if fresh and not complete and active is None:
            active = kind
        eta = ""
        # show ETA for the in-progress (not yet complete) layer whenever we have a rate
        if not complete and rate > 0 and target and 0 < done < target:
            eta = f" ETA {_fmt_dur((target - done) / rate)} @ {rate:.1f} c/s"
        wall = f" [{_fmt_dur(wall_s)}]" if wall_s else ""
        print(f"    {kind.upper():4} {state}{_bar(done, target)}  {done:>5}/{target}{wall}{eta}")
    return active


def graph_status():
    print("  GRAPH  (Neo4j tallies)")
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with d.session() as s:
            parts = []
            for _, lbl in LAYERS:
                n = s.run(f"MATCH (a:{lbl}) RETURN count(a) AS n").single()["n"]
                parts.append(f"{lbl}={n:,}")
            for extra in ("Term", "Intent"):
                n = s.run(f"MATCH (a:{extra}) RETURN count(a) AS n").single()["n"]
                parts.append(f"{extra}={n:,}")
            cj = s.run("MATCH (c:Commit {in_jit:true}) RETURN count(c) AS n").single()["n"]
        d.close()
        print("    " + "  ".join(parts))
        print(f"    in_jit commits = {cj:,}")
    except Exception as e:
        print(f"    (Neo4j unreachable: {str(e)[:60]})")


def pipeline_status():
    print("  PIPELINE  (artifacts under outputs/%s/)" % PROJECT)
    checks = [
        ("final_experiments_results.pkl", OUT / "final_experiments_results.pkl"),
        ("final_fusion_results.pkl", OUT / "final_fusion_results.pkl"),
        ("subgraph_rq_results.pkl", OUT / "subgraph_rq_results.pkl"),
        ("cstg_bundle.pkl", OUT / "cstg_bundle.pkl"),
        ("scalability/*.json", SCAL_OUT),
        ("figures", OUT / "figures"),
        ("executed notebooks", OUT / "notebooks"),
        ("neo4j snapshot", OUT / "neo4j_dump"),
    ]
    for name, p in checks:
        if p.is_dir():
            cnt = len(list(p.rglob("*.json"))) if "scal" in name else len(list(p.rglob("*")))
            mark = "OK" if cnt else "--"
            print(f"    [{mark}] {name:<26} ({cnt} files)")
        else:
            print(f"    [{'OK' if p.exists() else '--'}] {name}")


def snapshot(no_neo4j):
    print("=" * 68)
    print(f"  KGC_PROJECT = {PROJECT}    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    target = n_target()
    active = build_status(target)
    if active:
        print(f"  -> currently building the {active.upper()} layer")
    print()
    if not no_neo4j:
        graph_status(); print()
    pipeline_status()
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh continuously")
    ap.add_argument("--interval", type=int, default=15, help="watch refresh seconds")
    ap.add_argument("--no-neo4j", action="store_true", help="skip the Neo4j round-trip")
    args = ap.parse_args()
    if not args.watch:
        snapshot(args.no_neo4j); return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            snapshot(args.no_neo4j)
            print(f"  (watching; Ctrl-C to stop; refresh {args.interval}s)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
