#!/usr/bin/env python3
"""
Progress dashboard for the sequential alternative-subgraph builds
(build_subgraph_online_kg.py, driven by run_subgraphs_sequential.sh).

Shows, per kind (cfg/dfg/pdg/seq): commits done / 8059, a progress bar, live
commit rate + ETA (sampled over a short window), the Neo4j node/delta-edge
tallies landed so far, whether the build process is alive, and which kind the
sequential runner is currently on.

Run:  python check_sg_progress.py            # one snapshot
      python check_sg_progress.py --watch    # refresh every 15s
"""
import argparse, json, time, subprocess, sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
KINDS  = ["cfg", "dfg", "pdg", "seq"]
TOTAL  = 8059
LABEL  = {"cfg": "CFGNode", "dfg": "DFGNode", "pdg": "PDGNode", "seq": "SEQNode"}
RUNLOG = ROOT / "outputs" / "sg_runs" / "seq_runner.log"


def ckpt_index(kind):
    p = ROOT / "outputs" / f"online_kg_checkpoint_{kind}.json"
    try:
        return json.loads(p.read_text())["next_index"]
    except Exception:
        return None


def log_index(kind):
    """Latest [idx/8059] from the (unbuffered) build log -- finer than the
    checkpoint, which only saves every 20 commits. Falls back to None."""
    p = ROOT / "outputs" / "sg_runs" / f"build_{kind}.log"
    try:
        import re
        # scan the last few lines for the LAST complete [idx/8059] match
        # (the final line may be a partial mid-write line that won't match)
        for ln in reversed(p.read_text(errors="replace").splitlines()[-6:]):
            m = re.search(r"\[(\d+)/\d+\]", ln)
            if m:
                return int(m.group(1))
        return None
    except Exception:
        return None


def cur_index(kind):
    """Best available commit index: prefer the live log, else the checkpoint."""
    li, ci = log_index(kind), ckpt_index(kind)
    return max([x for x in (li, ci) if x is not None], default=None)


def neo4j_counts():
    """One round-trip: node + delta-edge counts per label. {} if DB unreachable."""
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password1234"))
        out = {}
        with d.session() as s:
            for k, L in LABEL.items():
                n = s.run(f"MATCH (a:{L}) RETURN count(a) AS n").single()["n"]
                e = s.run(f"MATCH (:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(:{L}) "
                          f"RETURN count(r) AS n").single()["n"]
                out[k] = (n, e)
        d.close()
        return out
    except Exception as e:
        return {"_err": str(e)[:50]}


def alive_kinds():
    """Set of kinds whose build process is currently running (via tasklist/ps)."""
    running = set()
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'build_subgraph_online_kg' } | "
             "ForEach-Object { if ($_.CommandLine -match '--kind (\\w+)') { $Matches[1] } }"],
            capture_output=True, text=True, timeout=15).stdout
        running = {ln.strip() for ln in out.splitlines() if ln.strip() in KINDS}
    except Exception:
        pass
    return running


def runner_status():
    if not RUNLOG.exists():
        return "runner log not found"
    lines = [l for l in RUNLOG.read_text().splitlines() if l.strip()]
    return lines[-1] if lines else "runner log empty"


def bar(frac, width=28):
    f = max(0.0, min(1.0, frac)); k = int(f * width)
    return "#" * k + "." * (width - k)


def snapshot(sample_secs):
    # sample rate over a short window using the live log index (finer than ckpt)
    a = {k: cur_index(k) for k in KINDS}
    t0 = time.time()
    time.sleep(sample_secs)
    b = {k: cur_index(k) for k in KINDS}
    dt = time.time() - t0

    counts = neo4j_counts()
    running = alive_kinds()

    print("=" * 78)
    print(f"  Subgraph build progress   {time.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"(rate over {dt:.0f}s)")
    print("=" * 78)
    hdr = f"  {'kind':<5}{'commits':>13}  {'progress':<30}{'rate':>9}{'ETA':>8}  {'run':>4}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in KINDS:
        idx = b[k]
        if idx is None:
            print(f"  {k:<5}{'no checkpoint':>13}"); continue
        rate = ((b[k] - a[k]) / dt) if (a[k] is not None and dt > 0) else 0.0
        done = idx >= TOTAL
        eta = "done" if done else (f"{(TOTAL-idx)/rate/3600:4.1f}h" if rate > 0 else "  --")
        rmark = "RUN" if k in running else ("OK" if done else "idle")
        rt = f"{rate*60:5.0f}/m" if rate > 0 else "   0/m"
        nc = ""
        if k in counts and isinstance(counts[k], tuple):
            nc = f"   [{counts[k][0]:,}n {counts[k][1]:,}e]"
        print(f"  {k:<5}{idx:>7}/{TOTAL:<5} {bar(idx/TOTAL)} {rt:>8}{eta:>8}  {rmark:>4}{nc}")
    if "_err" in counts:
        print(f"  (Neo4j unreachable: {counts['_err']})")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  runner: {runner_status()}")
    total_done = sum((b[k] or 0) for k in KINDS)
    print(f"  overall: {total_done:,}/{TOTAL*4:,} commit-steps "
          f"({100*total_done/(TOTAL*4):.1f}%) across all 4 layers")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh continuously")
    ap.add_argument("--interval", type=int, default=15, help="watch refresh seconds")
    ap.add_argument("--sample", type=int, default=5, help="rate-sampling window seconds")
    args = ap.parse_args()
    try:
        while True:
            snapshot(args.sample)
            if not args.watch:
                break
            time.sleep(max(0, args.interval - args.sample))
    except KeyboardInterrupt:
        print("\n(stopped)")


if __name__ == "__main__":
    main()
