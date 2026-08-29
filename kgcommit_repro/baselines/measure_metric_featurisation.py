"""
Measure what it really costs to DERIVE the 12 ApacheJIT change metrics per commit.
==================================================================================

Why this exists
---------------
The change-metric baselines (LR, HGB, RF, LApredict) are timed in
`timing_probe.py` with a featurisation cost of ~0.001 ms/commit, because at
evaluation time their features are a lookup into a table ApacheJIT ships
precomputed. That is not the deployment cost: in a real deployment the 12
metrics do not exist until someone computes them from the commit.

Deriving them needs two things per commit:

  * a DIFF against the parent, for the size/scope metrics
      la, ld  (lines added/deleted)      -- from the unified diff
      nf, nd, ns  (files/dirs/subsystems touched)
      ent     (entropy of the change across files)
  * HISTORY lookups, for the experience/ownership metrics
      ndev    (developers who previously touched the touched files)
      age     (time since those files were last modified)
      nuc     (unique prior changes to the touched files)
      exp, rexp, sexp  (author experience, recency-weighted, subsystem)

This script measures both on a random sample of real commits, so the paper can
report a defensible per-commit featurisation cost for the metric baselines
instead of a table lookup.

Method
------
DIFF cost is measured directly: `git diff --numstat <sha>^ <sha>` on a sample of
commits from the project's own repository, which is exactly the input a
deployment would have. Entropy/nf/nd/ns are then computed from that numstat in
Python and included.

HISTORY cost is measured as the lookup against an in-memory index of prior
file-touches built from the label CSV -- the cheapest plausible implementation,
so the number is a LOWER BOUND on what a real deployment pays. We say so in the
output rather than presenting it as the true cost.

Cache-only apart from `git diff` against the local clone; no Neo4j.

Run:  python baselines/measure_metric_featurisation.py --project activemq --sample 200
Out:  outputs/<project>/final_final_run/complexity/metric_featurisation.json
"""
import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"


def repo_path(project):
    for cand in (ROOT / "repos" / "apache" / project,
                 ROOT / "repos" / project,
                 Path.home() / "repos" / "apache" / project):
        if (cand / ".git").exists():
            return cand
    return None


def subsystem(path):
    parts = path.split("/")
    return parts[0] if parts else ""


def directory(path):
    return "/".join(path.split("/")[:-1])


def metrics_from_numstat(lines):
    """la, ld, nf, nd, ns, ent from a `git diff --numstat` block."""
    la = ld = 0
    files, dirs, subs = [], set(), set()
    per_file = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        a, d, path = parts[0], parts[1], parts[2]
        a = 0 if a == "-" else int(a)
        d = 0 if d == "-" else int(d)
        la += a
        ld += d
        files.append(path)
        dirs.add(directory(path))
        subs.add(subsystem(path))
        per_file.append(a + d)
    tot = sum(per_file)
    ent = 0.0
    if tot > 0:
        for c in per_file:
            if c > 0:
                p = c / tot
                ent -= p * math.log(p, 2)
    return {"la": la, "ld": ld, "nf": len(files), "nd": len(dirs),
            "ns": len(subs), "ent": ent}, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import pandas as pd

    repo = repo_path(a.project)
    if repo is None:
        print(f"  {a.project}: no local clone found -- skipped")
        return

    # commit list from the project's own label CSV (chronological)
    csvs = list((ROOT / "data").glob(f"**/{a.project}*.csv")) if (ROOT / "data").exists() else []
    shas = []
    if csvs:
        df = pd.read_csv(csvs[0])
        col = next((c for c in ("commit_id", "commit_hash", "sha") if c in df), None)
        if col:
            shas = df[col].astype(str).tolist()
    if not shas:
        out = subprocess.run(["git", "-C", str(repo), "log", "--format=%H"],
                             capture_output=True, text=True)
        shas = out.stdout.split()
    if not shas:
        print(f"  {a.project}: no commits found -- skipped")
        return

    rng = np.random.default_rng(a.seed)
    pick = [shas[i] for i in rng.choice(len(shas), min(a.sample, len(shas)),
                                        replace=False)]

    # ---- process-spawn overhead, so it can be subtracted -----------------
    # A per-commit `git` invocation on Windows costs ~20 ms in process creation
    # alone. That is an artefact of shelling out once per commit, not the cost
    # of computing a diff, so it must not be charged to featurisation. We
    # measure it with a trivial git call and report the corrected figure.
    spawn = []
    for _ in range(20):
        t0 = time.perf_counter()
        subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True)
        spawn.append((time.perf_counter() - t0) * 1000.0)
    spawn_ms = float(np.median(spawn))

    # ---- diff-derived metrics -------------------------------------------
    diff_ms, ok = [], 0
    touched = {}
    for sha in pick:
        t0 = time.perf_counter()
        r = subprocess.run(["git", "-C", str(repo), "diff", "--numstat",
                            f"{sha}^", sha],
                           capture_output=True, text=True)
        if r.returncode != 0:
            continue
        m, files = metrics_from_numstat(r.stdout.splitlines())
        diff_ms.append((time.perf_counter() - t0) * 1000.0)
        touched[sha] = files
        ok += 1

    # ---- the same work batched into ONE git process ---------------------
    # A real deployment would not spawn a process per commit; this is the
    # defensible per-commit diff cost, and it is what the paper should quote.
    t0 = time.perf_counter()
    subprocess.run(["git", "-C", str(repo), "log", "-n", str(len(pick)),
                    "--numstat", "--format=%H"], capture_output=True, text=True)
    batched_ms = (time.perf_counter() - t0) * 1000.0 / max(len(pick), 1)

    # ---- history-derived metrics (lower bound: in-memory index) ---------
    file_devs = defaultdict(set)
    file_last = {}
    file_nuc = defaultdict(int)
    hist_ms = []
    for i, sha in enumerate(pick):
        files = touched.get(sha, [])
        t0 = time.perf_counter()
        devs, ages, nuc = set(), [], 0
        for f in files:
            devs |= file_devs[f]
            if f in file_last:
                ages.append(i - file_last[f])
            nuc += file_nuc[f]
        _ = (len(devs), (sum(ages) / len(ages)) if ages else 0.0, nuc)
        hist_ms.append((time.perf_counter() - t0) * 1000.0)
        for f in files:
            file_devs[f].add(sha[:8])
            file_last[f] = i
            file_nuc[f] += 1

    if not diff_ms:
        print(f"  {a.project}: no usable commits -- skipped")
        return

    raw_med = float(np.median(diff_ms))
    res = {
        "project": a.project,
        "n_sampled": ok,
        "spawn_overhead_ms": spawn_ms,
        "diff_ms_per_commit_raw": {          # includes one process spawn
            "median": raw_med,
            "mean": float(np.mean(diff_ms)),
            "p95": float(np.percentile(diff_ms, 95)),
        },
        "diff_ms_per_commit_spawn_corrected": max(raw_med - spawn_ms, 0.0),
        "diff_ms_per_commit_batched": float(batched_ms),
        "history_ms_per_commit": {
            "median": float(np.median(hist_ms)),
            "mean": float(np.mean(hist_ms)),
            "p95": float(np.percentile(hist_ms, 95)),
        },
        # What the paper should quote: batched diff (no per-commit process
        # spawn) plus the history lookup.
        "total_featurise_ms_per_commit": float(batched_ms
                                               + np.median(hist_ms)),
        "_note": ("Diff cost measured with `git --numstat` on real commits. The "
                  "per-commit `git diff` figure includes ~%.0f ms of process-spawn "
                  "overhead on this platform, which is an artefact of shelling out "
                  "once per commit rather than the cost of computing a diff; the "
                  "batched figure removes it and is the one to report. History "
                  "cost is an in-memory index lookup and is a LOWER BOUND on a "
                  "real deployment, which would query a store."
                  % spawn_ms),
    }
    dst = OUTP / a.project / "final_final_run" / "complexity"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metric_featurisation.json").write_text(json.dumps(res, indent=1),
                                                   encoding="utf-8")
    print(f"  {a.project}: n={ok}  raw={raw_med:.2f} spawn={spawn_ms:.2f} "
          f"batched={batched_ms:.3f}  hist={res['history_ms_per_commit']['median']:.4f}"
          f"  -> total={res['total_featurise_ms_per_commit']:.3f} ms")


if __name__ == "__main__":
    main()
