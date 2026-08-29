"""
Per-commit featurisation cost, split by WHICH features a baseline actually needs.
=================================================================================

Charging every change-metric baseline the same featurisation cost is wrong:
they do not all consume the same features.

  LApredict   uses ONE feature, `la` (added lines).           run_extra_baselines.lapredict_scores
  LR/HGB/RF   use all 12 Kamei metrics.                       run_baselines.JIT_COLS
  Deeper      uses the same 12.
  JITLine     additionally tokenises the diff.
  DeepJIT     tokenises message + code.

The 12 metrics split into two groups with very different costs:

  DIFF group   la, ld, nf, nd, ns, ent
               one `git diff --numstat` against the parent, then arithmetic.
               `la` alone needs the SAME diff, so LApredict pays the diff but
               not the history work.

  HISTORY group  ndev, age, nuc, exp, rexp, sexp
               lookups over the touched files' prior history (who touched them,
               when, how often, and the author's prior experience).

So the honest accounting is:

  LApredict       = diff only            (it needs `la`, nothing else)
  LR / HGB / RF   = diff + history       (all 12)

This script measures the two groups separately, per project. Process-spawn
overhead is measured and excluded: invoking `git` once per commit costs ~20 ms
of process creation on this platform, which is a property of the harness, not
of computing a diff. The batched figure is what a deployment would pay and is
what we report.

The history-group cost is an in-memory index lookup, which is the cheapest
plausible implementation and therefore a LOWER BOUND on a real deployment that
would query a store. We say so rather than presenting it as the true cost.

Cache-only apart from `git` against the local clone. No Neo4j.

Run:  python baselines/measure_featurisation_bygroup.py --project activemq
Out:  outputs/<project>/final_final_run/complexity/featurisation_bygroup.json
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

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]


def repo_path(project):
    for c in (ROOT / "repos" / "apache" / project, ROOT / "repos" / project):
        if (c / ".git").exists():
            return c
    return None


def parse_numstat(lines):
    """la, ld and the per-file churn needed for nf/nd/ns/ent."""
    la = ld = 0
    files, per_file = [], []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        a = 0 if parts[0] == "-" else int(parts[0])
        d = 0 if parts[1] == "-" else int(parts[1])
        la += a
        ld += d
        files.append(parts[2])
        per_file.append(a + d)
    return la, ld, files, per_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import numpy as np

    repo = repo_path(a.project)
    if repo is None:
        print(f"  {a.project}: no clone -- skipped")
        return

    shas = subprocess.run(["git", "-C", str(repo), "log", "--format=%H"],
                          capture_output=True, text=True).stdout.split()
    if not shas:
        print(f"  {a.project}: no commits -- skipped")
        return
    rng = np.random.default_rng(a.seed)
    k = min(a.sample, len(shas))
    pick = [shas[i] for i in rng.choice(len(shas), k, replace=False)]

    # --- process-spawn overhead (to be excluded) -------------------------
    spawn = []
    for _ in range(20):
        t0 = time.perf_counter()
        subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True)
        spawn.append((time.perf_counter() - t0) * 1000.0)
    spawn_ms = float(np.median(spawn))

    # --- DIFF group, batched (one git process, k commits) ----------------
    t0 = time.perf_counter()
    r = subprocess.run(["git", "-C", str(repo), "log", "-n", str(k),
                        "--numstat", "--format=%H"],
                       capture_output=True, text=True)
    git_ms = (time.perf_counter() - t0) * 1000.0 / k

    # arithmetic on top of the numstat: la/ld are free once parsed; the
    # nf/nd/ns/ent terms need the per-file breakdown, timed separately.
    blocks, cur = [], []
    for ln in r.stdout.splitlines():
        if len(ln) == 40 and " " not in ln:
            if cur:
                blocks.append(cur)
            cur = []
        elif ln.strip():
            cur.append(ln)
    if cur:
        blocks.append(cur)

    t0 = time.perf_counter()
    touched = []
    for b in blocks:
        la, ld, files, per_file = parse_numstat(b)
        touched.append(files)
    la_ms = (time.perf_counter() - t0) * 1000.0 / max(len(blocks), 1)

    t0 = time.perf_counter()
    for b in blocks:
        la, ld, files, per_file = parse_numstat(b)
        dirs = {"/".join(f.split("/")[:-1]) for f in files}
        subs = {f.split("/")[0] for f in files}
        tot = sum(per_file)
        ent = 0.0
        if tot:
            for c in per_file:
                if c:
                    p = c / tot
                    ent -= p * math.log(p, 2)
        _ = (len(files), len(dirs), len(subs), ent)
    scope_ms = (time.perf_counter() - t0) * 1000.0 / max(len(blocks), 1)
    scope_ms = max(scope_ms - la_ms, 0.0)     # marginal over parsing alone

    # --- HISTORY group ---------------------------------------------------
    devs_of = defaultdict(set)
    last_of, nuc_of = {}, defaultdict(int)
    hist = []
    for i, files in enumerate(touched):
        t0 = time.perf_counter()
        devs, ages, nuc = set(), [], 0
        for f in files:
            devs |= devs_of[f]
            if f in last_of:
                ages.append(i - last_of[f])
            nuc += nuc_of[f]
        _ = (len(devs), (sum(ages) / len(ages)) if ages else 0.0, nuc)
        hist.append((time.perf_counter() - t0) * 1000.0)
        for f in files:
            devs_of[f].add(str(i))
            last_of[f] = i
            nuc_of[f] += 1
    hist_ms = float(np.median(hist)) if hist else 0.0

    diff_group = git_ms + la_ms + scope_ms          # la, ld, nf, nd, ns, ent
    la_only = git_ms + la_ms                        # LApredict needs only `la`

    res = {
        "project": a.project,
        "n_sampled": len(blocks),
        "spawn_overhead_ms_excluded": spawn_ms,
        "git_numstat_ms": git_ms,
        "parse_ms": la_ms,
        "scope_entropy_ms": scope_ms,
        "history_ms": hist_ms,
        "la_only_ms": la_only,                      # LApredict
        "twelve_metrics_ms": diff_group + hist_ms,  # LR / HGB / RF / Deeper
        "_note": ("Featurisation split by the features each baseline consumes. "
                  "LApredict uses only `la` and therefore pays the diff but not "
                  "the history work; LR/HGB/RF use all 12 Kamei metrics. "
                  "Per-commit `git` process-spawn overhead (%.1f ms here) is "
                  "measured and excluded as a harness artefact. The history "
                  "term is an in-memory index lookup and is a LOWER BOUND."
                  % spawn_ms),
    }
    dst = OUTP / a.project / "final_final_run" / "complexity"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "featurisation_bygroup.json").write_text(json.dumps(res, indent=1),
                                                    encoding="utf-8")
    print(f"  {a.project}: git={git_ms:.3f} parse={la_ms:.3f} "
          f"scope={scope_ms:.3f} hist={hist_ms:.4f} | "
          f"LA-only={la_only:.3f}  12-metrics={res['twelve_metrics_ms']:.3f} ms")


if __name__ == "__main__":
    main()
