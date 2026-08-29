"""
Per-layer values for the RQ3 matrix (T6), on the final_final_run protocol.
=========================================================================

The RQ3 matrix reports, per project and per graph column (core/ast/cfg/dfg/
pdg/final), five single-method rows and two fusion rows -- the per-project
chosen $F_{pp}$ and the fixed overall $F_{ov}$ = RN+PPR.

Why this script exists
----------------------
The stored pickles do not give a self-consistent table:

* `final_experiments_results.pkl` (method rows) was scored over `[W, N)` with
  **gap=0** -- `run_final_experiments` imports `GAP` but never passes it to
  `final_metrics`, so the verification latency G=50 was silently dropped.
* `final_fusion_results.pkl` (fusion rows) was scored over `[W+init, N)` with
  **gap=GAP**, and only on the deployed CSTG graph.

Two different protocols in one table, which is also why a rule that reduces to
a single method (e.g. activemq chosen = PPR) did not reproduce its own method
row.

What this script does
---------------------
Re-derives EVERY cell of the matrix under the one final_final_run protocol:

    K = 0.05 warm-up (W from the cache), M = BLOCK = 200, G = GAP = 50,
    REFIT_EVERY = 1, adaptive stacking init = protocol.init_window(y, W)

evaluated over the common span `[W+init, N)` so method and fusion rows are
directly comparable. A single-method rule then reproduces its method row
exactly, by construction.

Everything is replayed from `raw_method_scores.pkl`, which persists the
full-length per-commit score array for every method on every graph -- the cache
that exists precisely so this needs no Neo4j. Scoring uses the project's own
`online_jit.final_metrics` and `run_final_fusion.eval_subset`.

Cache-only: reads outputs/<project>/final_final_run (plus `init` from
final_run only as a cross-check). No Neo4j, no graph rebuild.

Run:  python inference/rq3_perlayer_fusion.py
Out:  outputs/tables/rq3_perlayer_fusion.json
"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
DEST = OUTP / "tables" / "rq3_perlayer_fusion.json"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
GRAPHS = ["core", "ast", "cfg", "dfg", "pdg", "final"]
M3 = ("Macro_F1", "G_Mean", "AUC")


def one_project(p):
    """Every matrix cell for one project, on the final_final_run protocol."""
    import numpy as np
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent))
    import _kgc_paths  # noqa: F401
    import protocol as P
    import run_final_fusion as rff
    from online_jit import final_metrics

    ffr = OUTP / p / "final_final_run"
    R = pickle.load(open(ffr / "experiments" / "raw_method_scores.pkl", "rb"))
    F = pickle.load(open(ffr / "fusion" / "final_fusion_results.pkl", "rb"))

    y = np.asarray(R["y"], int)
    N, W = len(y), int(R["warmup"])
    init = P.init_window(y, W)          # the adaptive rule, recomputed
    ev = np.arange(W + init, N)         # one common span for every row

    out = {"meta": {"N": N, "W": W, "init": init, "ev0": int(W + init),
                    "gap": int(P.GAP), "block": int(P.BLOCK),
                    "warmup_frac": float(P.WARMUP_FRAC)},
           "methods": {}, "pp": {}, "ov": {}}

    # single-method rows: the raw per-commit score, scored on the common span
    # with the verification gap -- the same treatment eval_subset gives a
    # size-1 subset.
    for g in GRAPHS:
        out["methods"][g] = {}
        for m in METHODS:
            s = np.asarray(R["scores"][g][m], float)
            s = np.clip(np.nan_to_num(s[ev], nan=float(y[ev].mean())), 0, 1)
            mm = final_metrics(y[ev], s, gap=P.GAP)
            out["methods"][g][m] = {k: float(mm[k]) for k in M3}

    # fusion rows: prequential stacking over the same span, same gap
    for rule, key in (("pp", "chosen"), ("ov", "chosen_overall")):
        subset = [x for x in F[key].split("+") if x]
        cols = {}
        for g in GRAPHS:
            sc = {m: np.asarray(R["scores"][g][m], float) for m in METHODS}
            mm, _, _, _ = rff.eval_subset(sc, subset, y, W, N, init=init,
                                          refit=P.REFIT_EVERY, gap=P.GAP)
            cols[g] = {k: float(mm[k]) for k in M3}
        out[rule] = {"name": F[key], "cols": cols}

    # invariant: a one-method rule must equal that method's row exactly
    for rule in ("pp", "ov"):
        mem = [x for x in out[rule]["name"].split("+") if x]
        if len(mem) == 1:
            for g in GRAPHS:
                a = out[rule]["cols"][g]["Macro_F1"]
                b = out["methods"][g][mem[0]]["Macro_F1"]
                assert abs(a - b) < 1e-6, f"{p}/{rule}/{g}: {a} != {b}"
    return out


def main():
    if len(sys.argv) > 1:
        p = sys.argv[1]
        print(json.dumps({p: one_project(p)}))
        return

    all_out = {}
    for p in PROJECTS:
        env = dict(os.environ, KGC_PROJECT=p)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), p],
                           capture_output=True, text=True, env=env, cwd=str(HERE))
        line = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
        if not line.startswith("{"):
            print(f"  {p}: FAILED\n{r.stderr[-500:]}")
            continue
        all_out.update(json.loads(line))
        m = all_out[p]["meta"]
        print(f"  {p:10} ok  W={m['W']:<5} init={m['init']:<4} "
              f"ev0={m['ev0']:<5} gap={m['gap']}")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(all_out, indent=1), encoding="utf-8")
    print(f"\nwrote {DEST.relative_to(ROOT)}  ({len(all_out)} projects)")


if __name__ == "__main__":
    main()
