"""
Driver: the FINAL RUN for one project (restore-only; nothing is rebuilt).
=========================================================================

Assumes this project's graph is ALREADY resident in Neo4j (restore its dump first
with `snapshot_neo4j.py restore`). Runs the full inference + scalability + reporting
chain under the frozen protocol in inference/protocol.py, then collects everything
into outputs/<project>/final_run/.

Difference from drivers/run_experiments.py, which this replaces for the final run:
  * adds the extra baselines, so BOTH JITLine variants are produced
    (incumbent = vocabulary frozen at warm-up; JITLine_fully_online = vocabulary
    and model refit every block, like CSTG)
  * adds the n=5 seed-robustness check (graph-bound: DW/KGE are re-derived per seed)
  * adds the per-metric RQ3 radars (axes = projects, the transpose of the
    per-project radars)
  * collects every artifact into final_run/ with the frozen config, so any table or
    figure can be redesigned later without touching Neo4j

Ordering rule: everything that needs the graph runs BEFORE anything that only needs
the caches, so the graph slot is occupied for as little time as possible.

Run:  KGC_PROJECT=groovy python drivers/run_final.py
      KGC_PROJECT=groovy python drivers/run_final.py --skip-seeds
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: F401
from config.project_config import PROJECT, summary

PKG = Path(__file__).resolve().parent.parent
INFER = PKG / "inference"


def _run(script, *args, label="", subdir=None, fatal=False):
    base = (PKG / subdir) if subdir else INFER
    cmd = [sys.executable, "-u", str(base / script), *[str(a) for a in args]]
    print(f"\n{'=' * 70}\n[{label}] {script}\n{'=' * 70}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        msg = f"[{label}] FAILED (exit {r.returncode})"
        if fatal:
            sys.exit(msg + " -- aborting.")
        print(msg + " -- continuing.")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seeds", action="store_true",
                    help="skip the n=5 robustness check (the slowest graph-bound step)")
    ap.add_argument("--no-reports", action="store_true")
    a = ap.parse_args()

    print(summary())
    sys.path.insert(0, str(INFER))
    import protocol as P
    print(f"\nFROZEN PROTOCOL: K={P.WARMUP_FRAC} M={P.BLOCK} G={P.GAP} "
          f"refit={P.REFIT_EVERY} roll={P.ROLL}")
    print(f"  overall F = {'+'.join(P.OVERALL_F)}   "
          f"init: adaptive [{P.INIT_FLOOR}..{P.INIT_CAP}], "
          f">={P.INIT_MIN_MINORITY} minority")
    print(f"  traj = stride {P.TRAJ_STRIDE} / window {P.TRAJ_WINDOW}")

    # ---------- graph-bound (Neo4j must hold THIS project) ----------
    _run("run_final_experiments.py", label="1 five methods x six graphs", fatal=True)
    print(f"\n{'=' * 70}\n[1b] precompute online feature streams\n{'=' * 70}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-u", "-c",
                    "import _kgc_paths, online_jit; online_jit.precompute_streams()"],
                   env=env, check=False)
    _run("run_final_fusion.py", label="2 fusion (31 combos, BOTH selection rules)",
         fatal=True)
    _run("run_subgraph_rq.py", label="3 subgraph-ablation RQ")
    _run("run_kg_methods_rq.py", label="4 five-method subgraph metrics")
    _run("collect_subgraph_stats.py", label="5 per-layer structural stats")
    if not a.skip_seeds:
        _run("run_seed_robustness.py", label="6 seed robustness (n=5)")

    # ---------- cache-only from here (no Neo4j needed) ----------
    _run("run_effort_eval.py", label="7 effort-aware eval (Popt / ACC@20)")
    _run("run_baselines.py", subdir="baselines", label="8 JIT baselines")
    _run("run_extra_baselines.py", subdir="baselines",
         label="9 extra baselines (BOTH JITLine variants)")
    _run("run_param_experiments.py", label="10 parameter sweeps")

    print(f"\n{'#' * 70}\nEXPERIMENTS COMPLETE for '{PROJECT}'.\n"
          f"Next: drivers/run_scalability.py, then drivers/make_reports.py\n"
          f"{'#' * 70}")


if __name__ == "__main__":
    main()
