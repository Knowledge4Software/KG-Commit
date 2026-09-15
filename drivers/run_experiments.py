"""
Driver: run the FINAL inference experiments for the active project (KGC_PROJECT),
against the just-built live Neo4j graph. Produces the per-project result pickles
under outputs/<project>/ that the renderers and scalability suite consume.

Order:
  1  inference/run_final_experiments.py   5 methods (RN/PPR/LP/DW/KGE) x 6 graphs
                                          -> outputs/<p>/final_experiments_results.pkl
  2  inference/run_final_fusion.py        31-combo fusion + F / F+G / F+M
                                          -> outputs/<p>/final_fusion_results.pkl
  3  inference/run_subgraph_rq.py         subgraph-ablation RQ (M/R/T/P/E/Fusion)
                                          -> outputs/<p>/subgraph_rq_results.pkl
  4  inference/run_kg_methods_rq.py       5-method subgraph metrics
                                          -> outputs/<p>/subgraph_kg_methods_results.pkl
  5  inference/collect_subgraph_stats.py  per-layer structural stats (read-only)
                                          -> outputs/<p>/subgraph_layer_stats.json

Needs a live Neo4j with THIS project's graph fully built (run build_project.py
first). The heavy feature streams are cached to outputs/<p>/ on first use, so
later reruns / the renderers are DB-free.

Run:  KGC_PROJECT=zookeeper python drivers/run_experiments.py
"""
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


def _run(script, *args, label="", code=None, subdir=None):
    if code is not None:
        cmd = [sys.executable, "-c", code]
    else:
        base = (PKG / subdir) if subdir else INFER
        cmd = [sys.executable, str(base / script), *[str(a) for a in args]]
    print(f"\n{'='*70}\n[{label}] {' '.join(cmd[:3])}...\n{'='*70}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"[{label}] FAILED (exit {r.returncode}).")


def main():
    print(summary())
    _run("run_final_experiments.py", label="1 five methods x six graphs")
    # precompute the online feature streams (M + G channels) that the fusion needs;
    # produces outputs/<project>/online_jit_streams_v5.pkl (Neo4j once, then cached).
    _run(None, label="1b precompute online feature streams",
         code="import _kgc_paths, online_jit; online_jit.precompute_streams()")
    _run("run_final_fusion.py", label="2 fusion (31 combos + F/F+G/F+M)")
    _run("run_subgraph_rq.py", label="3 subgraph-ablation RQ")
    _run("run_kg_methods_rq.py", label="4 five-method subgraph metrics")
    _run("collect_subgraph_stats.py", label="5 per-layer structural stats")
    # effort-aware evaluation (Popt / ACC@20%LOC) of the KG channels, from the
    # cached streams (no extra Neo4j) -- the standard JIT effort metrics.
    _run("run_effort_eval.py", label="6 effort-aware eval (Popt / ACC@20)")
    # standard change-metric BASELINES (LR/RF/HGB + naive refs) under the same
    # online protocol, from the label CSV (no Neo4j) -- the comparison point.
    _run("run_baselines.py", subdir="baselines", label="7 within-project JIT baselines")
    # parameter-sensitivity sweeps (warm-up K / block M / rolling ROLL) of the
    # deployed fusion, from the cached streams (no Neo4j) -- justifies the tuned
    # protocol constants per project.
    _run("run_param_experiments.py", label="8 parameter-sensitivity sweeps")
    print(f"\n{'#'*70}\nEXPERIMENTS COMPLETE for '{PROJECT}'.\n"
          f"Next: python drivers/run_scalability.py\n{'#'*70}")


if __name__ == "__main__":
    main()
