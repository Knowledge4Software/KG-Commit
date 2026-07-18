"""
Driver: run the scalability / complexity / statistical-analysis suite (the paper's
second research question) for the active project (KGC_PROJECT).

All outputs land under outputs/<project>/scalability/. The collectors read the
built graph read-only (or cached hub/stream pickles); nothing rebuilds the KG.
The per-commit build TIMING LOGS captured by build_project.py (in logs/<project>/)
feed the build-complexity analysis.

Order:
  E1  scalability/collect_kg_profile.py        final-KG statistical size profile
  E2  scalability/collect_growth_stats.py       per-commit online growth / churn / drift
  E3  scalability/time_build_replay.py           build/modify time-space complexity
  E4  scalability/time_prediction.py             predict-time latency (5 methods x 6 graphs)
  E5  scalability/stat_tests.py                   significance (DeLong / bootstrap / Holm)
  A1  scalability/collect_cstg_ablation.py        CSTG internal-component ablation
  A2  scalability/cstg_online_graph_ablation.py   5 methods over separated CSTG layers + G

Run:  KGC_PROJECT=zookeeper python drivers/run_scalability.py
Options:  --quick   pass through to the samplers/replayers where supported
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
SCAL = PKG / "scalability"


def _run(script, *args, label=""):
    cmd = [sys.executable, str(SCAL / script), *[str(a) for a in args]]
    print(f"\n{'='*70}\n[{label}] {' '.join(cmd)}\n{'='*70}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"[{label}] returned {r.returncode} (continuing).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = ["--quick"] if args.quick else []

    print(summary())
    _run("collect_kg_profile.py", label="E1 KG statistical profile")
    _run("collect_growth_stats.py", label="E2 per-commit growth / drift")
    _run("time_build_replay.py", label="E3 build/modify complexity")
    _run("time_prediction.py", *q, label="E4 predict-time latency")
    _run("stat_tests.py", *q, label="E5 statistical significance")
    _run("collect_cstg_ablation.py", label="A1 CSTG internal ablation")
    _run("cstg_online_graph_ablation.py", *q, label="A2 CSTG graph ablation")
    print(f"\n{'#'*70}\nSCALABILITY COMPLETE for '{PROJECT}'.\n"
          f"Next: python drivers/make_reports.py\n{'#'*70}")


if __name__ == "__main__":
    main()
