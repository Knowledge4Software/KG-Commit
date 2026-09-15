"""
Driver: render all tables, figures, and executed notebooks for the active project
(KGC_PROJECT). Cache-only -- reads the result pickles / JSON produced by
run_experiments.py and run_scalability.py; does NOT need a live Neo4j (except the
insight notebook / insight figures, which make read-only Term/Intent queries).

Everything lands under outputs/<project>/ (tables/, figures/, notebooks/).

Renders:
  inference/make_final_experiments_report.py   7 per-metric tables + stream figures
  inference/make_final_fusion_tables.py         fusion-ablation + F/F+G/F+M tables
  inference/make_paper_table.py                 the RQ3 evaluation matrix (this project row)
  inference/make_metrics_tables.py              headline 7-metric tables
  inference/make_metrics_figures.py             headline heatmaps
  inference/make_subgraph_tables.py             subgraph-RQ tables
  inference/make_subgraph_figures.py            subgraph-RQ figures
  scalability/make_scalability_tables.py        E1-E5 LaTeX tables
  scalability/make_scalability_figures.py       scalability figures
  scalability/make_insight_figures.py           insight figures (read-only Neo4j)
  inference/build_final_experiments_notebook.py  -> executed final_experiments.ipynb
  inference/build_subgraph_notebook.py           -> executed subgraph_ablation_v4.ipynb
  scalability/build_scalability_notebook.py      -> executed scalability_v4.ipynb
  scalability/build_insight_notebook.py          -> executed additional_experiments...ipynb

Run:  KGC_PROJECT=zookeeper python drivers/make_reports.py
Options:  --no-notebooks   render only tables/figures (skip nbconvert execution)
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
SCAL = PKG / "scalability"


def _run(script_dir, script, label=""):
    cmd = [sys.executable, str(script_dir / script)]
    print(f"\n{'='*70}\n[{label}] {' '.join(cmd)}\n{'='*70}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"[{label}] returned {r.returncode} (continuing).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-notebooks", action="store_true")
    args = ap.parse_args()

    print(summary())
    # ---- tables + figures ----
    for s, lbl in [
        ("make_final_experiments_report.py", "final per-metric tables + streams"),
        ("make_final_fusion_tables.py", "fusion tables"),
        ("make_paper_table.py", "RQ3 evaluation-matrix row"),
        ("make_metrics_tables.py", "headline metric tables"),
        ("make_metrics_figures.py", "headline heatmaps"),
        ("make_subgraph_tables.py", "subgraph-RQ tables"),
        ("make_subgraph_figures.py", "subgraph-RQ figures"),
        ("make_kg_vs_baseline.py", "KG-vs-baseline 7 stream figures + table"),
        ("make_param_report.py", "parameter-sensitivity tables + figures"),
    ]:
        _run(INFER, s, label=lbl)
    for s, lbl in [
        ("make_scalability_tables.py", "scalability tables"),
        ("make_scalability_figures.py", "scalability figures"),
        ("make_insight_figures.py", "insight figures (read-only Neo4j)"),
    ]:
        _run(SCAL, s, label=lbl)

    # ---- executed notebooks (build + nbconvert happens inside each builder) ----
    if not args.no_notebooks:
        _run(INFER, "build_final_experiments_notebook.py", label="final_experiments.ipynb")
        _run(INFER, "build_subgraph_notebook.py", label="subgraph_ablation_v4.ipynb")
        _run(SCAL, "build_scalability_notebook.py", label="scalability_v4.ipynb")
        _run(SCAL, "build_insight_notebook.py", label="additional_experiments...ipynb")

    print(f"\n{'#'*70}\nREPORTS COMPLETE for '{PROJECT}'. "
          f"See outputs/{PROJECT}/ (tables, figures, notebooks).\n{'#'*70}")


if __name__ == "__main__":
    main()
