"""
Driver: build the FULL knowledge graph for the active project (KGC_PROJECT),
Phase A -> B -> C, with per-commit timing logs for the scalability RQ.

This orchestrates the ten ordered build steps as subprocesses (each inherits
KGC_PROJECT and the package on PYTHONPATH). It is deliberately per-project and
sequential -- NOT a loop over projects. Build the light projects first.

Steps (see docs / RUNBOOK.md for the full explanation):
  A1  build/build_file_ast_graphs.py       base file-AST graphs (pickles)
  A2  build/ingest_base_kg_and_ast.py       Core process layer + base AST into Neo4j
  A3  build/add_positions_to_astnodes.py    stamp source positions on base AST nodes
  A4  build/apply_commit_labels.py          apply ApacheJIT labels + JIT metrics
  B5  build/build_online_kg.py              AST delta layer (online growth)  [+timing-log]
  B6  build/build_subgraph_online_kg.py     CFG/DFG/PDG/SEQ layers (once per kind) [+timing-log]
  C7  inference/validate_cstg.py            fit CSTG -> outputs/<p>/cstg_bundle.pkl
  C8  inference/ingest_cstg.py --ground     CSTG Term/MENTIONS/COOCCURS/GROUNDS_IN into Neo4j
  C9  inference/ingest_cstg_enrich.py       Intent / HAS_INTENT typing

Prerequisites (fail early if missing):
  * KGC_PROJECT set and its base_commit filled in config/projects.yaml
    (run find_base_commit.py first for a new project)
  * Neo4j running and EMPTY for this project (run reset_neo4j.py --yes when
    switching from a previously-built project)

Run:
    (PowerShell)  $env:KGC_PROJECT='zookeeper'; python drivers/build_project.py
    (bash)        KGC_PROJECT=zookeeper python drivers/build_project.py
Options:
    --limit N     cap commits processed by the online engines (smoke test)
    --skip-cstg   build only Core+AST+subgraphs (skip Phase C)
    --from STEP   resume from a step id (a1,a2,a3,a4,b5,b6,c7,c8,c9)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (PROJECT, base_commit, TIMING_DIR, summary,
                                    REPO_PATH, CSV_PATH)

PKG = Path(__file__).resolve().parent.parent
BUILD = PKG / "build"
INFER = PKG / "inference"
SUBKINDS = ["cfg", "dfg", "pdg", "seq"]


def _run(script: Path, *args, label=""):
    cmd = [sys.executable, str(script), *[str(a) for a in args]]
    print(f"\n{'='*70}\n[{label}] {' '.join(cmd)}\n{'='*70}", flush=True)
    # subprocess inherits our env (KGC_PROJECT); ensure the package is importable
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"[{label}] FAILED (exit {r.returncode}); build stopped.")


STEPS = ["a1", "a2", "a3", "a4", "b5", "b6", "c7", "c8", "c9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-cstg", action="store_true")
    ap.add_argument("--from", dest="from_step", default="a1", choices=STEPS)
    args = ap.parse_args()

    base_commit()                          # loud check: base must be set
    print(summary())
    if not REPO_PATH.exists():
        sys.exit(f"Repo not cloned: {REPO_PATH}")
    if not CSV_PATH.exists():
        sys.exit(f"Label CSV missing: {CSV_PATH}")

    lim = ["--limit", str(args.limit)] if args.limit else []
    start = STEPS.index(args.from_step)

    def do(step):  # gate by --from
        return STEPS.index(step) >= start

    # ---- Phase A : base snapshot + Core layer ----
    if do("a1"):
        _run(BUILD / "build_file_ast_graphs.py", label="A1 base file-AST graphs")
    if do("a2"):
        _run(BUILD / "ingest_base_kg_and_ast.py", label="A2 Core + base AST -> Neo4j")
    if do("a3"):
        _run(BUILD / "add_positions_to_astnodes.py", label="A3 stamp positions")
    if do("a4"):
        _run(BUILD / "apply_commit_labels.py", label="A4 ApacheJIT labels + metrics")

    # ---- Phase B : online growth engines (with per-commit timing logs) ----
    if do("b5"):
        _run(BUILD / "build_online_kg.py", "--reset", *lim,
             "--timing-log", str(TIMING_DIR / "ast_timing.csv"),
             label="B5 AST delta layer (online)")
    if do("b6"):
        for kind in SUBKINDS:
            _run(BUILD / "build_subgraph_online_kg.py", "--kind", kind, "--reset", *lim,
                 "--timing-log", str(TIMING_DIR / f"{kind}_timing.csv"),
                 label=f"B6 {kind.upper()} layer (online)")

    # ---- Phase C : CSTG semantic-text layer ----
    if not args.skip_cstg:
        if do("c7"):
            _run(INFER / "validate_cstg.py", label="C7 fit CSTG bundle")
        if do("c8"):
            _run(INFER / "ingest_cstg.py", "--ground", label="C8 CSTG graph -> Neo4j")
        if do("c9"):
            _run(INFER / "ingest_cstg_enrich.py", label="C9 Intent / HAS_INTENT")

    print(f"\n{'#'*70}\nBUILD COMPLETE for '{PROJECT}'. "
          f"Timing logs in {TIMING_DIR}.\nNext: python drivers/run_experiments.py\n{'#'*70}")


if __name__ == "__main__":
    main()
