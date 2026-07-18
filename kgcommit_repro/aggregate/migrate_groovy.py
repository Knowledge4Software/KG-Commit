"""
Migrate the first-phase groovy results into the standard per-project layout.
==========================================================================

Groovy is the ORIGINAL first-phase project: it was built before the
reproducibility package introduced per-project namespacing (OUT = outputs/<p>/),
so its result files live at the FLAT outputs/ root instead of outputs/groovy/.
That is why the cross-project aggregator (which discovers projects by
outputs/<name>/final_experiments_results.pkl) silently skips groovy.

This shim folds groovy in WITHOUT a rebuild, doing as much matching as possible:

  COPIED (schema-verified identical to the package format):
    * final_experiments_results.pkl, final_fusion_results.pkl,
      subgraph_rq_results.pkl, subgraph_kg_methods_results.pkl,
      subgraph_layer_stats.json      (outputs/  ->  outputs/groovy/)
    * scalability/*.json               (outputs/scalability/ -> outputs/groovy/scalability/)

  RECONSTRUCTED from groovy's cached streams (no Neo4j; same code the package
  uses per project, so the numbers are directly comparable):
    * baseline_results.pkl   (within-project JIT baselines)
    * effort_results.pkl     (Popt / ACC@20%LOC of the KG channels)

  NOT reconstructable without re-running groovy through the package (documented,
  left for the future full re-run):
    * raw_stream inside subgraph_rq_results.pkl  -> so groovy cannot yet join the
      genuinely-POOLED fusion metric (Type-2 weighted average still includes it).
    * B-resolution trajectories  -> groovy's stored trajectories are at the old
      stride/window; harmless for the aggregate (which reads the metric leaves).
    * neo4j snapshot + per-commit timing logs -> first-phase artifacts the flat
      layout never produced; only matter for restore / RQ2-scalability replay.

Idempotent: safe to re-run. Existing outputs/groovy/ files are overwritten only
for the migrated set. A future `KGC_PROJECT=groovy python drivers/run_experiments.py`
(after building groovy's graph) supersedes everything here with native artifacts.

Run:  python aggregate/migrate_groovy.py
"""
import shutil
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PKG_ROOT.parent
ROOT_OUT = PROJECT_ROOT / "outputs"        # flat first-phase layout
GV_OUT = ROOT_OUT / "groovy"               # target per-project layout
LABEL_CSV = PROJECT_ROOT / "data" / "apachejit" / "projects" / "apache_groovy.csv"

# package code for the reconstructions (import needs KGC_PROJECT; stub it since
# these functions are project-agnostic and we drive them with groovy's cache).
import os
os.environ.setdefault("KGC_PROJECT", "groovy")
sys.path.insert(0, str(PKG_ROOT / "inference"))
sys.path.insert(0, str(PKG_ROOT / "baselines"))
sys.path.insert(0, str(PKG_ROOT))

CORE_FILES = ["final_experiments_results.pkl", "final_fusion_results.pkl",
              "subgraph_rq_results.pkl", "subgraph_kg_methods_results.pkl",
              "subgraph_layer_stats.json"]


def copy_core():
    GV_OUT.mkdir(parents=True, exist_ok=True)
    done = []
    for f in CORE_FILES:
        src = ROOT_OUT / f
        if src.exists():
            shutil.copy2(src, GV_OUT / f)
            done.append(f)
        else:
            print(f"  !! missing at root: {f}")
    return done


def copy_scalability():
    src = ROOT_OUT / "scalability"
    dst = GV_OUT / "scalability"
    if not src.exists():
        return []
    dst.mkdir(parents=True, exist_ok=True)
    done = []
    for j in src.glob("*.json"):
        shutil.copy2(j, dst / j.name)
        done.append(j.name)
    return done


def reconstruct_baselines():
    """Run the package baselines on groovy's label CSV (no Neo4j)."""
    import run_baselines as RB          # baselines/run_baselines.py
    # run_baselines.main() reads CSV_PATH/OUT from config for groovy (KGC_PROJECT
    # is stubbed to groovy) and writes outputs/groovy/baseline_results.pkl.
    RB.main()
    return (GV_OUT / "baseline_results.pkl").exists()


def reconstruct_effort():
    """Run the package effort eval on groovy's cached streams (no Neo4j).
    The streams cache is at the FLAT root; the effort script reads OUT (=
    outputs/groovy). Stage the cache into outputs/groovy/ first if absent."""
    stream = GV_OUT / "online_jit_streams_v5.pkl"
    if not stream.exists() and (ROOT_OUT / "online_jit_streams_v5.pkl").exists():
        shutil.copy2(ROOT_OUT / "online_jit_streams_v5.pkl", stream)
    import run_effort_eval as RE         # inference/run_effort_eval.py
    RE.main()
    return (GV_OUT / "effort_results.pkl").exists()


def main():
    print(f"Migrating groovy first-phase results -> {GV_OUT}\n")

    print("[1] copy core result files (schema-identical):")
    for f in copy_core():
        print(f"    copied {f}")

    print("[2] copy scalability JSONs:")
    scal = copy_scalability()
    print(f"    copied {len(scal)} scalability json(s)")

    print("[3] reconstruct baselines from label CSV (no Neo4j):")
    try:
        ok = reconstruct_baselines()
        print(f"    baseline_results.pkl: {'OK' if ok else 'FAILED'}")
    except Exception as e:
        print(f"    baselines skipped: {type(e).__name__}: {e}")

    print("[4] reconstruct effort eval from cached streams (no Neo4j):")
    try:
        ok = reconstruct_effort()
        print(f"    effort_results.pkl: {'OK' if ok else 'FAILED'}")
    except Exception as e:
        print(f"    effort skipped: {type(e).__name__}: {e}")

    print("\n[!] NOT migrated (need a future full groovy re-run through the package):")
    print("    - raw_stream in subgraph_rq_results.pkl  (=> not in POOLED fusion yet)")
    print("    - B-resolution trajectories (stored ones are old stride/window)")
    print("    - neo4j snapshot + per-commit timing logs (first-phase never produced them)")

    print("\nDone. Now re-run: python drivers/run_aggregate.py  to fold groovy in.")


if __name__ == "__main__":
    main()
