"""
Close the cache-only artifact gaps identified in docs/Missing_artifacts_audit.tex.

Runs ONLY scripts that read cached pickles/JSON -- no Neo4j, no rebuild -- so it is
safe to run on a laptop while a build occupies the server's graph database.

Note: the ast_method graph and the V2e_ast_method subgraph variant were dropped from
the final methodology; the generators exclude them, so nothing here emits AST-m
rows/columns/figures even for projects whose pickles still carry that data.

Skips any project/step whose outputs already exist, so it is re-runnable and can be
interrupted freely.

Usage:
    python run_local_gaps.py                      # all projects with gaps
    python run_local_gaps.py --only hive flink    # restrict
    python run_local_gaps.py --dry-run            # show what would run
    python run_local_gaps.py --force              # re-run even if outputs exist
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
OUTPUTS = ROOT / "outputs"

PY = sys.executable

# step id -> (script relative to PKG, [sentinel outputs relative to outputs/<p>/])
STEPS = {
    "final_report": (
        "inference/make_final_experiments_report.py",
        ["tables/v4/final_experiments.csv",
         "tables/v4/tab_final_bymetric.tex",
         "figures/v4/final/fig_final_bygraph_core.png"],
    ),
    "param_sweep": (
        "inference/run_param_experiments.py",
        ["param_experiments/K.json",
         "param_experiments/M.json",
         "param_experiments/ROLL.json"],
    ),
    "param_report": (
        "inference/make_param_report.py",
        ["tables/v4/tab_param_K.tex",
         "figures/v4/param/fig_param_all.png"],
    ),
    "metrics_tables": (
        "inference/make_metrics_tables.py",
        ["tables/v4/metrics_all.csv", "tables/v4/tab_metrics_fusion.tex"],
    ),
    "metrics_figures": (
        "inference/make_metrics_figures.py",
        ["figures/v4/fig_metrics_fusion.png",
         "figures/v4/fig_metrics_allmethods.png"],
    ),
    "subgraph_tables": (
        "inference/make_subgraph_tables.py",
        ["tables/v4/tab_results.tex", "tables/v4/tab_layers.tex"],
    ),
    "subgraph_figures": (
        "inference/make_subgraph_figures.py",
        ["figures/v4/fig_permethod_bars.png",
         "figures/v4/fig_layer_sizes.png",
         "figures/v4/fig_trajectory.png"],
    ),
    "fusion_tables": (
        "inference/make_final_fusion_tables.py",
        ["tables/v4/tab_final_fusion_ablation.tex",
         "tables/v4/tab_final_fusion_gm.tex"],
    ),
    "kg_vs_baseline": (
        "inference/make_kg_vs_baseline.py",
        ["tables/v4/tab_kg_vs_baseline.tex"],
    ),
    "paper_table": (
        "inference/make_paper_table.py",
        ["tables/v4/tab_paper_groovy.tex"],
    ),
    "scal_tables": (
        "scalability/make_scalability_tables.py",
        ["scalability/tab_significance.tex"],
    ),
    "scal_figures": (
        "scalability/make_scalability_figures.py",
        ["figures/v4/scalability/fig_build_cost.png"],
    ),
}

# Per the audit: which projects still have cache-only gaps worth closing.
# groovy is excluded -- it is being rebuilt on the server.
DEFAULT_PROJECTS = ["hive", "flink", "hbase", "cassandra", "camel",
                    "activemq", "kafka", "spark", "zeppelin", "zookeeper"]

ORDER = ["final_report", "metrics_tables", "metrics_figures",
         "subgraph_tables", "subgraph_figures", "fusion_tables",
         "kg_vs_baseline", "paper_table",
         "param_sweep", "param_report",
         "scal_tables", "scal_figures"]


def have_all(project, sentinels):
    return all((OUTPUTS / project / s).exists() for s in sentinels)


def run(project, step, dry=False):
    script, sentinels = STEPS[step]
    path = PKG / script
    if not path.exists():
        return "noscript", 0.0
    env = dict(os.environ)
    env["KGC_PROJECT"] = project
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    if dry:
        return "would-run", 0.0
    t0 = time.time()
    r = subprocess.run([PY, str(path)], env=env, cwd=str(PKG),
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        msg = tail[-1][:160] if tail else f"exit {r.returncode}"
        return f"FAIL: {msg}", dt
    return "ok", dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--steps", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    projects = a.only if a.only else DEFAULT_PROJECTS
    steps = a.steps if a.steps else ORDER

    print(f"outputs root : {OUTPUTS}")
    print(f"projects     : {', '.join(projects)}")
    print(f"steps        : {len(steps)}")
    print("=" * 74)

    totals = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    for p in projects:
        if not (OUTPUTS / p).is_dir():
            print(f"\n### {p}: no outputs dir -- skipped")
            continue
        print(f"\n### {p}")
        for st in steps:
            if st not in STEPS:
                continue
            _, sentinels = STEPS[st]
            if not a.force and have_all(p, sentinels):
                print(f"   [skip] {st:<18} (outputs present)")
                totals["skip"] += 1
                continue
            status, dt = run(p, st, dry=a.dry_run)
            if status == "ok":
                print(f"   [ok  ] {st:<18} {dt:6.1f}s")
                totals["ok"] += 1
            elif status in ("would-run", "noscript"):
                print(f"   [{status}] {st}")
            else:
                print(f"   [FAIL] {st:<18} {status}")
                totals["fail"] += 1
                failures.append((p, st, status))

    print("\n" + "=" * 74)
    print(f"ran={totals['ok']}  skipped={totals['skip']}  failed={totals['fail']}")
    if failures:
        print("\nfailures:")
        for p, st, msg in failures:
            print(f"  {p}/{st}: {msg}")


if __name__ == "__main__":
    main()
