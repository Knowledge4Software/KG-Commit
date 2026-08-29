"""
Per-METRIC radars over all 11 projects, showing the per-LAYER advantage.
========================================================================

Regenerates the metric_radar__<metric> family for the FINAL results
(final_final_run: adaptive INIT + switch), with five series per radar:

  Core                        final_experiments['core']['Fusion']   (5-method stack)
  Core+AST                    final_experiments['ast']['Fusion']    (5-method stack)
  Core+AST+CSTG (F)           final_fusion part2 'F'      -- the deployed graph fusion
  F+G                         final_fusion part2 'F+G'    -- + the CSTG channel
  Switch to F+G at 200        switch_results grid['200']  -- the deployed rule

Each radar is ONE metric; its axes are the 11 projects, so a layer that genuinely
carries signal traces a large, roughly convex polygon, while one that wins only on a
few projects shows visible dents. Everything is emitted TWICE:

  __perproj   the per-project chosen fusion F        (part2 / switch['per_project'])
  __overall   the fixed overall fusion F = RN+PPR    (part2_overall / switch['overall'])

Core and Core+AST are identical between the two variants (they are per-graph
5-method stacks, not a chosen subset); only the three F-based series differ.

Cache-only: reads final_final_run/{experiments,fusion,switch}/. No Neo4j.

Out: <paper_material>/RQ3_subgraphs/metric_radar__<metric>__<variant>.{pdf,png}
     + layer_radar_data.json  (the plotted values, for re-rendering)

Run: python inference/make_rq3_layer_radars.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ3_subgraphs"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
M7L = {"Precision": "Precision", "Recall": "Recall", "Macro_F1": "Macro-F1",
       "Buggy_F1": "Buggy-F1", "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Accuracy"}

SERIES = [
    ("Core",                 "#9e9e9e", "-",  1.6),
    ("Core+AST",             "#1f77b4", "-",  1.6),
    ("Core+AST+CSTG (F)",    "#2ca02c", "-",  1.8),
    ("F+G",                  "#ff7f0e", "-",  1.8),
    ("Switch to F+G at 200", "#d62728", "-",  2.6),
]


def collect(variant):
    """project -> series -> {metric: value}."""
    part2_key = "part2" if variant == "perproj" else "part2_overall"
    switch_key = "per_project" if variant == "perproj" else "overall"
    data, chosen = {}, {}

    for p in PROJECTS:
        base = OUTP / p / "final_final_run"
        fe = base / "experiments" / "final_experiments_results.pkl"
        ff = base / "fusion" / "final_fusion_results.pkl"
        sw = base / "switch" / "switch_results.json"
        if not (fe.exists() and ff.exists() and sw.exists()):
            print(f"  skip {p}: missing artifacts")
            continue
        E = pickle.load(open(fe, "rb"))
        F = pickle.load(open(ff, "rb"))
        S = json.load(open(sw))

        d = {}
        for lab, g in (("Core", "core"), ("Core+AST", "ast")):
            node = E.get(g, {}).get("Fusion")
            if isinstance(node, dict) and "metrics" in node:
                d[lab] = {k: float(node["metrics"].get(k, np.nan)) for k in M7}

        p2 = F.get(part2_key, {})
        if "F" in p2:
            d["Core+AST+CSTG (F)"] = {k: float(p2["F"]["metrics"].get(k, np.nan))
                                      for k in M7}
        if "F+G" in p2:
            d["F+G"] = {k: float(p2["F+G"]["metrics"].get(k, np.nan)) for k in M7}

        grid = S.get(switch_key, {}).get("grid", {})
        if "200" in grid:
            d["Switch to F+G at 200"] = {k: float(grid["200"].get(k, np.nan))
                                         for k in M7}

        data[p] = d
        chosen[p] = (F.get("chosen") if variant == "perproj"
                     else F.get("chosen_overall", "RN+PPR"))
    return data, chosen


def radar(data, metric, variant, chosen, out_stem):
    projs = [p for p in PROJECTS if p in data]
    n = len(projs)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([ang, ang[:1]])

    fig, ax = plt.subplots(figsize=(8.4, 8.4), subplot_kw=dict(polar=True))
    lo, hi = 1.0, 0.0
    for lab, col, ls, lw in SERIES:
        vals = [data[p].get(lab, {}).get(metric, np.nan) for p in projs]
        v = np.array(vals, float)
        if np.all(np.isnan(v)):
            continue
        lo = min(lo, np.nanmin(v)); hi = max(hi, np.nanmax(v))
        vc = np.concatenate([v, v[:1]])
        ax.plot(closed, vc, color=col, ls=ls, lw=lw, label=lab,
                marker="o", ms=3.4, zorder=5 if "Switch" in lab else 3)
        if "Switch" in lab:
            ax.fill(closed, vc, color=col, alpha=.08, zorder=2)

    ax.set_xticks(ang)
    ax.set_xticklabels([DISP[p] for p in projs], fontsize=9.5)
    pad = max(.02, (hi - lo) * .08)
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=.35, lw=.7)

    vname = ("per-project chosen fusion $F$" if variant == "perproj"
             else "overall fusion $F=$ RN+PPR")
    ax.set_title(f"{M7L[metric]} by KG layer — {vname}\n(11 projects)",
                 fontsize=13, fontweight="bold", pad=26)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=2,
              frameon=False, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    dump = {}
    for variant in ("perproj", "overall"):
        print(f"\n[{variant}]")
        data, chosen = collect(variant)
        dump[variant] = {"data": data, "chosen_F": chosen}
        have = {lab for p in data for lab in data[p]}
        print(f"  {len(data)} projects, series present: {sorted(have)}")
        for m in M7:
            radar(data, m, variant, chosen,
                  str(PM / f"metric_radar__{m}__{variant}"))
        print(f"  wrote {len(M7)} radars -> metric_radar__<metric>__{variant}.{{pdf,png}}")

        # per-series cross-project mean, a quick sanity read on ordering
        print(f"  {'series':<24}" + "".join(f"{M7L[m][:9]:>10}" for m in M7))
        for lab, _, _, _ in SERIES:
            row = [np.nanmean([data[p].get(lab, {}).get(m, np.nan) for p in data])
                   for m in M7]
            print(f"  {lab:<24}" + "".join(f"{v:>10.3f}" for v in row))

    json.dump(dump, open(PM / "layer_radar_data.json", "w"), indent=2)
    print(f"\nsaved -> {PM}")


if __name__ == "__main__":
    main()
