#!/usr/bin/env python3
"""
MCC across every comparison the paper makes.
============================================

Matthews Correlation Coefficient is the metric least fooled by the class
imbalance in JIT-SDP, so this pulls it out of the caches for all five
comparison axes and puts them side by side.

Cache-only. Reads, per project:
  final_final_run/experiments/final_experiments_results.pkl   layers + methods
  final_final_run/experiments/subgraph_rq_results.pkl         Layer-2 subgraphs
  final_final_run/switch/switch_results.json                  switch F / F+G / S grid
  baseline_results.pkl, baseline_extra_results.pkl            baselines
No Neo4j, no graph build, no refit.

Out: Paper/paper_material/discussions/MCC/
       tables/*.csv                 one tidy CSV per comparison (+ aggregate rows)
       figures/*.{pdf,png}          per-comparison figures
       MANIFEST.json, README.md
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
DEST = ROOT / "Paper" / "paper_material" / "discussions" / "MCC"
TABD, FIGD = DEST / "tables", DEST / "figures"

M = "MCC"

LAYERS = [("core", "Core"), ("ast", "Core+AST"), ("final", "Core+AST+CSTG")]
METHODS = [("RN", "RN"), ("PPR", "PPR"), ("LP", "LP"), ("DW", "DW"),
           ("KGE", "KGE"), ("Fusion", "Fusion")]
SUBGRAPHS = [("V1_none", "Core only"), ("V2a_cfg", "+CFG"), ("V2b_dfg", "+DFG"),
             ("V2c_pdg", "+PDG"), ("V2d_seq", "+SEQ"), ("V3_ast", "+AST")]
BASELINES = [("B_LR", "LR"), ("B_RF", "RF"), ("B_HGB", "HGB"),
             ("B_LAPREDICT", "LApredict"), ("B_DEEPER", "Deeper"),
             ("B_JITLINE", "JITLine"), ("B_ALL1", "All-Buggy"),
             ("B_ALL0", "All-Benign"), ("B_RATE", "Base-rate")]


def load(path):
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text())
        return pickle.load(open(path, "rb"))
    except Exception:
        return None


def clean(v):
    """A finite float, or None. MCC is defined on [-1, 1]."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def weights():
    """(n_eval, n_buggy) per project, for micro / total aggregation."""
    w = {}
    for _d, f in PROJECTS:
        b = load(OUTP / f / "baseline_results.pkl") or {}
        n = b.get("n_eval")
        rate = b.get("bug_rate")
        if n and rate is not None:
            w[f] = (int(n), int(round(n * float(rate))))
        else:
            w[f] = (0, 0)
    return w


# ------------------------------------------------------------------ collectors
def collect_layers():
    rows = {}
    for disp, f in PROJECTS:
        E = load(OUTP / f / "final_final_run/experiments/final_experiments_results.pkl")
        if not E:
            continue
        r = {}
        for key, lab in LAYERS:
            node = (E.get(key) or {}).get("Fusion") or {}
            r[lab] = clean((node.get("metrics") or {}).get(M))
        rows[disp] = r
    return rows, [l for _k, l in LAYERS]


def collect_methods():
    rows = {}
    for disp, f in PROJECTS:
        E = load(OUTP / f / "final_final_run/experiments/final_experiments_results.pkl")
        if not E:
            continue
        fin = E.get("final") or {}
        rows[disp] = {lab: clean(((fin.get(k) or {}).get("metrics") or {}).get(M))
                      for k, lab in METHODS}
    return rows, [l for _k, l in METHODS]


def collect_subgraphs():
    rows = {}
    for disp, f in PROJECTS:
        SG = load(OUTP / f / "final_final_run/experiments/subgraph_rq_results.pkl")
        if not SG:
            continue
        rows[disp] = {lab: clean(((SG.get(k) or {}).get("Fusion") or {}).get(M))
                      for k, lab in SUBGRAPHS}
    return rows, [l for _k, l in SUBGRAPHS]


def collect_baselines():
    """Baselines plus the deployed KG model, so the contrast is on one row."""
    rows = {}
    labs = [l for _k, l in BASELINES] + ["KG-Commit (F+G)"]
    for disp, f in PROJECTS:
        r = {}
        for src in ("baseline_results.pkl", "baseline_extra_results.pkl"):
            d = load(OUTP / f / src) or {}
            for k, lab in BASELINES:
                v = clean((d.get("baselines", {}).get(k) or {}).get(M))
                if v is not None:
                    r[lab] = v
        sw = load(OUTP / f / "final_final_run/switch/switch_results.json") or {}
        r["KG-Commit (F+G)"] = clean(((sw.get("per_project") or {})
                                      .get("FG_only") or {}).get(M))
        if r:
            rows[disp] = r
    return rows, labs


def collect_switch():
    """F alone vs F+G vs the deployed S=200 switch."""
    rows = {}
    labs = ["$F$ only", "$F{+}G$", "Switch@200"]
    for disp, f in PROJECTS:
        sw = load(OUTP / f / "final_final_run/switch/switch_results.json") or {}
        node = sw.get("per_project") or {}
        grid = node.get("grid") or {}
        rows[disp] = {
            "$F$ only": clean((node.get("F_only") or {}).get(M)),
            "$F{+}G$": clean((node.get("FG_only") or {}).get(M)),
            "Switch@200": clean((grid.get("200") or {}).get(M)),
        }
    return rows, labs


# ------------------------------------------------------------------ aggregation
def add_aggregates(rows, labs, w):
    """Append Macro / Micro / Total average rows, computed per column."""
    out = dict(rows)
    folder = {d: f for d, f in PROJECTS}
    for name, mode in (("Average (macro)", "macro"),
                       ("Average (micro)", "micro"),
                       ("Average (total)", "total")):
        agg = {}
        for lab in labs:
            num = den = 0.0
            for disp, r in rows.items():
                v = r.get(lab)
                if v is None:
                    continue
                n, pos = w.get(folder.get(disp, ""), (0, 0))
                wt = 1.0 if mode == "macro" else (float(n) if mode == "micro" else float(pos))
                if wt <= 0:
                    continue
                num += wt * v
                den += wt
            agg[lab] = num / den if den else None
        out[name] = agg
    return out


def write_csv(name, rows, labs):
    TABD.mkdir(parents=True, exist_ok=True)
    p = TABD / f"{name}.csv"
    with p.open("w", newline="") as fh:
        fh.write("project," + ",".join(labs) + "\n")
        for disp, r in rows.items():
            cells = ["" if r.get(l) is None else f"{r[l]:.6f}" for l in labs]
            fh.write(f"{disp}," + ",".join(cells) + "\n")
    return p


def fig_grouped(name, title, rows, labs):
    """Grouped bars: projects on x, one bar per comparison arm. Averages last."""
    projs = [p for p in rows if not p.startswith("Average")]
    avgs = [p for p in rows if p.startswith("Average")]
    order = projs + avgs
    if not order:
        return None

    FIGD.mkdir(parents=True, exist_ok=True)
    n = len(labs)
    x = np.arange(len(order), dtype=float)
    # visual gap between the projects and the aggregate rows
    for i in range(len(projs), len(order)):
        x[i] += 0.6
    width = 0.8 / n
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(order) + 2.6), 4.4))
    for j, lab in enumerate(labs):
        vals = [rows[p].get(lab) for p in order]
        xs = [x[i] + (j - (n - 1) / 2) * width for i in range(len(order))]
        ax.bar(xs, [0 if v is None else v for v in vals], width,
               label=lab, color=cmap(j % 20), edgecolor="white", linewidth=0.4)

    ax.axhline(0, color="#444", linewidth=0.9)
    if avgs:
        ax.axvline((x[len(projs) - 1] + x[len(projs)]) / 2, color="#999",
                   linestyle=":", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("MCC")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=7.5, ncol=min(n, 5), framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return FIGD / f"{name}.png"


def fig_switch_delta(rows):
    """The switch's MCC gain over its two endpoints -- the D3 claim, in MCC."""
    projs = [p for p in rows if not p.startswith("Average")]
    if not projs:
        return None
    dF, dG, keep = [], [], []
    for p in projs:
        r = rows[p]
        s, f_, g = r.get("Switch@200"), r.get("$F$ only"), r.get("$F{+}G$")
        if None in (s, f_, g):
            continue
        keep.append(p)
        dF.append(s - f_)
        dG.append(s - g)
    if not keep:
        return None

    FIGD.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(max(7.5, 0.95 * len(keep) + 2.4), 4.0))
    ax.bar(x - 0.2, dF, 0.4, label=r"Switch@200 $-$ $F$ only", color="#2ca02c")
    ax.bar(x + 0.2, dG, 0.4, label=r"Switch@200 $-$ $F{+}G$", color="#ff7f0e")
    ax.axhline(0, color="#444", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(keep, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\Delta$ MCC")
    ax.set_title("What the switch buys, in MCC (positive = switch is better)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"fig_mcc_switch_delta.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return True


def main():
    w = weights()
    DEST.mkdir(parents=True, exist_ok=True)

    specs = [
        ("mcc_layers", "MCC by representation layer (deployed fusion)", collect_layers),
        ("mcc_methods", "MCC by inference method on the full KG", collect_methods),
        ("mcc_subgraphs", "MCC by Layer-2 structural subgraph", collect_subgraphs),
        ("mcc_baselines", "MCC: KG-Commit vs. baselines", collect_baselines),
        ("mcc_switch", "MCC: $F$ vs $F{+}G$ vs Switch@200", collect_switch),
    ]

    man = {"metric": "MCC", "requires_neo4j": False,
           "run": "python kgcommit_repro/inference/make_mcc_report.py",
           "sources": [
               "outputs/<p>/final_final_run/experiments/final_experiments_results.pkl",
               "outputs/<p>/final_final_run/experiments/subgraph_rq_results.pkl",
               "outputs/<p>/final_final_run/switch/switch_results.json",
               "outputs/<p>/baseline_results.pkl",
               "outputs/<p>/baseline_extra_results.pkl",
           ],
           "aggregations": {
               "macro": "unweighted mean over projects",
               "micro": "weighted by each project's evaluated commit count",
               "total": "weighted by each project's buggy commit count"},
           "comparisons": {}, "project_weights": {
               f: {"n_eval": n, "n_buggy": p} for f, (n, p) in w.items()}}

    for name, title, fn in specs:
        rows, labs = fn()
        if not rows:
            print(f"  {name}: no data")
            continue
        rows = add_aggregates(rows, labs, w)
        write_csv(name, rows, labs)
        fig_grouped(f"fig_{name}", title, rows, labs)
        man["comparisons"][name] = {
            "title": title, "arms": labs,
            "projects": [p for p in rows if not p.startswith("Average")],
            "table": f"tables/{name}.csv", "figure": f"figures/fig_{name}.pdf"}
        macro = rows.get("Average (macro)", {})
        best = max((l for l in labs if macro.get(l) is not None),
                   key=lambda l: macro[l], default=None)
        print(f"  {name}: {len(labs)} arms, "
              f"{len([p for p in rows if not p.startswith('Average')])} projects"
              + (f"  best(macro)={best} {macro[best]:.4f}" if best else ""))

    sw, _ = collect_switch()
    if sw:
        fig_switch_delta(add_aggregates(sw, ["$F$ only", "$F{+}G$", "Switch@200"], w))

    # --- baselines, restricted to projects where EVERY arm ran ---------------
    # Three deep baselines are missing on Flink/HBase/Hive (and Deeper on two
    # more). Averaging over each arm's own project subset compares different
    # sets of projects, which flatters whichever baseline skipped the harder
    # ones. This table repeats the comparison on the common subset only.
    brows, blabs = collect_baselines()
    common = {d: r for d, r in brows.items()
              if all(r.get(l) is not None for l in blabs)}
    if common:
        cr = add_aggregates(common, blabs, w)
        write_csv("mcc_baselines_common_subset", cr, blabs)
        fig_grouped("fig_mcc_baselines_common_subset",
                    f"MCC: KG-Commit vs. baselines "
                    f"(common subset, n={len(common)} projects, all arms present)",
                    cr, blabs)
        man["comparisons"]["mcc_baselines_common_subset"] = {
            "title": "baselines on the common subset",
            "arms": blabs, "projects": sorted(common),
            "excluded": sorted(set(brows) - set(common)),
            "why": "three deep baselines did not run on Flink/HBase/Hive; the "
                   "all-projects average therefore compares different project "
                   "sets per arm",
            "table": "tables/mcc_baselines_common_subset.csv",
            "figure": "figures/fig_mcc_baselines_common_subset.pdf"}
        print(f"  mcc_baselines_common_subset: {len(common)} projects "
              f"(excluded {', '.join(sorted(set(brows) - set(common)))})")

    (DEST / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    print(f"\ntables  -> {TABD}\nfigures -> {FIGD}")


if __name__ == "__main__":
    main()
