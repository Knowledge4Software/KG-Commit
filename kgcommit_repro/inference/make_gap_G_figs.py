#!/usr/bin/env python3
"""
Robustness to the verification-latency gap G.
=============================================

G is the fourth online-protocol knob in D4 ("Gap, refit window, warmup") and the
one missing from the robustness folder. It is a genuine EVALUATION-protocol
parameter, unlike ROLL: a commit's label is only usable for tuning once G later
commits have arrived, so when the operating threshold is re-tuned at commit i
only labels up to i-G are revealed. G=0 means labels are available immediately.

Because G only changes which labels are visible to the online THRESHOLD tuner,
it can be swept exactly from the cached per-commit score vectors -- no refit, no
Neo4j. The model's scores p are fixed; only the decision threshold moves.

Consequence worth stating in the paper: threshold-FREE metrics (AUC, PR-AUC) and
metrics taken at the fixed 0.5 cut (MCC, Brier) are mathematically invariant to
G. Only the online-tuned operating-point metrics -- Precision, Recall, Buggy-F1,
Macro-F1, G-Mean, Accuracy -- can move. Both groups are emitted; the invariant
ones are the control showing the sweep is behaving as designed.

Out: Paper/paper_material/discussions/Robustness/
       figures/fig_robust_perproject_G_<metric>.{pdf,png,csv}
       figures/fig_robust_overall_<agg>_G_<metric>.{pdf,png,csv}
       Source_data/gap_G_sweep/<project>__G.json
Run: python kgcommit_repro/inference/make_gap_G_figs.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# online_jit's import chain reads the single-project config, which refuses to
# load unless KGC_PROJECT is set. This script is cross-project and uses only the
# pure-numpy metric helpers (final_metrics / online_decisions) -- no Neo4j
# connection is opened at import -- so a placeholder satisfies the import and
# nothing project-specific is read from it.
import os  # noqa: E402

os.environ.setdefault("KGC_PROJECT", "zookeeper")

import _kgc_paths  # noqa: E402,F401
import online_jit as OJ  # noqa: E402
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
DEST = ROOT / "Paper" / "paper_material" / "discussions" / "Robustness"
# figures sit at the folder root alongside the existing K/M sweeps; the raw
# per-project sweep goes under Source_data/ with the other inputs
FIGD, SRCD = DEST, DEST / "Source_data" / "gap_G_sweep"

G_GRID = [0, 10, 25, 50, 100, 200]
DEPLOYED_G = 50

# tuned-threshold metrics: these are the ones G can actually move
TUNED = {
    "Macro_F1": "Macro-F1",
    "G_Mean": "G-Mean",
    "Buggy_F1": "F1 (buggy class)",
    "Precision": "Precision",
    "Recall": "Recall",
    "ACC": "Accuracy",
}
# threshold-free / fixed-cut metrics: invariant to G by construction
INVARIANT = {"AUC": "AUC", "PR_AUC": "PR-AUC", "MCC": "MCC"}
METRICS = {**TUNED, **INVARIANT}

AGGS = {"macro": "Macro-average", "micro": "Micro-average", "total": "Total"}


def load_scores(folder):
    """Cached per-commit scores of the deployed F+G model."""
    f = OUTP / folder / "final_final_run" / "fusion" / "raw_fusion_scores.pkl"
    if not f.exists():
        return None
    import pickle
    d = pickle.load(open(f, "rb"))
    y = np.asarray(d["y"], int)
    sc = d.get("scores", {})
    if "F+G" not in sc:
        return None
    return y, np.asarray(sc["F+G"], float)


def sweep(folder):
    got = load_scores(folder)
    if got is None:
        return None
    y, p = got
    if len(np.unique(y)) < 2:
        return None
    out = {}
    for g in G_GRID:
        m = OJ.final_metrics(y, p, gap=g)
        out[str(g)] = {k: (None if m.get(k) is None or not np.isfinite(m.get(k, np.nan))
                           else float(m[k])) for k in METRICS}
    out["_meta"] = {"knob": "GAP", "grid": G_GRID, "deployed": DEPLOYED_G,
                    "n_eval": int(len(y)), "n_buggy": int(y.sum()),
                    "source": f"outputs/{folder}/final_final_run/fusion/"
                              "raw_fusion_scores.pkl (scores['F+G'])",
                    "note": "G moves only the online-tuned threshold; the model "
                            "scores are fixed. AUC/PR_AUC/MCC are invariant."}
    return out


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save(fig, stem):
    # PNG + CSV only, matching the K/M figures already in this folder
    FIGD.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGD / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_per_project(metric, data):
    label = METRICS[metric]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    rows, drawn = [], 0
    for i, (disp, d) in enumerate(sorted(data.items())):
        xs, ys = [], []
        for g in G_GRID:
            v = d.get(str(g), {}).get(metric)
            if v is None:
                continue
            xs.append(g)
            ys.append(v)
            rows.append([disp, g, v])
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, marker=markers[i % len(markers)], markersize=4.5,
                linewidth=1.4, color=cmap(i % 20), label=disp, alpha=0.9)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None

    ax.axvline(DEPLOYED_G, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.annotate(f"deployed G={DEPLOYED_G}", (DEPLOYED_G, 0.02),
                xycoords=("data", "axes fraction"), ha="left", va="bottom",
                fontsize=7.5, color="#d62728",
                xytext=(4, 0), textcoords="offset points")
    ax.set_xticks(G_GRID)
    ax.set_xlabel("verification-latency gap $G$ (commits whose labels are withheld "
                  "from the\nthreshold tuner; $G{=}0$ = labels available immediately)",
                  fontsize=9)
    ax.set_ylabel(label)
    inv = " — invariant to $G$ by construction" if metric in INVARIANT else ""
    ax.set_title(f"Robustness to the gap $G$: per-project {label}{inv}", fontsize=10.5)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9, loc="best")
    fig.tight_layout()

    stem = f"fig_robust_perproject_G_{metric}"
    save(fig, stem)
    write_csv(FIGD / f"{stem}.csv", ["project", "G", metric], rows)
    return stem


def fig_overall(metric, agg, data, weights):
    label = METRICS[metric]
    folder = {d: f for d, f in PROJECTS}
    xs, ys, rows = [], [], []
    for g in G_GRID:
        num = den = 0.0
        n_proj = 0
        for disp, d in data.items():
            v = d.get(str(g), {}).get(metric)
            if v is None:
                continue
            n, pos = weights.get(folder.get(disp, ""), (0, 0))
            w = 1.0 if agg == "macro" else (float(n) if agg == "micro" else float(pos))
            if w <= 0:
                continue
            num += w * v
            den += w
            n_proj += 1
        if den:
            xs.append(g)
            ys.append(num / den)
            rows.append([g, num / den, n_proj])
    if len(xs) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(xs, ys, marker="o", markersize=5.5, linewidth=1.9, color="#1f77b4")
    ax.axvline(DEPLOYED_G, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.annotate(f"deployed G={DEPLOYED_G}", (DEPLOYED_G, 1.005),
                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                fontsize=7.5, color="#d62728")
    ax.set_xticks(G_GRID)
    ax.set_xlabel("verification-latency gap $G$ (commits)", fontsize=9)
    ax.set_ylabel(f"{label} — {AGGS[agg]}")
    ax.set_title(f"Overall robustness to $G$: {label} ({AGGS[agg]}, "
                 f"n={max(r[2] for r in rows)} projects)", fontsize=10.5)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()

    stem = f"fig_robust_overall_{agg}_G_{metric}"
    save(fig, stem)
    write_csv(FIGD / f"{stem}.csv", ["G", f"{metric}_{agg}", "n_projects"], rows)
    return stem


def main():
    data, weights = {}, {}
    SRCD.mkdir(parents=True, exist_ok=True)
    for disp, folder in PROJECTS:
        res = sweep(folder)
        if not res:
            print(f"  {disp}: no cached F+G scores, skipped")
            continue
        data[disp] = res
        weights[folder] = (res["_meta"]["n_eval"], res["_meta"]["n_buggy"])
        (SRCD / f"{folder}__G.json").write_text(json.dumps(res, indent=2))
    print(f"swept G on {len(data)} projects: {', '.join(sorted(data))}")

    made = []
    for metric in METRICS:
        s = fig_per_project(metric, data)
        if s:
            made.append(s)
        for agg in AGGS:
            s = fig_overall(metric, agg, data, weights)
            if s:
                made.append(s)

    # how much does each metric actually move across the grid?
    spread = {}
    for metric in METRICS:
        per = []
        for d in data.values():
            vals = [d.get(str(g), {}).get(metric) for g in G_GRID]
            vals = [v for v in vals if v is not None]
            if len(vals) > 1:
                per.append(max(vals) - min(vals))
        if per:
            spread[metric] = {"mean_range": float(np.mean(per)),
                              "max_range": float(np.max(per))}

    (DEST / "MANIFEST_G.json").write_text(json.dumps({
        "knob": "G (verification-latency gap)",
        "grid": G_GRID, "deployed": DEPLOYED_G,
        "requires_neo4j": False,
        "run": "python kgcommit_repro/inference/make_gap_G_figs.py",
        "computed_from": "outputs/<p>/final_final_run/fusion/raw_fusion_scores.pkl"
                         "  ->  scores['F+G'], re-thresholded via "
                         "online_jit.final_metrics(gap=G)",
        "invariant_metrics": sorted(INVARIANT),
        "tuned_metrics": sorted(TUNED),
        "projects": sorted(data),
        "per_metric_range_across_G": spread,
        "figures": sorted(made),
    }, indent=2))

    print(f"\n{len(made)} figures -> {FIGD}")
    print(f"{len(data)} sweep files -> {SRCD}")
    print("\nmean range across the G grid (per project, averaged):")
    for m, s in sorted(spread.items(), key=lambda kv: -kv[1]["mean_range"]):
        tag = "  (invariant)" if m in INVARIANT else ""
        print(f"  {METRICS[m]:<18} {s['mean_range']:.4f}{tag}")


if __name__ == "__main__":
    main()
