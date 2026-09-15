"""
Standalone cross-project aggregate FIGURES (PNG + PDF) under
outputs/aggregate/figures/. Complements the notebook's inline chart with saved
figures matching the per-project figure convention.

  * fig_type1_vs_type2   deployed fusion: macro vs commit-weighted total, per metric
  * fig_methods_final    5 methods on the final graph (macro bars)
  * fig_kg_vs_baseline   deployed fusion vs the best baseline (if baselines present)
  * fig_effort           Popt / ACC@20 per KG channel (if effort present)
  * fig_per_project      per-project deployed-fusion Buggy-F1 (spread across projects)

Run (after aggregate_projects.py):  python aggregate/make_aggregate_figures.py
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PKG_ROOT.parent / "outputs"
AGG = OUTPUTS / "aggregate"
FIG = AGG / "figures"

HEAD = ["Buggy_F1", "Macro_F1", "G_Mean", "AUC", "PR_AUC", "MCC"]
C_MACRO, C_NEVAL, C_N = "#1f6f6b", "#6a4c93", "#b5622a"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _macro(leaf, keys=HEAD):
    return [leaf[k]["macro"] if leaf.get(k) and leaf[k]["macro"] is not None else np.nan
            for k in keys]


def fig_type1_vs_type2(R):
    dep = R["fusion"]["deployed_F"]
    x = np.arange(len(HEAD)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.bar(x - w, [dep[k]["macro"] for k in HEAD], w, label="Type-1 macro", color=C_MACRO)
    ax.bar(x, [dep[k]["total_neval"] for k in HEAD], w, label="Type-2 total (n_eval)", color=C_NEVAL)
    ax.bar(x + w, [dep[k]["total_N"] for k in HEAD], w, label="Type-2 total (N)", color=C_N)
    ax.set_xticks(x); ax.set_xticklabels(HEAD, rotation=18)
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True)
    ax.set_title(f"Deployed {R['fusion']['deployed_label']}: Type-1 vs Type-2 "
                 f"across {R['meta']['n_projects']} projects")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout(); _save(fig, "fig_type1_vs_type2")


def fig_methods_final(R):
    fe = R["final_experiments"]; g = "final" if "final" in fe["graphs"] else fe["graphs"][-1]
    methods = fe["methods"]
    x = np.arange(len(methods)); w = 0.8 / len(HEAD)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for k, mk in enumerate(HEAD):
        vals = [fe["aggregated"][g][m][mk]["macro"] for m in methods]
        ax.bar(x + (k - (len(HEAD) - 1) / 2) * w, vals, w, label=mk)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, 1); ax.set_ylabel("macro score")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True)
    ax.set_title(f"Graph-inference methods on the '{g}' graph (Type-1 macro)")
    ax.legend(frameon=False, ncol=6, fontsize=8)
    fig.tight_layout(); _save(fig, "fig_methods_final")


def fig_kg_vs_baseline(R):
    if not R.get("baselines"):
        return
    dep = R["fusion"]["deployed_F"]
    bl = R["baselines"]["aggregated"]
    # best learned baseline by macro Buggy_F1
    learned = [b for b in R["baselines"]["baselines"] if b in ("B_LR", "B_RF", "B_HGB")]
    if not learned:
        return
    best = max(learned, key=lambda b: bl[b]["Buggy_F1"]["macro"] or 0)
    x = np.arange(len(HEAD)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.bar(x - w / 2, [dep[k]["macro"] for k in HEAD], w,
           label=f"KG {R['fusion']['deployed_label']}", color=C_MACRO)
    ax.bar(x + w / 2, [bl[best][k]["macro"] if bl[best].get(k) else np.nan for k in HEAD], w,
           label=f"best baseline ({best})", color="#999999")
    ax.set_xticks(x); ax.set_xticklabels(HEAD, rotation=18)
    ax.set_ylim(0, 1); ax.set_ylabel("macro score")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True)
    ax.set_title(f"KG method vs. best JIT baseline "
                 f"({R['baselines'] and len([p for p in R['projects_index'] if R['projects_index'][p].get('baselines')])} projects)")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout(); _save(fig, "fig_kg_vs_baseline")


def fig_effort(R):
    if not R.get("effort"):
        return
    ef = R["effort"]["aggregated"]; models = R["effort"]["models"]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.bar(x - w / 2, [ef[m]["Popt"]["macro"] for m in models], w, label="Popt", color=C_MACRO)
    ax.bar(x + w / 2, [ef[m]["ACC20"]["macro"] for m in models], w, label="ACC@20%LOC", color=C_N)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("macro score")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True)
    ax.set_title("Effort-aware evaluation of KG channels (Type-1 macro)")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout(); _save(fig, "fig_effort")


def fig_per_project(R):
    if not R.get("fusion"):
        return
    projs = R["meta"]["projects"]
    # per-project deployed Buggy-F1 from the F+G combo isn't per-project stored;
    # use the chosen-fusion label + N as a spread chart of project sizes instead.
    idx = R["projects_index"]
    Ns = [idx[p].get("N", 0) for p in projs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(projs, Ns, color=C_MACRO)
    ax.set_ylabel("labelled commits (N)")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True)
    for i, (p, n) in enumerate(zip(projs, Ns)):
        ax.annotate(str(R["fusion"]["chosen_per_project"].get(p, "")),
                    (i, n), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, rotation=0)
    ax.set_title("Projects aggregated (bar = size; label = chosen fusion)")
    fig.tight_layout(); _save(fig, "fig_per_project")


def main():
    R = pickle.load(open(AGG / "aggregate_results.pkl", "rb"))
    print(f"figures -> {FIG}")
    fig_type1_vs_type2(R)
    fig_methods_final(R)
    fig_kg_vs_baseline(R)
    fig_effort(R)
    fig_per_project(R)
    print("done.")


if __name__ == "__main__":
    main()
