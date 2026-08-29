"""
Tables + stream figures for the final experiments (run_final_experiments.py):
five graph-inference methods on six knowledge graphs, seven metrics.

Writes to outputs/tables/v4/:
  tab_final_bymetric.tex   seven tables (one per metric): 5 methods x 6 graphs
  final_experiments.csv    long-form dump
And to outputs/figures/v4/final/ (png for the notebook, pdf for the paper):
  fig_final_bymethod_<M>   35 stream subplots = 7 metrics x 5 methods,
                           each subplot's trends = the six graphs
  fig_final_bygraph_<G>    42 stream subplots = 7 metrics x 6 graphs,
                           each subplot's trends = the five methods

Run:  python inference/make_final_experiments_report.py
"""
import pickle, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import OUT  # per-project outputs/<project>/
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--outdir-suffix", type=str, default="")
args = ap.parse_args()

out_dir = OUT if not args.outdir_suffix else OUT / args.outdir_suffix
TAB = out_dir / "tables" / "v4"; FIG = out_dir / "figures" / "v4" / "final"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

R = pickle.load(open(out_dir / "final_experiments_results.pkl", "rb"))
# ast_method (the per-method AST variant) was DROPPED from the final methodology, so
# the main tree stays on the final 6-graph family even when a project's pickle still
# carries ast_method data. The dedicated v4_ast_method/ side branch is the one place
# it belongs, so keep it there (that branch exists precisely to study it).
_SIDE_BRANCH = "ast_method" in (args.outdir_suffix or "")
GRAPHS = ["core", "ast", "cfg", "dfg", "pdg", "final"]
if _SIDE_BRANCH and "ast_method" in R:
    GRAPHS.insert(2, "ast_method")
GNAME = {"core": "Core", "ast": "Core+AST", "ast_method": "Core+AST-m", "cfg": "Core+CFG", "dfg": "Core+DFG",
         "pdg": "Core+PDG", "final": "Core+AST+CSTG"}
GSHORT = {"core": "Core", "ast": "+AST", "ast_method": "+AST-m", "cfg": "+CFG", "dfg": "+DFG",
          "pdg": "+PDG", "final": "Final"}
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
MNAME = {"RN": "Relational neighbour", "PPR": "Personalized PageRank",
         "LP": "Label propagation", "DW": "DeepWalk embedding", "KGE": "KG embedding (DistMult)"}
M7 = [("Precision", "Prec."), ("Recall", "Rec."), ("Macro_F1", "Macro-F1"),
      ("Buggy_F1", "Buggy-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"), ("ACC", "Acc.")]
GCOL = {"core": "#999999", "ast": "#56B4E9", "ast_method": "#CC79A7", "cfg": "#E69F00", "dfg": "#009E73",
        "pdg": "#0072B2", "final": "#D55E00"}
MCOL = {"RN": "#56B4E9", "PPR": "#D55E00", "LP": "#009E73", "DW": "#0072B2", "KGE": "#CC79A7"}
plt.rcParams.update({"savefig.dpi": 150, "font.size": 10, "font.family": "DejaVu Sans"})


# ── seven per-metric LaTeX tables (rows=methods, cols=graphs) ────────────────

def tables():
    blocks = []
    for mk, lbl in M7:
        best_col = {g: max(R[g][m]["metrics"][mk] for m in METHODS) for g in GRAPHS}
        rows = []
        for m in METHODS:
            cells = []
            for g in GRAPHS:
                v = R[g][m]["metrics"][mk]
                s = f"{v:.3f}"
                cells.append(r"\textbf{" + s + "}" if abs(v - best_col[g]) < 1e-9 else s)
            rows.append(f"{MNAME[m]} & " + " & ".join(cells) + r" \\")
        blocks.append("\n".join([
            r"\begin{table}[H]", r"\centering \small",
            (rf"\caption{{Online {lbl} of the five graph-inference methods across the "
             r"six knowledge graphs (Groovy; no X; online-tuned operating point). Best "
             r"method per graph in \textbf{bold}.}"),
            rf"\label{{tab:v4-final-{mk.lower()}}}",
            r"\begin{tabular}{l" + "r" * len(GRAPHS) + "}", r"\toprule",
            "Method & " + " & ".join(GSHORT[g] for g in GRAPHS) + r" \\", r"\midrule",
            "\n".join(rows), r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    (TAB / "tab_final_bymetric.tex").write_text("\n\n".join(blocks))
    with open(TAB / "final_experiments.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "graph"] + [mk for mk, _ in M7])
        for m in METHODS:
            for g in GRAPHS:
                mm = R[g][m]["metrics"]
                w.writerow([m, GNAME[g]] + [f"{mm[mk]:.4f}" for mk, _ in M7])
    print("wrote tab_final_bymetric.tex + final_experiments.csv")


# ── stream figures ───────────────────────────────────────────────────────────

def _panel(ax, series, mk, title):
    for key, (xs, ys, color, lw, ls, lab) in series.items():
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, label=lab)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(color="#EEE"); ax.set_axisbelow(True)
    ax.set_xlabel("Commit index", fontsize=8); ax.tick_params(labelsize=8)


def by_method_figs():
    """5 figures (one per method); each has 7 metric panels; trends = graphs."""
    for m in METHODS:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
        for ax, (mk, lbl) in zip(axes, M7):
            series = {}
            for g in GRAPHS:
                t = R[g][m]["traj"]
                lw = 2.6 if g == "final" else (1.5 if g != "core" else 1.8)
                ls = "-" if g != "core" else "--"
                series[g] = (t["idx"], t[mk], GCOL[g], lw, ls, GSHORT[g])
            _panel(ax, series, mk, lbl)
        h, l = axes[0].get_legend_handles_labels()
        axes[-1].axis("off"); axes[-1].legend(h, l, loc="center", ncol=2, fontsize=11,
                                              title="graph", frameon=False)
        fig.suptitle(f"Online-evaluation streams — {MNAME[m]} ({m}); trends = graphs",
                     fontsize=13, weight="bold", y=1.01)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(FIG / f"fig_final_bymethod_{m}.{e}", bbox_inches="tight")
        plt.close(fig); print(f"  wrote fig_final_bymethod_{m}")


def by_graph_figs():
    """6 figures (one per graph); each has 7 metric panels; trends = methods."""
    for g in GRAPHS:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
        for ax, (mk, lbl) in zip(axes, M7):
            series = {}
            for m in METHODS:
                t = R[g][m]["traj"]
                lw = 2.6 if m == "PPR" else 1.6
                series[m] = (t["idx"], t[mk], MCOL[m], lw, "-", m)
            _panel(ax, series, mk, lbl)
        h, l = axes[0].get_legend_handles_labels()
        axes[-1].axis("off"); axes[-1].legend(h, l, loc="center", ncol=2, fontsize=11,
                                              title="method", frameon=False)
        fig.suptitle(f"Online-evaluation streams — {GNAME[g]} graph; trends = methods",
                     fontsize=13, weight="bold", y=1.01)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(FIG / f"fig_final_bygraph_{g}.{e}", bbox_inches="tight")
        plt.close(fig); print(f"  wrote fig_final_bygraph_{g}")


def main():
    tables()
    print("by-method figures (35 subplots = 7 metrics x 5 methods):")
    by_method_figs()
    print("by-graph figures (42 subplots = 7 metrics x 6 graphs):")
    by_graph_figs()
    print("done.")


if __name__ == "__main__":
    main()
