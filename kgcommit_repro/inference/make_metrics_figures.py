"""
Headline-metric figures for v4 (the seven metrics: Precision, Recall, Macro-F1,
Buggy-F1, G-Mean, AUC, Accuracy). Additive to make_subgraph_figures.py.

Writes to outputs/figures/v4/ (pdf + png):
  fig_metrics_fusion     subgraph x 7-metric heatmap for the deployed Fusion model
  fig_metrics_allmethods method x 7-metric heatmap at the AST subgraph
                         (original + graph-native methods, both scopes)
  fig_kg_methods_bf1     the five graph-native methods across subgraphs (Buggy-F1),
                         subgraph-only vs full scope

Run:  python inference/make_metrics_figures.py
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import OUT  # per-project outputs/<project>/
FIG = OUT / "figures" / "v4"; FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["V1_none", "V2a_cfg", "V2b_dfg", "V2c_pdg", "V2d_seq", "V3_ast"]
SHORT = {"V1_none": "Core", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
         "V2c_pdg": "PDG", "V2d_seq": "Seq", "V3_ast": "AST"}
COLOR = {"V1_none": "#999999", "V2a_cfg": "#56B4E9", "V2b_dfg": "#009E73",
         "V2c_pdg": "#0072B2", "V2d_seq": "#E69F00", "V3_ast": "#D55E00"}
M7 = [("Precision", "Prec."), ("Recall", "Rec."), ("Macro_F1", "Macro-F1"),
      ("Buggy_F1", "Buggy-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"), ("ACC", "Acc.")]
OLD = ["M", "R", "T", "P", "E", "Fusion"]
OLD_NAME = {"M": "JIT metrics", "R": "Priors (wvRN)", "T": "Struct TF-IDF",
            "P": "PageRank", "E": "KG embed SVD", "Fusion": "Fusion"}
KG = ["RN", "PPR", "LP", "DW", "KGE"]
KG_NAME = {"RN": "Rel. neighbour", "PPR": "PageRank RWR", "LP": "Label prop.",
           "DW": "DeepWalk", "KGE": "KGE DistMult"}
plt.rcParams.update({"savefig.dpi": 200, "font.size": 11, "font.family": "DejaVu Sans"})


def _save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{e}", bbox_inches="tight")
    plt.close(fig); print(f"  wrote {name}")


def _heatmap(ax, Mv, rlabels, clabels, title, ring_best_col=True):
    im = ax.imshow(Mv, cmap="YlGnBu", aspect="auto", vmin=Mv.min(), vmax=Mv.max())
    ax.set_xticks(range(len(clabels))); ax.set_xticklabels(clabels, fontsize=9)
    ax.set_yticks(range(len(rlabels))); ax.set_yticklabels(rlabels, fontsize=9)
    lo, hi = Mv.min(), Mv.max()
    for r in range(Mv.shape[0]):
        for c in range(Mv.shape[1]):
            v = Mv[r, c]
            best = ring_best_col and r == int(np.argmax(Mv[:, c]))
            ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    weight="bold" if best else "normal",
                    color="white" if v > lo + 0.6 * (hi - lo) else "#222")
            if best:
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                           edgecolor="#D55E00", lw=2))
    ax.set_xticks(np.arange(-.5, len(clabels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rlabels), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4); ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=12, weight="bold")
    return im


def fig_metrics_fusion(res):
    Mv = np.array([[res[v]["methods"]["Fusion"][mk] for mk, _ in M7] for v in ORDER])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    _heatmap(ax, Mv, [SHORT[v] for v in ORDER], [l for _, l in M7],
             "Deployed Fusion across subgraphs — seven headline metrics\n"
             "(orange ring = best subgraph per metric)")
    _save(fig, "fig_metrics_fusion")


def fig_metrics_allmethods(res, kg):
    recs = [(OLD_NAME[m], res["V3_ast"]["methods"][m]) for m in OLD]
    if kg:
        for scope in ("subgraph", "full"):
            for m in KG:
                recs.append((f"{KG_NAME[m]} [{scope[:3]}]", kg["V3_ast"][scope][m]))
    Mv = np.array([[r[mk] for mk, _ in M7] for _, r in recs])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(recs) + 1.5))
    _heatmap(ax, Mv, [n for n, _ in recs], [l for _, l in M7],
             "All inference methods on the AST subgraph — seven metrics\n"
             "(orange ring = best method per metric)")
    _save(fig, "fig_metrics_allmethods")


def fig_kg_methods_bf1(kg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    col = {"RN": "#56B4E9", "PPR": "#E69F00", "LP": "#009E73", "DW": "#0072B2", "KGE": "#CC79A7"}
    for ax, scope in zip(axes, ("subgraph", "full")):
        nV = len(ORDER)
        width = 0.8 / len(KG)
        for k, m in enumerate(KG):
            vals = [kg[v][scope][m]["Buggy_F1"] for v in ORDER]
            xs = np.arange(nV) + (k - (len(KG) - 1) / 2) * width
            ax.bar(xs, vals, width=width * 0.95, label=KG_NAME[m], color=col[m],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(nV)); ax.set_xticklabels([SHORT[v] for v in ORDER])
        ax.set_title(f"scope: {scope}", fontsize=12, weight="bold")
        ax.grid(axis="y", color="#eee"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Buggy-F1 (online-tuned)")
    axes[1].legend(ncol=1, fontsize=9, frameon=False, loc="upper right")
    fig.suptitle("Graph-native inference methods across subgraphs (Buggy-F1)",
                 fontsize=13, weight="bold", y=1.02)
    _save(fig, "fig_kg_methods_bf1")


def fig_metric_streams(res):
    """Online-evaluation trend of every headline metric across the stream: x =
    commit index, y = rolling metric, one line per subgraph (deployed Fusion)."""
    panels = [("Precision", "Precision"), ("Recall", "Recall"),
              ("Macro_F1", "Macro-F1"), ("Buggy_F1", "Buggy-F1"),
              ("G_Mean", "G-Mean"), ("AUC", "AUC"), ("ACC", "Accuracy")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
    for ax, (mk, title) in zip(axes, panels):
        for v in ORDER:
            t = res[v].get("traj7")
            if not t:
                continue
            lw = 2.4 if v == "V3_ast" else (1.3 if v != "V1_none" else 1.6)
            ls = "-" if v != "V1_none" else "--"
            z = 3 if v == "V3_ast" else 2
            ax.plot(t["idx"], t[mk], color=COLOR[v], lw=lw, ls=ls, label=SHORT[v], zorder=z)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(color="#ECECEC"); ax.set_axisbelow(True)
        ax.set_xlabel("Commit index", fontsize=9)
        ax.tick_params(labelsize=8)
    ax_leg = axes[-1]; ax_leg.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    ax_leg.legend(h, l, loc="center", ncol=2, fontsize=12, title="subgraph",
                  title_fontsize=12, frameon=False)
    fig.suptitle("Online evaluation trend per metric across subgraphs "
                 "(deployed Fusion; rolling window = 150 commits, stride = 25)",
                 fontsize=14, weight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_metric_streams")


def main():
    res = pickle.load(open(OUT / "subgraph_rq_results.pkl", "rb"))
    kgp = OUT / "subgraph_kg_methods_results.pkl"
    kg = pickle.load(open(kgp, "rb")) if kgp.exists() else None
    print(f"figures -> {FIG}  (kg={'yes' if kg else 'PENDING'})")
    fig_metrics_fusion(res)
    fig_metrics_allmethods(res, kg)
    fig_metric_streams(res)
    if kg:
        fig_kg_methods_bf1(kg)
    print("done.")


if __name__ == "__main__":
    main()
