"""
Publication figures for the v4 "which subgraph is best?" ablation.

Reads the two caches produced by run_subgraph_rq.py and collect_subgraph_stats.py
(outputs/subgraph_rq_results.pkl, outputs/subgraph_layer_stats.json) and renders,
to outputs/figures/v4/ (both .pdf for LaTeX and .png for the notebook):

  fig_fusion_metrics   grouped bars: Fusion PR-AUC / ROC-AUC / F1-online per variant
  fig_tonly_signal     isolated structural signal (T-only PR-AUC) per variant
  fig_vocab_scatter    token-vocabulary size vs T-only PR-AUC (the mechanism)
  fig_trajectory       rolling PR-AUC over the commit stream per variant
  fig_layer_sizes      per-layer graph scale (nodes / delta-edges)

Design: form chosen per data job (magnitude->bars, relationship->scatter,
change-over-time->lines); Okabe-Ito colourblind-safe categorical palette in FIXED
order; AST (incumbent/winner) carries the bold accent; thin marks, direct value
labels, recessive grid, legend for >=2 series. No dual axes, no rainbow.

Run:  python inference/make_subgraph_figures.py
"""
import json, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import OUT  # per-project outputs/<project>/
FIG = OUT / "figures" / "v4"; FIG.mkdir(parents=True, exist_ok=True)

# fixed variant order + Okabe-Ito CVD-safe colours (AST = bold vermillion accent)
ORDER  = ["V1_none", "V2a_cfg", "V2b_dfg", "V2c_pdg", "V2d_seq", "V2e_ast_method", "V3_ast"]
SHORT  = {"V1_none": "Core\n(no subgraph)", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
          "V2c_pdg": "PDG/CPG", "V2d_seq": "Token-seq", "V2e_ast_method": "AST-m", "V3_ast": "AST"}
COLOR  = {"V1_none": "#999999", "V2a_cfg": "#56B4E9", "V2b_dfg": "#009E73",
          "V2c_pdg": "#0072B2", "V2d_seq": "#E69F00", "V2e_ast_method": "#CC79A7", "V3_ast": "#D55E00"}
STATKEY = {"V1_none": "_core", "V2a_cfg": "cfg", "V2b_dfg": "dfg",
           "V2c_pdg": "pdg", "V2d_seq": "seq", "V2e_ast_method": "ast_method", "V3_ast": "ast"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


def _load():
    res = pickle.load(open(OUT / "subgraph_rq_results.pkl", "rb"))
    stats = json.load(open(OUT / "subgraph_layer_stats.json"))
    return res, stats


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _barlabels(ax, bars, fmt="{:.3f}", dy=0.004, size=9):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom", size=size)


def fig_fusion_metrics(res):
    metrics = [("PR_AUC", "PR-AUC"), ("ROC_AUC", "ROC-AUC"), ("F1_online", "F1-online")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (mk, title) in zip(axes, metrics):
        vals = [res[v]["Fusion"][mk] for v in ORDER]
        cols = [COLOR[v] for v in ORDER]
        bars = ax.bar(range(len(ORDER)), vals, color=cols, width=0.72,
                      edgecolor="white", linewidth=1.2)
        # emphasise AST
        bars[-1].set_edgecolor("#222222"); bars[-1].set_linewidth(1.6)
        _barlabels(ax, bars)
        compact = {"V1_none": "Core", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
                   "V2c_pdg": "PDG", "V2d_seq": "Seq", "V2e_ast_method": "AST-m", "V3_ast": "AST"}
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([compact[v] for v in ORDER], fontsize=9.5, rotation=0)
        lo = min(vals); ax.set_ylim(max(0, lo - 0.06), max(vals) + 0.03)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.margins(x=0.02)
    axes[0].set_ylabel("score (prequential, X & G off)")
    fig.suptitle("Fusion (M+T+R+P) online performance by structural subgraph",
                 fontsize=13, weight="bold", y=1.02)
    _save(fig, "fig_fusion_metrics")


def fig_tonly_signal(res):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    vals = [res[v]["T_only"]["PR_AUC"] for v in ORDER]
    cols = [COLOR[v] for v in ORDER]
    bars = ax.bar(range(len(ORDER)), vals, color=cols, width=0.7,
                  edgecolor="white", linewidth=1.2)
    bars[-1].set_edgecolor("#222222"); bars[-1].set_linewidth(1.6)
    _barlabels(ax, bars, dy=0.006)
    # random baseline (stream bug-rate) reference line
    br = res["V3_ast"]["stream_bug"]
    ax.axhline(br, ls="--", lw=1.3, color="#666666")
    ax.text(len(ORDER) - 0.4, br + 0.006, f"random (bug-rate {br:.2f})",
            ha="right", va="bottom", fontsize=8.5, color="#666666")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([COMPACT[v] for v in ORDER], fontsize=9)
    ax.set_ylabel("PR-AUC of structural tokens alone (T-only)")
    ax.set_title("Isolated structural signal: the subgraph's change-tokens alone",
                 fontsize=12, weight="bold")
    ax.set_ylim(0, max(vals) + 0.06)
    _save(fig, "fig_tonly_signal")


def fig_vocab_scatter(res, stats):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for v in ORDER:
        if v == "V1_none":
            continue
        x = res[v]["n_token_types"]; y = res[v]["T_only"]["PR_AUC"]
        ax.scatter(x, y, s=150, color=COLOR[v], edgecolor="#222222",
                   linewidth=1.3, zorder=3)
        ax.annotate(SHORT[v].replace("\n", " "), (x, y),
                    xytext=(7, 4), textcoords="offset points", fontsize=10)
    xs = [res[v]["n_token_types"] for v in ORDER if v != "V1_none"]
    ys = [res[v]["T_only"]["PR_AUC"] for v in ORDER if v != "V1_none"]
    # log-x trend (vocabulary richness -> signal)
    lx = np.log(xs); b, a = np.polyfit(lx, ys, 1)
    xx = np.linspace(min(xs), max(xs), 100)
    ax.plot(xx, a + b * np.log(xx), ls="--", lw=1.3, color="#999999", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("token-vocabulary size (distinct {edge}:{type}, log scale)")
    ax.set_ylabel("isolated structural signal (T-only PR-AUC)")
    ax.set_title("Why AST wins: richer change-token vocabulary → stronger signal",
                 fontsize=12, weight="bold")
    _save(fig, "fig_vocab_scatter")


def fig_trajectory(res):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for v in ORDER:
        t = res[v]["traj"]
        lw = 2.6 if v == "V3_ast" else (1.6 if v != "V1_none" else 1.8)
        ls = "-" if v != "V1_none" else "--"
        ax.plot(t["idx"], t["PR_AUC"], color=COLOR[v], lw=lw, ls=ls,
                label=SHORT[v].replace("\n", " "), zorder=3 if v == "V3_ast" else 2)
    ax.set_xlabel("commit index in chronological stream")
    ax.set_ylabel("rolling PR-AUC (window = 150 commits, stride = 25)")
    ax.set_title("Consistency over the stream: AST leads throughout",
                 fontsize=12, weight="bold")
    ax.legend(ncol=3, fontsize=9, frameon=False, loc="lower right")
    _save(fig, "fig_trajectory")


def fig_layer_sizes(stats):
    layers = ["ast_method", "ast", "cfg", "dfg", "pdg", "seq"]
    vmap = {"ast": "V3_ast", "ast_method": "V2e_ast_method", "cfg": "V2a_cfg", "dfg": "V2b_dfg",
            "pdg": "V2c_pdg", "seq": "V2d_seq"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [(axes[0], "nodes", "nodes per layer"),
                           (axes[1], "delta_total", "delta-edges per layer")]:
        vals = [stats[l][key] for l in layers]
        cols = [COLOR[vmap[l]] for l in layers]
        bars = ax.bar(range(len(layers)), vals, color=cols, width=0.7,
                      edgecolor="white", linewidth=1.2)
        bars[0].set_edgecolor("#222222"); bars[0].set_linewidth(1.6)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{val/1e3:.0f}k", ha="center", va="bottom", size=9)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([stats[l]["pretty"] for l in layers], fontsize=9.5)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    fig.suptitle("Structural-layer scale (Groovy, 8,059 labelled commits)",
                 fontsize=13, weight="bold", y=1.01)
    _save(fig, "fig_layer_sizes")


METHOD_ORDER = ["M", "R", "T", "P", "E", "Fusion"]
METHOD_LABEL = {"M": "JIT metrics", "R": "Priors (wvRN)", "T": "Struct TF-IDF",
                "P": "PPR", "E": "KG embed (SVD)", "Fusion": "Fusion"}
STRUCT = ["T", "P", "E", "Fusion"]
COMPACT = {"V1_none": "Core", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
           "V2c_pdg": "PDG", "V2d_seq": "Seq", "V2e_ast_method": "AST-m", "V3_ast": "AST"}


def fig_permethod_heatmap(res, metric="PR_AUC", mlabel="PR-AUC"):
    """method x variant heatmap; each structural row ringed at its best variant."""
    M = np.array([[res[v]["methods"][m][metric] for v in ORDER] for m in METHOD_ORDER])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=M.min(), vmax=M.max())
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([COMPACT[v] for v in ORDER])
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABEL[m] + (" *" if m in STRUCT else "")
                        for m in METHOD_ORDER])
    for r, m in enumerate(METHOD_ORDER):
        best_c = int(np.argmax(M[r])) if m in STRUCT else None
        for c in range(len(ORDER)):
            val = M[r, c]
            txt = f"{val:.3f}"
            weight = "bold" if best_c is not None and c == best_c else "normal"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if val > M.min() + 0.62 * (M.max() - M.min()) else "#222",
                    weight=weight)
            if best_c is not None and c == best_c:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                           edgecolor="#D55E00", lw=2.2))
    ax.set_title(f"Inference method x structural subgraph ({mlabel})\n"
                 f"* = subgraph-dependent; orange ring = best variant for that method",
                 fontsize=11.5, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=mlabel)
    ax.set_xticks(np.arange(-.5, len(ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
    _save(fig, "fig_permethod_heatmap")


def fig_permethod_bars(res):
    """grouped bars: the 4 structural methods across variants (PR-AUC)."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    nV = len(ORDER); width = 0.8 / len(STRUCT)
    hatch = {"T": "", "P": "//", "E": "..", "Fusion": ""}
    mcol = {"T": "#0072B2", "P": "#E69F00", "E": "#009E73", "Fusion": "#D55E00"}
    for k, m in enumerate(STRUCT):
        vals = [res[v]["methods"][m]["PR_AUC"] for v in ORDER]
        xs = np.arange(nV) + (k - (len(STRUCT) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.95, label=METHOD_LABEL[m],
               color=mcol[m], edgecolor="white", linewidth=0.6, hatch=hatch[m])
    ax.set_xticks(range(nV)); ax.set_xticklabels([COMPACT[v] for v in ORDER])
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0.15, 0.66)
    ax.axhline(res["V3_ast"]["stream_bug"], ls="--", lw=1, color="#888")
    ax.set_title("Each KG-native inference method across subgraphs (PR-AUC, X & G off)",
                 fontsize=12, weight="bold")
    ax.legend(ncol=4, fontsize=9.5, frameon=False, loc="upper left")
    _save(fig, "fig_permethod_bars")


def main():
    res, stats = _load()
    print(f"figures -> {FIG}")
    fig_fusion_metrics(res)
    fig_tonly_signal(res)
    fig_vocab_scatter(res, stats)
    fig_trajectory(res)
    fig_layer_sizes(stats)
    fig_permethod_heatmap(res)
    fig_permethod_bars(res)
    print("done.")


if __name__ == "__main__":
    main()
