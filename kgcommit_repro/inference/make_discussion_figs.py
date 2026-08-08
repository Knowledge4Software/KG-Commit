"""
Discussion figures D4 / D6 and Appendix I / J (cache-only, from scalability JSON).

D4 / Appendix I (cross-project difficulty), per project:
  - graph_change_vs_bugrate: cumulative AST-delta growth vs rolling bug-rate over time.
  - churn_vs_density: per-commit AST delta size (churn proxy) vs graph edge/node density.
  - vocab_density: structural token-vocabulary size per layer (bar).
D6 / Appendix J (growth / long-horizon sustainability), per project:
  - growth_multilayer: cumulative nodes of all layers vs commit index (log y).

Sources: scalability/growth_arrays.json, growth.json, subgraph_layer_stats.json.
Writes into discussions/D4_difficulty, discussions/D6_sustainability,
appendices/I_difficulty, appendices/J_sustainability.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
LAYER_COL = {"ast": "#1f77b4", "cfg": "#ff7f0e", "dfg": "#2ca02c",
             "pdg": "#d62728", "seq": "#9467bd", "cstg": "#8c564b"}


def _load(folder):
    ga = OUTP / folder / "scalability" / "growth_arrays.json"
    if not ga.exists():
        return None
    return json.load(open(ga))


def rolling(a, w=200):
    a = np.asarray(a, float)
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        lo = max(0, i - w + 1); out[i] = a[lo:i+1].mean()
    return out


def d4_difficulty(folder, disp, into):
    ga = _load(folder)
    if not ga:
        return False
    y = np.asarray(ga["y"], float); n = len(y); x = np.arange(n)
    ast = ga["layers"].get("ast", {})
    cum = np.asarray(ast.get("total", [0]*n), float)
    # graph-change vs bug-rate
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, cum, color="#1f77b4", lw=2, label="Cumulative AST nodes")
    ax1.set_xlabel("Commit index"); ax1.set_ylabel("Cumulative AST nodes", color="#1f77b4")
    ax2 = ax1.twinx(); ax2.plot(x, rolling(y), color="#d62728", lw=1.6, label="Bug rate")
    ax2.set_ylabel("Bug rate", color="#d62728")
    ax1.set_title(f"{disp}", weight="bold")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(into / f"graph_change_vs_bugrate__{folder}.{e}", bbox_inches="tight")
    plt.close(fig)
    # churn vs density (per-commit delta vs edges/nodes)
    per = np.diff(np.concatenate([[0], cum]))     # per-commit AST node additions
    edges = np.asarray(ast.get("cum_edges", cum), float); nodes = np.maximum(cum, 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, rolling(per), color="#2ca02c", lw=1.6, label="rolling per-commit AST churn")
    ax.set_xlabel("Commit index"); ax.set_ylabel("AST churn")
    axb = ax.twinx(); axb.plot(x, edges/nodes, color="#8c564b", lw=1.2, label="Edge/node ratio")
    axb.set_ylabel("Edge/node ratio", color="#8c564b")
    ax.set_title(f"{disp}", weight="bold"); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(into / f"churn_vs_density__{folder}.{e}", bbox_inches="tight")
    plt.close(fig)
    # vocabulary density per layer
    st = OUTP / folder / "subgraph_layer_stats.json"
    if st.exists():
        S = json.load(open(st))
        labs, vals = [], []
        for g in ["ast", "cfg", "dfg", "pdg", "seq"]:
            gs = S.get(g, {}) if isinstance(S, dict) else {}
            nv = gs.get("n_token_types") or gs.get("token_types")
            if nv:
                labs.append(g.upper()); vals.append(nv)
        if labs:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.bar(labs, vals, color=[LAYER_COL[l.lower()] for l in labs])
            ax.set_ylabel("Vocabulary size")
            ax.set_title(f"{disp}", weight="bold"); fig.tight_layout()
            for e in ("pdf", "png"):
                fig.savefig(into / f"vocab_density__{folder}.{e}", bbox_inches="tight")
            plt.close(fig)
    return True


def d6_sustain(folder, disp, into):
    ga = _load(folder)
    if not ga:
        return False
    n = len(ga["y"]); x = np.arange(n)
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, col in LAYER_COL.items():
        L = ga["layers"].get(g)
        if L and "total" in L:
            ax.plot(x, np.maximum(np.asarray(L["total"], float), 0.1), color=col, lw=1.6, label=g.upper())
    ax.set_yscale("log"); ax.set_xlabel("Commit index"); ax.set_ylabel("Cumulative nodes")
    ax.set_title(f"{disp}", weight="bold")
    ax.legend(fontsize=8, ncol=3); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(into / f"growth_multilayer__{folder}.{e}", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    d4 = PM / "discussions" / "D4_difficulty"; d6 = PM / "discussions" / "D6_sustainability"
    ai = PM / "appendices" / "I_difficulty"; aj = PM / "appendices" / "J_sustainability"
    for d in (d4, d6, ai, aj):
        d.mkdir(parents=True, exist_ok=True)
    for disp, folder in PROJECTS:
        ok4 = d4_difficulty(folder, disp, ai)     # full per-project set in appendix I
        ok6 = d6_sustain(folder, disp, aj)        # full per-project set in appendix J
        print(f"  {disp}: D4/I={'ok' if ok4 else '--'} D6/J={'ok' if ok6 else '--'}")
    # pick 2 example projects for the main-text D4/D6 (kafka=mid, zookeeper=hard)
    for disp, folder in [("Kafka", "kafka"), ("Zookeeper", "zookeeper")]:
        d4_difficulty(folder, disp, d4); d6_sustain(folder, disp, d6)
    print("wrote D4/D6 (2 example) + Appendix I/J (all projects)")


if __name__ == "__main__":
    main()
