"""
RQ3 compact (non-stream) figures, cache-only:
  1. marginal_gain_radar__<p>: radar/spider over 7 metrics for Core, Core+AST,
     Core+AST+CSTG (F), F+G -- shows each layer's marginal gain.
  2. ast_beats_subgraphs: grouped-bar (Macro-F1) of Core+AST vs +CFG/+DFG/+PDG per
     project -- compact evidence AST dominates the alternative subgraphs.
  3. why_ast_wins: scatter of structural token-vocabulary size vs Macro-F1 gain over
     Core, per subgraph -- AST has the richest vocabulary and the largest gain.
Sources: final_experiments_results.pkl (per-graph Fusion metrics),
         subgraph_layer_stats.json (token vocab per layer), final_fusion (F+G).
"""
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ3_subgraphs"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
M7L = ["Prec", "Rec", "MacroF1", "BuggyF1", "GMean", "AUC", "Acc"]


def _fus(fe, g):
    n = fe.get(g, {}).get("Fusion")
    return n["metrics"] if isinstance(n, dict) else None


def radar(folder, disp):
    fe = OUTP / folder / "final_experiments_results.pkl"
    ff = OUTP / folder / "final_fusion_results.pkl"
    if not (fe.exists() and ff.exists()):
        return False
    E = pickle.load(open(fe, "rb")); F = pickle.load(open(ff, "rb"))
    series = {}
    for lab, g in [("Core", "core"), ("Core+AST", "ast"), ("Core+AST+CSTG (F)", "final")]:
        m = _fus(E, g)
        if m:
            series[lab] = [m.get(k, np.nan) for k in M7]
    series["F+G"] = [F["part2"]["F+G"]["metrics"].get(k, np.nan) for k in M7]
    ang = np.linspace(0, 2*np.pi, len(M7), endpoint=False); ang = np.concatenate([ang, [ang[0]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    cols = {"Core": "#9e9e9e", "Core+AST": "#1f77b4", "Core+AST+CSTG (F)": "#2ca02c", "F+G": "#D55E00"}
    for lab, vals in series.items():
        v = np.array(vals + [vals[0]])
        ax.plot(ang, v, lw=2 if lab == "F+G" else 1.4, label=lab, color=cols.get(lab))
        ax.fill(ang, v, alpha=0.05, color=cols.get(lab))
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(M7L, fontsize=8)
    ax.set_title(f"{disp}", weight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"marginal_gain_radar__{folder}.{e}", bbox_inches="tight")
    plt.close(fig); return True


def ast_beats():
    names, ast, cfg, dfg, pdg = [], [], [], [], []
    for disp, folder in PROJECTS:
        fe = OUTP / folder / "final_experiments_results.pkl"
        if not fe.exists():
            continue
        E = pickle.load(open(fe, "rb"))
        def mf1(g):
            m = _fus(E, g); return m.get("Macro_F1", np.nan) if m else np.nan
        names.append(disp); ast.append(mf1("ast")); cfg.append(mf1("cfg"))
        dfg.append(mf1("dfg")); pdg.append(mf1("pdg"))
    if not names:
        return
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(7, len(names)*1.15), 4))
    for i, (lab, d, c) in enumerate([("+AST", ast, "#1f77b4"), ("+CFG", cfg, "#ff7f0e"),
                                     ("+DFG", dfg, "#2ca02c"), ("+PDG", pdg, "#d62728")]):
        ax.bar(x + (i-1.5)*w, d, w, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1"); ax.set_title("AST vs. subgraphs", weight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"ast_beats_subgraphs.{e}", bbox_inches="tight")
    plt.close(fig)


def why_ast():
    vocab = {"ast": [], "cfg": [], "dfg": [], "pdg": []}; gain = {"ast": [], "cfg": [], "dfg": [], "pdg": []}
    for disp, folder in PROJECTS:
        fe = OUTP / folder / "final_experiments_results.pkl"
        st = OUTP / folder / "subgraph_layer_stats.json"
        if not (fe.exists() and st.exists()):
            continue
        E = pickle.load(open(fe, "rb")); S = json.load(open(st))
        core = _fus(E, "core"); core_mf1 = core.get("Macro_F1", np.nan) if core else np.nan
        for g in ["ast", "cfg", "dfg", "pdg"]:
            m = _fus(E, g); nv = None
            # token vocab count from layer stats (key names vary; try common ones)
            gs = S.get(g, {}) if isinstance(S, dict) else {}
            nv = gs.get("n_token_types") or gs.get("token_types") or gs.get("n_node_types")
            if m and nv:
                vocab[g].append(nv); gain[g].append(m.get("Macro_F1", np.nan) - core_mf1)
    fig, ax = plt.subplots(figsize=(5, 4))
    cols = {"ast": "#1f77b4", "cfg": "#ff7f0e", "dfg": "#2ca02c", "pdg": "#d62728"}
    for g in ["ast", "cfg", "dfg", "pdg"]:
        if vocab[g]:
            ax.scatter(vocab[g], gain[g], s=40, color=cols[g], label=g.upper(), alpha=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Vocabulary size"); ax.set_ylabel("Macro-F1 gain")
    ax.set_title("Vocabulary vs. gain", weight="bold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"why_ast_wins.{e}", bbox_inches="tight")
    plt.close(fig)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    for disp, folder in PROJECTS:
        print(f"  radar {disp}: {'ok' if radar(folder, disp) else '--'}")
    ast_beats(); why_ast()
    print("wrote radars + ast_beats_subgraphs + why_ast_wins")


if __name__ == "__main__":
    main()
