"""
D5 (feature/inference isolation and significance) figures + table, cache-only from
scalability/significance.json.

  - richness_significance: per project, the AUC of Core+AST and Core+AST+CSTG(final)
    vs Core, with DeLong significance (Holm) markers -- shows the richer
    representations significantly beat Core.
  - tab_significance_summary: cross-project table of the AST>Core and final>Core AUC
    gains and their Holm-corrected p-values (fusion channel).
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
PM = ROOT / "Paper" / "paper_material" / "discussions" / "D5_significance"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
def load(folder):
    f = OUTP / folder / "scalability" / "significance.json"
    return json.load(open(f)) if f.exists() else None


def _pick(rv, method, which):
    """which in {'AST','final'}; return (auc_a, auc_b, p_holm) for method>Core."""
    key = f"{method}: {which}>Core (ROC)"
    d = rv.get(key)
    if not d:
        return None
    return d.get("auc_a"), d.get("auc_b"), d.get("p_holm")


def figure():
    PM.mkdir(parents=True, exist_ok=True)
    names, ast_gain, fin_gain, ast_sig, fin_sig = [], [], [], [], []
    for disp, folder in PROJECTS:
        S = load(folder)
        if not S or "richness_vs_core" not in S:
            continue
        rv = S["richness_vs_core"]
        # aggregate over the 5 methods: mean AUC gain of AST>Core, final>Core
        ag, fg, asig, fsig = [], [], [], []
        for m in ["RN", "PPR", "LP", "DW", "KGE"]:
            a = _pick(rv, m, "AST"); f = _pick(rv, m, "final")
            if a:
                ag.append(a[0] - a[1]); asig.append(a[2] is not None and a[2] < 0.05)
            if f:
                fg.append(f[0] - f[1]); fsig.append(f[2] is not None and f[2] < 0.05)
        if not ag:
            continue
        names.append(disp)
        ast_gain.append(np.mean(ag)); fin_gain.append(np.mean(fg))
        ast_sig.append(np.mean(asig)); fin_sig.append(np.mean(fsig))
    if not names:
        return
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(names)*1.1), 4.2))
    b1 = ax.bar(x - w/2, ast_gain, w, label="Core+AST $-$ Core", color="#1f77b4")
    b2 = ax.bar(x + w/2, fin_gain, w, label="Core+AST+CSTG $-$ Core", color="#d62728")
    # star significant (majority of methods Holm-sig)
    for xi, g, s in zip(x - w/2, ast_gain, ast_sig):
        if s >= 0.5:
            ax.text(xi, g + 0.002, "*", ha="center", fontsize=11)
    for xi, g, s in zip(x + w/2, fin_gain, fin_sig):
        if s >= 0.5:
            ax.text(xi, g + 0.002, "*", ha="center", fontsize=11)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC gain")
    ax.set_title("Richness vs. Core", weight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"richness_significance.{e}", bbox_inches="tight")
    plt.close(fig)

    # summary table (PPR channel as representative, AST>Core & final>Core)
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{D5: statistical significance of representation richness "
         r"(DeLong test on ROC-AUC, Holm-corrected). Mean AUC gain over Core across "
         r"the five methods; $\dagger$ = Holm-significant for the majority of methods.}",
         r"\label{tab:d5_significance}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"Project & Core+AST $-$ Core & Core+AST+CSTG $-$ Core \\ \midrule"]
    for disp, g1, g2, s1, s2 in zip(names, ast_gain, fin_gain, ast_sig, fin_sig):
        d1 = r"$\dagger$" if s1 >= 0.5 else ""
        d2 = r"$\dagger$" if s2 >= 0.5 else ""
        L.append(f"{disp} & {g1:+.3f}{d1} & {g2:+.3f}{d2} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / "tab_significance_summary.tex").write_text("\n".join(L), encoding="utf-8")


def main():
    figure()
    print("wrote D5 richness_significance figure + tab_significance_summary")


if __name__ == "__main__":
    main()
