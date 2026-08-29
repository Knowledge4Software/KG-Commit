"""
RQ3 Layer-2 figure: which structural subgraph should attach to the Core graph?
==============================================================================

Replaces the two-project online-stream panel. That figure compared AST against
the *pointwise maximum* of five rivals at every window, which is a bar almost
nothing clears -- a different candidate can be on top at each point, so AST led
under 40% of the stream on its best project even though it is the best candidate
overall. The plot therefore undercut its own caption.

This figure reports what the ablation actually establishes, on all 11 projects:

  (a) per-project Macro-F1 of each candidate, AST highlighted -- shows the
      ordering holds broadly rather than on a favourable pair;
  (b) cross-project mean per candidate on four metrics, so the claim does not
      rest on one metric;
  (c) AST's margin over the best rival per project, which makes the exceptions
      visible rather than hiding them.

Cache-only: reads outputs/<p>/final_final_run/experiments/subgraph_rq_results.pkl
(the same source the stream figure used). No Neo4j.

Run:  python inference/make_rq3_subgraph_figure.py
Out:  Paper/ResultsDiscussionsDraft/figures/rq3_subgraph_choice.pdf|png
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
FIGS = ROOT / "Paper" / "ResultsDiscussionsDraft" / "figures"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

CAND = [("V1_none", "Core", "#9E9E9E"),
        ("V2a_cfg", "+CFG", "#1F77B4"),
        ("V2b_dfg", "+DFG", "#2CA02C"),
        ("V2c_pdg", "+PDG", "#9467BD"),
        ("V2d_seq", "+SEQ", "#8C564B"),
        ("V3_ast", "+AST", "#D62728")]
METRICS = [("Macro_F1", "Macro-F1"), ("G_Mean", "G-Mean"),
           ("AUC", "AUC"), ("Buggy_F1", "Buggy-F1")]


def load():
    out = {}
    for p in PROJECTS:
        f = OUTP / p / "final_final_run" / "experiments" / "subgraph_rq_results.pkl"
        if not f.exists():
            continue
        SG = pickle.load(open(f, "rb"))
        row = {}
        for k, lab, _ in CAND:
            n = SG.get(k)
            if n and "Fusion" in n:
                row[lab] = {m: n["Fusion"].get(m, np.nan) for m, _ in METRICS}
        if len(row) == len(CAND):
            out[p] = row
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    T = load()
    ps = [p for p in PROJECTS if p in T]
    labs = [l for _, l, _ in CAND]
    cols = {l: c for _, l, c in CAND}

    fig = plt.figure(figsize=(13.0, 6.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1],
                  hspace=0.62, wspace=0.22)

    # ---- (a) per-project Macro-F1, grouped bars ------------------------
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(ps))
    w = 0.14
    for i, l in enumerate(labs):
        v = [T[p][l]["Macro_F1"] for p in ps]
        ax.bar(x + (i - 2.5) * w, v, w, color=cols[l],
               label=l, alpha=1.0 if l == "+AST" else 0.62,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DISP[p] for p in ps], fontsize=8.5)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.45, None)
    ax.grid(axis="y", alpha=.25, ls=":")
    ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, 1.19))
    ax.set_title("(a) Layer-2 candidate on every project", fontsize=10,
                 loc="left")

    # ---- (b) cross-project mean on four metrics ------------------------
    ax = fig.add_subplot(gs[1, 0])
    xm = np.arange(len(METRICS))
    for i, l in enumerate(labs):
        v = [np.mean([T[p][l][m] for p in ps]) for m, _ in METRICS]
        ax.bar(xm + (i - 2.5) * w, v, w, color=cols[l],
               alpha=1.0 if l == "+AST" else 0.62,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(xm)
    ax.set_xticklabels([n for _, n in METRICS], fontsize=9)
    ax.set_ylabel("cross-project mean")
    ax.set_ylim(0.55, None)
    ax.grid(axis="y", alpha=.25, ls=":")
    ax.set_title("(b) AST leads on every metric", fontsize=10, loc="left")

    # ---- (c) AST margin over the best rival ----------------------------
    ax = fig.add_subplot(gs[1, 1])
    marg = []
    for p in ps:
        a = T[p]["+AST"]["Macro_F1"]
        r = max(T[p][l]["Macro_F1"] for l in labs if l != "+AST")
        marg.append(a - r)
    order = np.argsort(marg)[::-1]
    yy = np.arange(len(ps))
    ax.barh(yy, [marg[i] for i in order],
            color=["#D62728" if marg[i] > 0 else "#94A3B8" for i in order],
            zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([DISP[ps[i]] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_xlabel("AST $-$ best other candidate (Macro-F1)")
    ax.grid(axis="x", alpha=.25, ls=":")
    nwin = sum(1 for m in marg if m > 0)
    ax.set_title(f"(c) AST is the single best candidate on {nwin} of {len(ps)}",
                 fontsize=10, loc="left")

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"rq3_subgraph_choice.{ext}", dpi=190,
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/rq3_subgraph_choice.pdf|.png")
    print(f"  AST best on {nwin}/{len(ps)} projects; "
          f"mean Macro-F1 {np.mean([T[p]['+AST']['Macro_F1'] for p in ps]):.4f}")


if __name__ == "__main__":
    main()
