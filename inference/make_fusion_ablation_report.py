"""
Paper artefacts for the final-method fusion ablation (run_fusion_ablation.py):
  outputs/tables/v4/tab_fusion_singles.tex     10 methods (singles) x 7 metrics
  outputs/tables/v4/tab_fusion_recommend.tex   key fusion formulas x 7 metrics
  outputs/figures/v4/fig_fusion_pareto.{pdf,png}  #components vs Buggy-F1 (Pareto)
Prints the recommended final Fusion formula.

Run:  python inference/make_fusion_ablation_report.py
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "outputs"
TAB = OUT / "tables" / "v4"; FIG = OUT / "figures" / "v4"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
M7 = [("Precision", "Prec."), ("Recall", "Rec."), ("Macro_F1", "Macro-F1"),
      ("Buggy_F1", "Buggy-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"), ("ACC", "Acc.")]
R = pickle.load(open(OUT / "fusion_ablation_results.pkl", "rb"))


def find(fam, methods):
    s = set(methods)
    for v in R[fam].values():
        if set(v["methods"]) == s:
            return v
    return None


def _table(caption, label, rows, colspec="lccrrrrrrr"):
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\begin{tabular}{" + colspec + "}", r"\toprule",
         r"Fusion & graph & \#ch & " + " & ".join(l for _, l in M7) + r" \\",
         r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def singles_table():
    sing = sorted([v for v in R["stack"].values() if len(v["methods"]) == 1],
                  key=lambda v: -v["Buggy_F1"])
    rows = []
    for v in sing:
        g = r"\checkmark" if v["graph_only"] else "--"
        cells = " & ".join(f"{v[mk]:.3f}" for mk, _ in M7)
        rows.append(f"{v['methods'][0]} & {g} & 1 & {cells} " + r"\\")
    return _table("Single inference methods on the final graph (Core+AST+CSTG, no X),"
                  " seven headline metrics, ranked by Buggy-F1. The non-graph JIT"
                  " metrics (M) are among the weakest; CSTG (G) is the strongest single"
                  " channel.", "tab:v4-fusion-singles", rows)


def recommend_table():
    picks = [
        ("feat", ["M", "T", "R", "P"], "M+T+R+P (previous Fusion)"),
        ("feat", ["M", "R", "T", "G"], "M+R+T+G (best overall)"),
        ("stack", ["R", "T", "P", "G", "RN", "LP", "KGE"], "R+T+P+G+RN+LP+KGE (best graph stack)"),
        ("feat", ["R", "T", "P", "G"], "R+T+P+G"),
        ("feat", ["T", "G"], "T+G (structure+semantics)"),
        ("feat", ["R", "T", "G"], r"\textbf{R+T+G (recommended)}"),
    ]
    rows = []
    for fam, methods, name in picks:
        v = find(fam, methods)
        if not v:
            continue
        g = r"\checkmark" if v["graph_only"] else "--"
        cells = " & ".join(f"{v[mk]:.3f}" for mk, _ in M7)
        rows.append(f"{name} & {g} & {len(v['methods'])} & {cells} " + r"\\")
        if name.startswith("R+T+P+G+"):
            rows.append(r"\midrule")
    return _table("Candidate final Fusion formulas on the seven headline metrics "
                  "(feature-fusion unless noted; online-tuned operating point). The "
                  "recommended purely graph-native \\textbf{R+T+G} (relational priors, "
                  "AST structural tokens, CSTG semantics) improves on the previous "
                  "metrics-inclusive M+T+R+P on the minority-class metrics "
                  "(Recall, Buggy-F1, G-Mean, AUC, Macro-F1) while dropping the non-graph"
                  " JIT metrics and using fewer channels.",
                  "tab:v4-fusion-recommend", rows)


def pareto_fig():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for fam, mark in (("stack", "o"), ("feat", "s")):
        for v in R[fam].values():
            n = len(v["methods"]); bf = v["Buggy_F1"]
            c = "#0072B2" if v["graph_only"] else "#D55E00"
            ax.scatter(n, bf, s=18, marker=mark, color=c, alpha=0.35,
                       edgecolor="none", zorder=1)
    # annotate key formulas
    ann = [("feat", ["M", "T", "R", "P"], "M+T+R+P (old)", (10, -4)),
           ("feat", ["R", "T", "G"], "R+T+G (recommended)", (-58, 16)),
           ("feat", ["T", "G"], "T+G", (-6, 12)),
           ("stack", ["G"], "G alone (CSTG)", (10, -2))]
    for fam, methods, name, off in ann:
        v = find(fam, methods)
        if not v:
            continue
        n, bf = len(v["methods"]), v["Buggy_F1"]
        star = "recommended" in name
        ax.scatter(n, bf, s=180 if star else 90, marker="*" if star else "D",
                   color="#009E73" if star else "#222", zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(name, (n, bf), xytext=off, textcoords="offset points",
                    fontsize=9.5, weight="bold" if star else "normal")
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", label="graph-only", markersize=9),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00", label="includes M (non-graph)", markersize=9),
           Line2D([0], [0], marker="*", color="w", markerfacecolor="#009E73", label="recommended (R+T+G)", markersize=15)]
    ax.legend(handles=leg, fontsize=9.5, frameon=False, loc="lower right")
    ax.set_xlabel("number of components in the fusion"); ax.set_ylabel("Buggy-F1 (online)")
    ax.set_title("Fusion ablation: performance vs parsimony\n"
                 "a compact graph-only fusion (R+T+G) matches the best while dropping "
                 "non-graph M", fontsize=12, weight="bold")
    ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig_fusion_pareto.{e}", bbox_inches="tight")
    plt.close(fig)


def bysize_tables(metric="Buggy_F1", topn=8):
    """Ten tables: for each fusion size k=1..10 the best `topn` score-stacking
    fusions (ranked by `metric`), all seven metrics as columns."""
    blocks = []
    for k in range(1, 11):
        subs = sorted([v for v in R["stack"].values() if len(v["methods"]) == k],
                      key=lambda v: -v[metric])[:topn]
        rows = []
        for v in subs:
            nm = "+".join(v["methods"])
            g = r"\checkmark" if v["graph_only"] else "--"
            cells = " & ".join(f"{v[mk]:.3f}" for mk, _ in M7)
            rows.append(f"{nm} & {g} & {cells} " + r"\\")
        body = "\n".join(rows)
        s = "s" if k > 1 else ""
        blocks.append("\n".join([
            r"\begin{table}[H]", r"\centering \scriptsize",
            (rf"\caption{{Best {min(topn,len(subs))} fusion(s) of $k={k}$ method{s} "
             r"by Buggy-F1 (score-stacking over the ten inference methods; final "
             r"Core+AST+CSTG graph, no X). ``graph'' = purely graph-native (no "
             r"non-graph JIT metrics $M$).}"),
            rf"\label{{tab:v4-fusion-k{k}}}",
            r"\begin{tabular}{lcrrrrrrr}", r"\toprule",
            r"Fusion & graph & Prec. & Rec. & Macro-F1 & Buggy-F1 & G-Mean & AUC & Acc. \\",
            r"\midrule", body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    return "\n\n".join(blocks)


def bysize_csv(metric="Buggy_F1", topn=8):
    import csv
    with open(TAB / "fusion_bysize_top8.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k", "fusion", "graph_only"] + [mk for mk, _ in M7])
        for k in range(1, 11):
            subs = sorted([v for v in R["stack"].values() if len(v["methods"]) == k],
                          key=lambda v: -v[metric])[:topn]
            for v in subs:
                w.writerow([k, "+".join(v["methods"]), int(v["graph_only"])] +
                           [f"{v[mk]:.4f}" for mk, _ in M7])


def ablation_viz():
    stack = list(R["stack"].values())
    ks = np.array([len(v["methods"]) for v in stack])
    # (1) Buggy-F1 distribution by k + best-per-k line
    fig, ax = plt.subplots(figsize=(9, 4.6))
    data = [[v["Buggy_F1"] for v in stack if len(v["methods"]) == k] for k in range(1, 11)]
    bp = ax.boxplot(data, positions=range(1, 11), widths=0.6, patch_artist=True,
                    showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor="#CDE3F0", edgecolor="#4a4a4a")
    for med in bp["medians"]:
        med.set(color="#4a4a4a")
    best = [max(d) for d in data]
    ax.plot(range(1, 11), best, "-o", color="#D55E00", lw=2, zorder=5,
            label="best fusion of size $k$")
    ax.set_xlabel("number of methods in the fusion ($k$)")
    ax.set_ylabel("Buggy-F1 (online)")
    ax.set_title("Fusion performance vs size: best saturates by $k\\approx3$–4",
                 fontsize=12, weight="bold")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig_ablation_bysize.{e}", bbox_inches="tight")
    plt.close(fig)

    # (2) method frequency among the top-5% fusions by Buggy-F1
    top = sorted(stack, key=lambda v: -v["Buggy_F1"])[:max(1, len(stack) // 20)]
    from collections import Counter
    cnt = Counter(m for v in top for m in v["methods"])
    order = ["M", "R", "T", "P", "E", "G", "RN", "LP", "DW", "KGE"]
    vals = [100 * cnt.get(m, 0) / len(top) for m in order]
    # green = near-universal essential trio (>=90%); orange = non-graph M; blue = rest
    col = ["#009E73" if v >= 90 else ("#D55E00" if m == "M" else "#0072B2")
           for m, v in zip(order, vals)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    b = ax.bar(order, vals, color=col, edgecolor="white")
    for bar, val in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(90, ls="--", lw=1, color="#009E73")
    ax.set_ylabel(f"appearance in the best {len(top)} fusions (%)")
    ax.set_title("Which methods make the best fusions? (top-5% by Buggy-F1)\n"
                 "R, T and G are near-universal (green) --- the essential trio; "
                 "others (incl. non-graph M) are optional",
                 fontsize=11.5, weight="bold")
    ax.set_ylim(0, 105); ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig_ablation_method_freq.{e}", bbox_inches="tight")
    plt.close(fig)

    # (3) best achievable per metric vs k (7 small multiples)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2)); axes = axes.ravel()
    for ax, (mk, lbl) in zip(axes, M7):
        bestk = [max(v[mk] for v in stack if len(v["methods"]) == k) for k in range(1, 11)]
        ax.plot(range(1, 11), bestk, "-o", color="#009E73", lw=2)
        ax.set_title(lbl, fontsize=11, weight="bold"); ax.set_xlabel("$k$", fontsize=9)
        ax.grid(color="#EEE"); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    axes[-1].axis("off")
    fig.suptitle("Best achievable value of each metric vs fusion size $k$",
                 fontsize=13, weight="bold", y=1.01)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig_ablation_bestperk.{e}", bbox_inches="tight")
    plt.close(fig)


def main():
    (TAB / "tab_fusion_singles.tex").write_text(singles_table())
    (TAB / "tab_fusion_recommend.tex").write_text(recommend_table())
    (TAB / "tab_fusion_bysize.tex").write_text(bysize_tables())
    bysize_csv(); ablation_viz()
    pareto_fig()
    rec = find("feat", ["R", "T", "G"]); old = find("feat", ["M", "T", "R", "P"])
    print("RECOMMENDED FINAL FUSION:  R + T + G  (feature-fusion, graph-only)")
    print(f"  vs previous M+T+R+P (has non-graph M):")
    for mk, lbl in M7:
        d = rec[mk] - old[mk]
        print(f"    {lbl:<9} R+T+G={rec[mk]:.3f}  M+T+R+P={old[mk]:.3f}  ({'+' if d>=0 else ''}{d:.3f})")
    print("\nwrote tab_fusion_singles.tex, tab_fusion_recommend.tex, fig_fusion_pareto.*")


if __name__ == "__main__":
    main()
