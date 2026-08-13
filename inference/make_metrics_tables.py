"""
Headline-metric tables for v4: reports the project's seven evaluation metrics
(Precision, Recall, Macro-F1, Buggy-F1, G-Mean, AUC, Accuracy) for every method
and every subgraph, covering BOTH method families:
  * the original methods (M, R, T, P, E, Fusion)     -> subgraph_rq_results.pkl
  * the five graph-native methods (RN, PPR, LP, DW, KGE), on the subgraph-only and
    full graph scopes                                -> subgraph_kg_methods_results.pkl

Writes to outputs/tables/v4/:
  tab_metrics_fusion.tex      variants x 7 metrics for the deployed Fusion model
  tab_metrics_allmethods.tex  every method x 7 metrics, at the AST subgraph
  tab_metrics_kg_bf1.tex      the 5 graph-native methods x subgraph, Buggy-F1, both scopes
  metrics_all.csv             long-form dump of everything
Threshold-based metrics are at the leakage-free online-tuned operating point;
AUC is threshold-free (see online_jit.final_metrics).

Run:  python inference/make_metrics_tables.py
"""
import pickle, csv
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
TAB = OUT / "tables" / "v4"; TAB.mkdir(parents=True, exist_ok=True)

ORDER = ["V1_none", "V2a_cfg", "V2b_dfg", "V2c_pdg", "V2d_seq", "V3_ast"]
SHORT = {"V1_none": "Core", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
         "V2c_pdg": "PDG", "V2d_seq": "Seq", "V3_ast": "AST"}
M7 = [("Precision", "Prec."), ("Recall", "Rec."), ("Macro_F1", "Macro-F1"),
      ("Buggy_F1", "Buggy-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"), ("ACC", "Acc.")]

OLD = ["M", "R", "T", "P", "E", "Fusion"]
OLD_NAME = {"M": "JIT metrics (M)", "R": "Relational priors (R)",
            "T": "Structural TF-IDF (T)", "P": "Personalized PageRank (P)",
            "E": "KG embedding SVD (E)", "Fusion": "Fusion (M+T+R+P)"}
KG = ["RN", "PPR", "LP", "DW", "KGE"]
KG_NAME = {"RN": "Relational neighbour (RN)", "PPR": "PageRank RWR (PPR)",
           "LP": "Label propagation (LP)", "DW": "DeepWalk embed (DW)",
           "KGE": "KG embed DistMult (KGE)"}


def _tabular(colspec, header, rows, caption, label):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}",
         r"\label{" + label + "}", r"\begin{tabular}{" + colspec + "}", r"\toprule",
         header + r" \\", r"\midrule"]
    L += rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def _bold_best(vals, fmt="{:.3f}"):
    best = max(vals)
    return [(r"\textbf{" + fmt.format(v) + "}") if abs(v - best) < 1e-9 else fmt.format(v)
            for v in vals]


def fusion_table(res):
    rows = []
    cols = {mk: [res[v]["methods"]["Fusion"][mk] for v in ORDER] for mk, _ in M7}
    bestcell = {mk: _bold_best(cols[mk]) for mk, _ in M7}
    for i, v in enumerate(ORDER):
        rows.append(SHORT[v] + " & " + " & ".join(bestcell[mk][i] for mk, _ in M7) + r" \\")
        if v == "V1_none":
            rows.append(r"\midrule")
    header = "Subgraph & " + " & ".join(lbl for _, lbl in M7)
    return _tabular("l" + "r" * len(M7), header, rows,
                    "Deployed Fusion ($M{+}T{+}R{+}P$) model across structural "
                    "subgraphs on the seven headline metrics (Groovy; X and G off; "
                    "threshold-based metrics at the online-tuned operating point). "
                    "Best per column in \\textbf{bold}.", "tab:v4-metrics-fusion")


def allmethods_table(res, kg):
    """all methods x 7 metrics at the AST subgraph."""
    recs = [(OLD_NAME[m], res["V3_ast"]["methods"][m]) for m in OLD]
    if kg:
        for scope in ("subgraph", "full"):
            for m in KG:
                recs.append((f"{KG_NAME[m]} [{scope}]", kg["V3_ast"][scope][m]))
    cols = {mk: [r[mk] for _, r in recs] for mk, _ in M7}
    best = {mk: max(cols[mk]) for mk, _ in M7}
    rows = []
    for i, (nm, r) in enumerate(recs):
        cells = []
        for mk, _ in M7:
            s = f"{r[mk]:.3f}"
            cells.append(r"\textbf{" + s + "}" if abs(r[mk] - best[mk]) < 1e-9 else s)
        rows.append(nm + " & " + " & ".join(cells) + r" \\")
        if nm.startswith("Fusion") or (kg and nm.endswith("[subgraph]") and "KGE" in nm):
            rows.append(r"\midrule")
    header = "Method & " + " & ".join(lbl for _, lbl in M7)
    return _tabular("l" + "r" * len(M7), header, rows,
                    "All inference methods on the AST subgraph, seven headline "
                    "metrics. Original methods, then the five graph-native methods "
                    "on the subgraph-only and full graph scopes. Best per column in "
                    "\\textbf{bold}.", "tab:v4-metrics-allmethods")


def kg_bf1_table(kg):
    rows = []
    for scope in ("subgraph", "full"):
        rows.append(r"\multicolumn{%d}{l}{\emph{scope: %s}} \\" % (len(ORDER) + 1, scope))
        for m in KG:
            vals = [kg[v][scope][m]["Buggy_F1"] for v in ORDER]
            cells = _bold_best(vals)
            rows.append(KG_NAME[m] + " & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    header = "Method & " + " & ".join(SHORT[v] for v in ORDER)
    return _tabular("l" + "r" * len(ORDER), header, rows,
                    "The five graph-native methods across structural subgraphs "
                    "(Buggy-F1), on the subgraph-only and full graph scopes. Best "
                    "subgraph per row in \\textbf{bold}.", "tab:v4-metrics-kgbf1")


def main():
    res = pickle.load(open(OUT / "subgraph_rq_results.pkl", "rb"))
    kgp = OUT / "subgraph_kg_methods_results.pkl"
    kg = pickle.load(open(kgp, "rb")) if kgp.exists() else None

    (TAB / "tab_metrics_fusion.tex").write_text(fusion_table(res))
    (TAB / "tab_metrics_allmethods.tex").write_text(allmethods_table(res, kg))
    if kg:
        (TAB / "tab_metrics_kg_bf1.tex").write_text(kg_bf1_table(kg))

    # long-form CSV over everything
    with open(TAB / "metrics_all.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "scope", "variant"] + [mk for mk, _ in M7])
        for v in ORDER:
            for m in OLD:
                r = res[v]["methods"][m]
                w.writerow(["original", m, "-", SHORT[v]] + [f"{r[mk]:.4f}" for mk, _ in M7])
        if kg:
            for v in ORDER:
                for scope in ("subgraph", "full"):
                    for m in KG:
                        r = kg[v][scope][m]
                        w.writerow(["graph-native", m, scope, SHORT[v]] +
                                   [f"{r[mk]:.4f}" for mk, _ in M7])

    print("=== tab_metrics_fusion.tex ===\n" + fusion_table(res))
    if kg:
        print("\n=== tab_metrics_allmethods.tex ===\n" + allmethods_table(res, kg))
    print(f"\nwrote -> {TAB}/tab_metrics_*.tex, metrics_all.csv  (kg={'yes' if kg else 'PENDING'})")


if __name__ == "__main__":
    main()
