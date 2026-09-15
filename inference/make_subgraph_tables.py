"""
Comprehensive tables for the v4 subgraph ablation.

Reads the eval + stats caches and writes, to outputs/tables/v4/:
  results.csv                full metric table (all variants x all metrics)
  tab_results.tex            LaTeX booktabs main results table (Fusion + T/P-only)
  tab_layers.tex             LaTeX booktabs structural-layer statistics table
Also prints both LaTeX tables to stdout.

Run:  python inference/make_subgraph_tables.py
"""
import json, pickle, csv
from pathlib import Path

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import OUT  # per-project outputs/<project>/
TAB = OUT / "tables" / "v4"; TAB.mkdir(parents=True, exist_ok=True)

# V2e_ast_method was DROPPED from the final methodology -- excluded here even when a
# project's subgraph_rq_results.pkl still carries it.
ORDER = ["V1_none", "V2a_cfg", "V2b_dfg", "V2c_pdg", "V2d_seq", "V3_ast"]
NAME  = {"V1_none": "Core (no subgraph)", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
         "V2c_pdg": "PDG/CPG", "V2d_seq": "Token-seq", "V2e_ast_method": "AST-m", "V3_ast": "AST"}
SHORT = {"V1_none": "Core", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
         "V2c_pdg": "PDG", "V2d_seq": "Seq", "V2e_ast_method": "AST-m", "V3_ast": "AST"}
SKEY  = {"V2a_cfg": "cfg", "V2b_dfg": "dfg", "V2c_pdg": "pdg",
         "V2d_seq": "seq", "V2e_ast_method": "ast_method", "V3_ast": "ast"}
METHOD_ORDER = ["M", "R", "T", "P", "E", "Fusion"]
METHOD_NAME  = {"M": "JIT metrics", "R": "Relational priors", "T": "Structural TF-IDF",
                "P": "Personalized PageRank", "E": "KG embedding (SVD)",
                "Fusion": "Fusion ($M{+}T{+}R{+}P$)"}
STRUCTURAL   = {"T", "P", "E", "Fusion"}


def permethod_table(res, metric="PR_AUC", mlabel="PR-AUC"):
    """methods x variants LaTeX table for one metric; best structural variant per
    row in bold; M/R rows collapse (subgraph-independent)."""
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Per-method online " + mlabel + r" under each structural subgraph"
         r" (Groovy; X and G off). Rows marked $^\star$ consume the subgraph's"
         r" change-tokens and therefore vary; JIT-metrics and relational-priors are"
         r" subgraph-independent references. Best structural variant per row in"
         r" \textbf{bold}.}",
         r"\label{tab:v4-permethod}",
         r"\begin{tabular}{l" + "r" * len(ORDER) + "}", r"\toprule",
         "Method & " + " & ".join(SHORT[v] for v in ORDER) + r" \\", r"\midrule"]
    for m in METHOD_ORDER:
        star = r"$^\star$" if m in STRUCTURAL else ""
        vals = [res[v]["methods"][m][metric] for v in ORDER]
        best = max(v for v in vals) if m in STRUCTURAL else None
        cells = []
        for val in vals:
            s = f"{val:.3f}"
            cells.append(f"\\textbf{{{s}}}" if best is not None and abs(val - best) < 1e-9 else s)
        L.append(f"{METHOD_NAME[m]}{star} & " + " & ".join(cells) + r" \\")
        if m == "R":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    res = pickle.load(open(OUT / "subgraph_rq_results.pkl", "rb"))
    stats = json.load(open(OUT / "subgraph_layer_stats.json"))

    # ---- CSV (everything) ----
    with open(TAB / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "token_types", "commits_with_tokens",
                    "Fusion_PR", "Fusion_ROC", "Fusion_F1online", "Fusion_F1@.5",
                    "Fusion_MCC", "Fusion_Brier", "Tonly_PR", "Ponly_PR"])
        for v in ORDER:
            r = res[v]; F = r["Fusion"]
            w.writerow([NAME[v], r["n_token_types"], r["n_commits_with_tokens"],
                        f"{F['PR_AUC']:.4f}", f"{F['ROC_AUC']:.4f}",
                        f"{F['F1_online']:.4f}", f"{F['F1']:.4f}",
                        f"{F['MCC']:.4f}", f"{F['Brier']:.4f}",
                        f"{r['T_only']['PR_AUC']:.4f}", f"{r['P_only']['PR_AUC']:.4f}"])

    # ---- main results LaTeX table ----
    best = {m: max(res[v]["Fusion"][m] for v in ORDER)
            for m in ("PR_AUC", "ROC_AUC", "F1_online", "F1", "MCC")}
    def cell(v, m, lo=False):
        val = res[v]["Fusion"][m]
        s = f"{val:.3f}"
        return f"\\textbf{{{s}}}" if abs(val - best[m]) < 1e-9 else s
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Online prequential performance of the KG under each structural"
        r" subgraph (Groovy, 8{,}059 labelled commits). All variants share the same"
        r" core (JIT metrics $M$, relational priors $R$, PPR $P$) and online"
        r" protocol; text ($X$) and the semantic-text subgraph ($G$) are disabled."
        r" \emph{Fusion} $=M{+}T{+}R{+}P$. Best per column in \textbf{bold}.}",
        r"\label{tab:v4-results}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Subgraph & \#tok & PR-AUC & ROC-AUC & F1$_{\text{on}}$ & MCC & "
        r"$T$-only & $P$-only \\",
        r"         &       &        &         &                  &     & PR & PR \\",
        r"\midrule",
    ]
    for v in ORDER:
        r = res[v]
        lines.append(
            f"{NAME[v]} & {r['n_token_types']} & {cell(v,'PR_AUC')} & "
            f"{cell(v,'ROC_AUC')} & {cell(v,'F1_online')} & {cell(v,'MCC')} & "
            f"{r['T_only']['PR_AUC']:.3f} & {r['P_only']['PR_AUC']:.3f} \\\\")
        if v == "V1_none" or v == "V2d_seq":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tab_results = "\n".join(lines)
    (TAB / "tab_results.tex").write_text(tab_results)

    # ---- layer statistics LaTeX table ----
    L2 = [r"\begin{table}[t]", r"\centering",
          r"\caption{Scale of each structural layer materialised in the knowledge"
          r" graph and its per-commit change-token stream (Groovy).}",
          r"\label{tab:v4-layers}",
          r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"Layer & Nodes & Delta-edges & Node types & Token types & Commits w/ tok. \\",
          r"\midrule"]
    for v in ["V3_ast", "V2a_cfg", "V2b_dfg", "V2c_pdg", "V2d_seq"]:
        if SKEY[v] not in stats:
            continue
        s = stats[SKEY[v]]
        L2.append(f"{NAME[v]} & {s['nodes']:,} & {s['delta_total']:,} & "
                  f"{s['n_node_types']} & {s['n_token_types']} & "
                  f"{s['commits_with_tokens']:,} \\\\".replace(",", "{,}"))
    L2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tab_layers = "\n".join(L2)
    (TAB / "tab_layers.tex").write_text(tab_layers)

    # ---- per-method x variant tables (PR-AUC headline; ROC/F1 also written) ----
    tab_permethod = permethod_table(res, "PR_AUC", "PR-AUC")
    (TAB / "tab_permethod.tex").write_text(tab_permethod)
    (TAB / "tab_permethod_roc.tex").write_text(permethod_table(res, "ROC_AUC", "ROC-AUC"))
    (TAB / "tab_permethod_f1.tex").write_text(permethod_table(res, "F1_online", "F1-online"))
    # per-method CSV (long form)
    with open(TAB / "permethod.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "variant", "PR_AUC", "ROC_AUC", "F1_online"])
        for m in METHOD_ORDER:
            for v in ORDER:
                mm = res[v]["methods"][m]
                w.writerow([METHOD_NAME[m], NAME[v], f"{mm['PR_AUC']:.4f}",
                            f"{mm['ROC_AUC']:.4f}", f"{mm['F1_online']:.4f}"])

    print("=== tab_results.tex ===\n" + tab_results)
    print("\n=== tab_layers.tex ===\n" + tab_layers)
    print("\n=== tab_permethod.tex ===\n" + tab_permethod)
    print(f"\nwrote -> {TAB}/ results.csv, permethod.csv, tab_results.tex, "
          f"tab_layers.tex, tab_permethod{{,_roc,_f1}}.tex")


if __name__ == "__main__":
    main()
