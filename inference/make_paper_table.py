"""
Generate the Groovy portion of the paper's RQ3 "massive evaluation matrix":
5 inference methods (rows) x {V1 Core | V2: AST,CFG,DFG,PDG | V3 Core+AST+CSTG}
(column groups), 5 metrics each (P, R, F1, G, AUC). Real numbers from the final
experiments; only Groovy, no placeholder rows for other projects.

Run:  python inference/make_paper_table.py   ->  outputs/tables/v4/tab_paper_groovy.tex
"""
import pickle
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
TAB = OUT / "tables" / "v4"; TAB.mkdir(parents=True, exist_ok=True)
R = pickle.load(open(OUT / "final_experiments_results.pkl", "rb"))

METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
# column groups in paper order: V1 Core | V2 AST,CFG,DFG,PDG | V3 final
GROUPS = ["core", "ast", "cfg", "dfg", "pdg", "final"]
# paper metric columns P R F1 G AUC  ->  our metric keys
MK = ["Precision", "Recall", "Buggy_F1", "G_Mean", "AUC"]


def fmt(v):
    return f"{v:.2f}"[1:] if v < 1 else f"{v:.2f}"    # ".44" style, no leading 0


def main():
    body = [r"\multirow{5}{*}{\texttt{Groovy}}"]
    for k, m in enumerate(METHODS):
        cells = []
        for g in GROUPS:
            met = R[g][m]["metrics"]
            cells += [fmt(met[mk]) for mk in MK]
        lead = " " if k else ""     # first data row sits on the \multirow line
        body.append(f"{lead} & {m} & " + " & ".join(cells) + r" \\")
    table = r"""\begin{table*}[p!]
\centering
\caption{Cumulative online (prequential) predictive performance on the Apache
\texttt{Groovy} project (RQ3, real-time deployment setting) across the five
graph-inference methods (RN, PPR, LP, DW, KGE). Columns follow the additive,
hierarchical KG-Commit progression: base representation (\textbf{V1}: Core), the
alternative structural subgraph choices at the \textbf{V2} layer (AST, CFG, DFG,
PDG, each added on Core), and the fully unified multi-layer architecture
(\textbf{V3}: Core$+$AST$+$CSTG). Metrics: precision (P), recall (R), buggy
$F_1$, G-mean (G) and AUC. Only \texttt{Groovy} is instantiated here; the
remaining projects of the paper table are omitted.}
\label{tab:massive_evaluation_matrix_groovy}

\tiny
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.2}

\resizebox{0.95\textwidth}{!}{%
\begin{tabular}{ll ccccc ccccc ccccc ccccc ccccc ccccc}
\toprule
\textbf{Dataset} & \textbf{Inf.} & \multicolumn{5}{c}{\textbf{V1}} & \multicolumn{20}{c}{\textbf{V2 Subgraph Architecture}} & \multicolumn{5}{c}{\textbf{V3}} \\
\cmidrule(lr){3-7} \cmidrule(lr){8-27} \cmidrule(lr){28-32}
 & & \multicolumn{5}{c}{Core} & \multicolumn{5}{c}{AST} & \multicolumn{5}{c}{CFG} & \multicolumn{5}{c}{DFG} & \multicolumn{5}{c}{PDG} & \multicolumn{5}{c}{Full Layers w/AST} \\
\cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17} \cmidrule(lr){18-22} \cmidrule(lr){23-27} \cmidrule(lr){28-32}
 & & P & R & $F_1$ & G & AUC & P & R & $F_1$ & G & AUC & P & R & $F_1$ & G & AUC & P & R & $F_1$ & G & AUC & P & R & $F_1$ & G & AUC & P & R & $F_1$ & G & AUC \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}"""
    (TAB / "tab_paper_groovy.tex").write_text(table)
    print(table)
    print("\nwrote -> " + str(TAB / "tab_paper_groovy.tex"))


if __name__ == "__main__":
    main()
