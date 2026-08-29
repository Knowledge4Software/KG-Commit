"""
RQ3/RQ4 matrix tables for the Results draft (T6 and T7).
========================================================

T6  tab_rq3_matrix.tex   the [layers x methods] matrix, all 11 projects.
    Rows per project: the five inference methods, then TWO fusion rows --
    the per-project chosen F and the fixed overall F = RN+PPR -- so the paper can
    show both without dropping the per-project analysis. Columns follow the
    hierarchy: Layer 1 (Core), Layer 2 candidates (AST/CFG/DFG/PDG), Layer 3 (CSTG).

T7  tab_rq4_combos_groovy.tex   all 31 fusion combinations on Groovy, regenerated
    from the final data, with the parsimony band and the deployed choice marked.

Cache-only. No Neo4j.

Run: python inference/make_draft_rq3_tables.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
# column order: Core (L1) | AST CFG DFG PDG (L2) | CSTG (L3)
COLS = [("core", "Core"), ("ast", "AST"), ("cfg", "CFG"),
        ("dfg", "DFG"), ("pdg", "PDG"), ("final", "CSTG")]
M3 = ("Macro_F1", "G_Mean", "AUC")
TOL = 0.005

# Per-layer fusion values for the two deployed rules, precomputed from the
# persisted per-commit method scores by rq3_perlayer_fusion.py. Regenerate with
#   python inference/rq3_perlayer_fusion.py
_PL = OUTP / "tables" / "rq3_perlayer_fusion.json"
if not _PL.exists():
    raise SystemExit(
        f"missing {_PL}\nRun: python inference/rq3_perlayer_fusion.py")
PERLAYER = json.loads(_PL.read_text(encoding="utf-8"))


def d3(v):
    """Leading-zero-stripped 3-decimal, matching the existing matrix style."""
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0.") else s


def t6_matrix():
    L = [r"\begin{table*}[tp]", r"\centering",
         r"\caption{RQ3: cumulative predictive performance across all 11 active "
         r"projects under the online protocol, for the five inference methods and "
         r"two fusion rules. Columns follow the hierarchical construction of "
         r"KG-Commit: the Core process layer (Layer~1), each candidate structural "
         r"subgraph attached to it (Layer~2), and the full deployed graph with the "
         r"semantic tier (Layer~3). Per project, the $F_{\mathrm{pp}}$ row names "
         r"that project's parsimony-selected fusion and the $F_{\mathrm{ov}}$ row "
         r"is the fixed overall fusion RN+PPR deployed everywhere. Each fusion is "
         r"selected once, on the deployed CSTG graph, and the same combination is "
         r"then re-evaluated unchanged on every ablated column; it is therefore "
         r"not re-optimised there, and a single method may exceed it. Where a "
         r"rule reduces to one method its row coincides with that method's row. "
         r"Every cell is scored under the single deployed online protocol "
         r"($K{=}0.05$ warm-up, refresh block $M{=}200$, verification gap "
         r"$G{=}50$, adaptive stacking-head initialisation) over one common "
         r"evaluation span, so all rows are directly comparable. Within each "
         r"project block the best value per column is in \textbf{bold}.}",
         r"\label{tab:massive_evaluation_matrix}", r"\scriptsize",
         r"\setlength{\tabcolsep}{2.2pt}",
         r"\renewcommand{\arraystretch}{0.86}",
         # Scale to the text width but cap the height at the available page
         # body, so a 77-row table cannot grow past the bottom margin.
         # \resizebox{\textwidth}{!} preserves the aspect ratio and therefore
         # overflowed vertically; \adjustbox bounds both dimensions.
         r"\adjustbox{max width=\textwidth,max totalheight=0.93\textheight}{%",
         r"\begin{tabular}{ll " + " ".join(["ccc"] * len(COLS)) + "}",
         r"\toprule",
         r"\textbf{Dataset} & \textbf{Inf.} & \multicolumn{3}{c}{\textbf{Layer 1}} "
         r"& \multicolumn{12}{c}{\textbf{Layer 2 (Subgraph Ablation)}} "
         r"& \multicolumn{3}{c}{\textbf{Layer 3}} \\",
         r"\cmidrule(lr){3-5} \cmidrule(lr){6-17} \cmidrule(lr){18-20}",
         " &  & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}"
                               for _, n in COLS) + r" \\",
         " ".join(rf"\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(len(COLS))),
         " &  & " + " & ".join([r"$F_1$ & G & AUC"] * len(COLS)) + r" \\",
         r"\midrule"]

    for pi, p in enumerate(PROJECTS):
        base = OUTP / p / "final_final_run"
        E = pickle.load(open(base / "experiments"
                             / "final_experiments_results.pkl", "rb"))
        F = pickle.load(open(base / "fusion" / "final_fusion_results.pkl", "rb"))
        if pi:
            L.append(r"\midrule")
        L.append(rf"\multirow{{7}}{{*}}{{\texttt{{{DISP[p]}}}}}")

        # Every row -- methods and fusions alike -- comes from PERLAYER, which
        # re-derives the whole matrix on the single final_final_run protocol
        # (K=0.05, M=200, G=50, adaptive init) over one common span. The stored
        # method metrics in E were scored with gap=0 over a different span, so
        # mixing the two sources is what previously made a single-method rule
        # disagree with its own method row. E is still used for the caption's
        # provenance only.
        PL = PERLAYER[p]

        # best value per column, to bold within the project block
        best = {}
        for g, _ in COLS:
            for k in M3:
                vals = [PL["methods"][g][m][k] for m in METHODS]
                vals += [PL[r]["cols"][g][k] for r in ("pp", "ov")]
                vals = [v for v in vals if np.isfinite(v)]
                best[(g, k)] = max(vals) if vals else np.nan

        for m in METHODS:
            cells = []
            for g, _ in COLS:
                mm = PL["methods"][g][m]
                for k in M3:
                    v = mm.get(k, np.nan)
                    s = d3(v)
                    if np.isfinite(v) and np.isfinite(best[(g, k)]) \
                            and abs(v - best[(g, k)]) < 1e-9:
                        s = rf"\textbf{{{s}}}"
                    cells.append(s)
            L.append(f" & {m} & " + " & ".join(cells) + r" \\")

        # The two deployed fusion rules, on every column, from the same
        # re-derivation as the method rows above -- so a rule that reduces to a
        # single method reproduces that method's row exactly.
        for tag, ckey, rk in ((r"F_{\mathrm{pp}}", "chosen", "pp"),
                              (r"F_{\mathrm{ov}}", "chosen_overall", "ov")):
            name = F.get(ckey, "")
            cells = []
            for g, _ in COLS:
                mm = PL[rk]["cols"][g]
                for k in M3:
                    v = mm.get(k, np.nan)
                    s = d3(v)
                    if np.isfinite(v) and np.isfinite(best[(g, k)]) \
                            and abs(v - best[(g, k)]) < 1e-9:
                        s = rf"\textbf{{{s}}}"
                    cells.append(s)
            L.append(rf" & \textbf{{${tag}$}} (\texttt{{\scriptsize {name}}}) & "
                     + " & ".join(cells) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def t7_combos(project="zookeeper"):
    F = pickle.load(open(OUTP / project / "final_final_run" / "fusion"
                         / "final_fusion_results.pkl", "rb"))
    p1 = F["part1"]
    best = max(v["metrics"]["Macro_F1"] for v in p1.values())
    band = {k for k, v in p1.items() if v["metrics"]["Macro_F1"] >= best - TOL}
    chosen = F["chosen"]
    overall = F["chosen_overall"]

    items = sorted(p1.items(), key=lambda kv: (kv[1]["n"],
                                               kv[1]["metrics"]["Macro_F1"]))
    L = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
         rf"\caption{{All 31 fusion combinations on {DISP[project]}. "
         r"\textbf{Bold} marks every combination whose Macro-F1 lies within the "
         r"selection tolerance $\tau=0.005$ of the best; \underline{underline} "
         r"marks the parsimony-selected fusion $F_{\mathrm{pp}}$, chosen from that "
         r"band as the one with the fewest inference methods. The fixed overall "
         r"fusion $F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$ is marked $^{\dagger}$.}",
         r"\label{tab:combos_example}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Combination & Macro-F1 & G-Mean & AUC \\ \midrule"]
    for k, v in items:
        m = v["metrics"]
        nm = k
        if k == chosen:
            nm = rf"\underline{{{nm}}}"
        if k == overall:
            nm = nm + r"$^{\dagger}$"
        f1 = f"{m['Macro_F1']:.3f}"
        if k in band:
            f1 = rf"\textbf{{{f1}}}"
        L.append(f"{nm} & {f1} & {m['G_Mean']:.3f} & {m['AUC']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    (DRAFT / "tab_rq3_matrix.tex").write_text(t6_matrix() + "\n", encoding="utf-8")
    print("  wrote tab_rq3_matrix.tex")
    (DRAFT / "tab_rq4_combos.tex").write_text(t7_combos() + "\n",
                                                     encoding="utf-8")
    print("  wrote tab_rq4_combos.tex")

    F = pickle.load(open(OUTP / "zookeeper" / "final_final_run" / "fusion"
                         / "final_fusion_results.pkl", "rb"))
    print(f"\n  groovy chosen F  = {F['chosen']}")
    print(f"  zookeeper overall F = {F['chosen_overall']}")


if __name__ == "__main__":
    main()
