"""
LaTeX tables for the final-fusion analysis (run_final_fusion.py):
  tab_final_fusion_ablation.tex  all 31 method-fusions ranked by (Macro-F1+G-Mean)/2
  tab_final_fusion_gm.tex        F / F+G / F+M / F+G+M
Written to outputs/tables/v4/ and copied for the paper.

Run:  python inference/make_final_fusion_tables.py
"""
import pickle
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
TAB = OUT / "tables" / "v4"; TAB.mkdir(parents=True, exist_ok=True)
R = pickle.load(open(OUT / "final_fusion_results.pkl", "rb"))
M7 = [("Precision", "Prec."), ("Recall", "Rec."), ("Macro_F1", "Macro-F1"),
      ("Buggy_F1", "Buggy-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"), ("ACC", "Acc.")]


def bal(v): return 0.5 * (v["metrics"]["Macro_F1"] + v["metrics"]["G_Mean"])


def ablation_table():
    chosen = R["chosen"]
    items = sorted(R["part1"].items(), key=lambda kv: -bal(kv[1]))
    rows = []
    for rank, (k, v) in enumerate(items, 1):
        m = v["metrics"]
        cells = [f"{rank}", k, str(v["n"]), f"{bal(v):.3f}"] + [f"{m[mk]:.3f}" for mk, _ in M7]
        if k == chosen:
            line = " & ".join((r"\textbf{" + c + "}") for c in cells) + r" \\"
        else:
            line = " & ".join(cells) + r" \\"
        rows.append(line)
    return "\n".join([
        r"\begin{table}[H]", r"\centering \scriptsize",
        (r"\caption{All $2^5-1=31$ fusions of the five graph-inference methods on the "
         r"final Core+AST+CSTG graph (prequential LR stacking), ranked by "
         r"$(\text{Macro-F1}+\text{G-Mean})/2$. The chosen fusion "
         r"$F=$\,RN$+$PPR --- the fewest methods within $0.005$ of the best balance "
         r"--- is highlighted.}"),
        r"\label{tab:v4-final-fusion-ablation}",
        r"\begin{tabular}{rlccrrrrrrr}", r"\toprule",
        r"\# & Fusion & $k$ & bal. & Prec. & Rec. & Macro-F1 & Buggy-F1 & G-Mean & AUC & Acc. \\",
        r"\midrule", "\n".join(rows), r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def gm_table():
    best = {mk: max(R["part2"][n]["metrics"][mk] for n in ["F", "F+G", "F+M", "F+G+M"])
            for mk, _ in M7}
    rows = []
    for name in ["F", "F+G", "F+M", "F+G+M"]:
        m = R["part2"][name]["metrics"]
        cells = []
        for mk, _ in M7:
            s = f"{m[mk]:.3f}"
            cells.append(r"\textbf{" + s + "}" if abs(m[mk] - best[mk]) < 1e-9 else s)
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[H]", r"\centering \small",
        (r"\caption{Effect of appending the CSTG channel ($G$) and the non-graph JIT "
         r"metrics ($M$) to the chosen graph-method fusion $F=$\,RN$+$PPR on the final "
         r"KG. Best per column in \textbf{bold}. Adding $G$ helps markedly; adding $M$ "
         r"alone hurts and adds nothing over $G$.}"),
        r"\label{tab:v4-final-fusion-gm}",
        r"\begin{tabular}{l" + "r" * len(M7) + "}", r"\toprule",
        r"Fusion & " + " & ".join(l for _, l in M7) + r" \\", r"\midrule",
        "\n".join(rows), r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    (TAB / "tab_final_fusion_ablation.tex").write_text(ablation_table())
    (TAB / "tab_final_fusion_gm.tex").write_text(gm_table())
    print(f"chosen F = {R['chosen']}")
    print(f"wrote tab_final_fusion_ablation.tex, tab_final_fusion_gm.tex -> {TAB}")


if __name__ == "__main__":
    main()
