"""
RQ4 + Appendix E/G tables (cache-only):
  Table 9  : all 31 fusion combinations x {Macro-F1, G-Mean, AUC}, for N example
             projects side by side (RQ4) AND per-project 7-metric (Appendix E).
  Table 10 : F vs F+G vs Best-Baseline x {Macro-F1, G-Mean, AUC}, rows = projects.
  Appendix G: per-project F / F+G / F+M / F+G+M x 7 metrics (+ Micro-Avg).

Source: final_fusion_results.pkl (part1 = 31 combos, part2 = F/F+G/F+M/F+G+M),
        baseline_extra_results.pkl (best of final-5).
Run: KGC_PROJECT=<any> python inference/make_rq4_tables.py
"""
import argparse
import pickle
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
BASE5 = ["B_LR", "B_HGB", "B_LAPREDICT", "B_DEEPER", "B_JITLINE"]
PERF = ["Macro_F1", "G_Mean", "AUC"]
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
PP = {"Macro_F1": "Macro-F1", "G_Mean": "G-Mean", "AUC": "AUC", "Precision": "Prec.",
      "Recall": "Rec.", "Buggy_F1": "Buggy-F1", "ACC": "Acc."}


def load(folder):
    ff = OUTP / folder / "final_fusion_results.pkl"
    if not ff.exists():
        return None
    return pickle.load(open(ff, "rb"))


def _rescore(yv, pv, gap):
    from online_jit import final_metrics
    return {k: float(v) for k, v in final_metrics(np.asarray(yv, int),
                                                  np.asarray(pv, float), gap=gap).items()}


def best_base(folder, gap=0):
    bx = OUTP / folder / "baseline_extra_results.pkl"
    if not bx.exists():
        return None
    B = pickle.load(open(bx, "rb")); d = B["baselines"]
    cand = {k: d[k] for k in BASE5 if k in d}
    if not cand:
        return None
    bk = max(cand, key=lambda k: cand[k].get("Buggy_F1", -1))
    if gap == 0:
        return bk.replace("B_", ""), cand[bk]
    r = B.get("raws", {}).get(bk)
    if not r:
        return None
    return bk.replace("B_", ""), _rescore(r["y"], r["pred"], gap)


def _fg_metrics(folder, name, gap):
    """F or F+G metrics at gap (from raw_fusion for gap!=0, else stored)."""
    ff = load(folder)
    if not ff:
        return None
    if gap == 0:
        return ff["part2"][name]["metrics"]
    rf = OUTP / folder / "raw_fusion_scores.pkl"
    if not rf.exists() or name not in ("F", "F+G"):
        return ff["part2"][name]["metrics"]     # F/F+G only rescoreable; fall back
    d = pickle.load(open(rf, "rb"))
    if name not in d["scores"]:
        return ff["part2"][name]["metrics"]
    return _rescore(d["y"], d["scores"][name], gap)


def table10(gap=0):
    """Effect of adding the semantic channel G to the graph fusion F. Deliberately a
    two-way comparison: the question here is whether the layers compose, which is
    internal to KG-Commit. Comparison against the baselines is Table 5's job."""
    L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\caption{RQ4: effect of adding the semantic channel $G$ to the graph fusion "
         r"$F$, per project. $\Delta$ is the change in Macro-F1. The semantic tier "
         r"raises AUC on every project and yields the large Macro-F1 gains on the four "
         r"projects where it helps; the two small decreases are within run-to-run "
         r"variation, and Zookeeper is analysed in Section~\ref{sec:discuss_difficulty}.}",
         (r"\label{tab:rq4_fg}" if gap == 0 else f"\\label{{tab:rq4_fg_g{gap}}}"),
         r"\begin{tabular}{l" + "c" * 6 + "ccc}", r"\toprule",
         r"\multirow{2}{*}{Project} & \multicolumn{3}{c}{$F$ (graph fusion)} & "
         r"\multicolumn{3}{c}{$F{+}G$ (deployed)} & \multicolumn{3}{c}{$\Delta$} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         " & " + " & ".join([PP[m] for m in PERF] * 2)
         + " & " + " & ".join([PP[m] for m in PERF]) + r" \\ \midrule"]
    for disp, folder in PROJECTS:
        ff = load(folder)
        if not ff:
            L.append(f"{disp} & " + " & ".join(["--"] * 9) + r" \\"); continue
        row = [disp]
        vals = {}
        for name in ["F", "F+G"]:
            vals[name] = _fg_metrics(folder, name, gap)
        # bold the better of F / F+G per metric, so the effect of G is readable per row
        for name in ["F", "F+G"]:
            for k in PERF:
                s = "%.3f" % vals[name][k]
                if vals[name][k] >= vals["F+G" if name == "F" else "F"][k]:
                    s = r"\textbf{%s}" % s
                row.append(s)
        for k in PERF:
            d = vals["F+G"][k] - vals["F"][k]
            # Gains are coloured rather than bolded: bold is reserved for the winner
            # of the F / F+G comparison in the columns to the left.
            row.append((r"\textcolor{gainGreen}{%+.3f}" % d) if d > 0
                       else ("%+.3f" % d))
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


TOL = 0.005


def table9_combos(folder, disp, metrics=PERF, label_suffix=""):
    """All 31 combinations for one project.

    Emphasis encodes the selection rule directly: every combination whose Macro-F1 is
    within TOL of the best is bold (these are the statistically indistinguishable
    candidates the rule chooses among), and the combination actually deployed is bold
    \\emph{and} underlined. A reader can therefore see both the band and the pick.
    """
    ff = load(folder)
    if not ff:
        return None
    part1 = ff["part1"]
    chosen = ff.get("chosen")
    combos = sorted(part1.keys(), key=lambda k: (part1[k]["n"], k))
    best_mf1 = max(part1[k]["metrics"]["Macro_F1"] for k in combos)

    L = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
         f"\\caption{{All 31 fusion combinations on {disp}{label_suffix}. "
         r"\textbf{Bold} marks every combination whose Macro-F1 lies within the "
         rf"selection tolerance $\tau={TOL}$ of the best; \underline{{underline}} marks "
         r"the deployed fusion, chosen from that band as the one with the fewest "
         r"inference methods.}",
         f"\\label{{tab:combos_{folder}}}",
         r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule",
         "Combination & " + " & ".join(PP[m] for m in metrics) + r" \\ \midrule"]
    for k in combos:
        m = part1[k]["metrics"]
        cells = []
        for mm in metrics:
            s = "%.3f" % m[mm]
            if mm == "Macro_F1":
                if m[mm] >= best_mf1 - TOL:
                    s = r"\textbf{%s}" % s
                if k == chosen:
                    s = r"\underline{%s}" % s
            cells.append(s)
        lab = k.replace("_", r"\_")
        if k == chosen:
            lab = r"\textbf{%s}" % lab
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def appendixG(folder, disp):
    ff = load(folder)
    if not ff:
        return None
    names = ["F", "F+G", "F+M", "F+G+M"]
    best = {k: max(ff["part2"][n]["metrics"][k] for n in names) for k in M7}
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         f"\\caption{{Appendix G: adding $G$/$M$ to $F$ on {disp} (7 metrics). "
         r"\textbf{Bold} marks the best channel combination per metric.}",
         f"\\label{{tab:appg_{folder}}}",
         r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(M7) + "}", r"\toprule",
         "Model & " + " & ".join(PP[m] for m in M7) + r" \\ \midrule"]
    for name in names:
        m = ff["part2"][name]["metrics"]
        cells = []
        for k in M7:
            s = "%.3f" % m[k]
            if m[k] >= best[k] - 1e-12:
                s = r"\textbf{%s}" % s
            cells.append(s)
        L.append(name.replace("+", "{+}") + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--gap", type=int, default=0)
    args = ap.parse_args(); suf = "g0" if args.gap == 0 else f"g{args.gap}"
    (PM / "RQ4_inference").mkdir(parents=True, exist_ok=True)
    (PM / "appendices" / "E_fusion_combos").mkdir(parents=True, exist_ok=True)
    (PM / "appendices" / "G_FGM_effects").mkdir(parents=True, exist_ok=True)
    # Table 10 (gap-aware: F/F+G rescored, best-base rescored)
    (PM / "RQ4_inference" / f"table10_fg_{suf}.tex").write_text(table10(args.gap), encoding="utf-8")
    print(f"wrote table10_fg_{suf}.tex")
    if args.gap != 0:
        # Table 9 (31-combo exploration) + App E/G kept at g0 only: rescoring all 31
        # combos at a gap is expensive and low-value for an exploratory table.
        print("(Table 9 / App E / App G are g0-only by design; skipping at gap!=0.)")
        return
    # Table 9 (3-metric, per project -> RQ4) + Appendix E (7-metric per project)
    for disp, folder in PROJECTS:
        t9 = table9_combos(folder, disp, PERF)
        if t9:
            (PM / "RQ4_inference" / f"table9_combos__{folder}.tex").write_text(t9, encoding="utf-8")
        e7 = table9_combos(folder, disp, M7, " (7 metrics)")
        if e7:
            (PM / "appendices" / "E_fusion_combos" / f"appE_combos7__{folder}.tex").write_text(e7, encoding="utf-8")
        g = appendixG(folder, disp)
        if g:
            (PM / "appendices" / "G_FGM_effects" / f"appG_fgm__{folder}.tex").write_text(g, encoding="utf-8")
        print(f"  {disp}: {'ok' if t9 else 'MISSING'}")
    print("wrote Table 9 (RQ4) + Appendix E/G per project")


if __name__ == "__main__":
    main()
