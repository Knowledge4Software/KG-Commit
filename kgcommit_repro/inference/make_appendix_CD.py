"""
Appendix C: per-project full 7-metric table, F+G vs the five baselines (+ Micro-Avg).
Appendix D: per-project per-layer 7-metric table
            (Core, +AST, +CFG, +DFG, +PDG, +CSTG) using the per-graph fusion (F).
Cache-only. gap-aware via --gap (Setting B re-scores from raw where available).
"""
import argparse
import pickle
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from online_jit import final_metrics
import common_window as cw

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
BASE5 = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
         ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]
GRAPHS = [("core", "Core"), ("ast", "Core+AST"), ("cfg", "Core+CFG"),
          ("dfg", "Core+DFG"), ("pdg", "Core+PDG"), ("final", "Core+AST+CSTG")]
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
PP = {"Precision": "Prec.", "Recall": "Rec.", "Macro_F1": "Macro-F1", "Buggy_F1": "Buggy-F1",
      "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Acc."}


def fmt(v):
    return "%.3f" % v if (v is not None and v == v) else "--"


def _rescore(yv, pv, gap):
    yv = np.asarray(yv, int); pv = np.asarray(pv, float)
    return {k: float(v) for k, v in final_metrics(yv, pv, gap=gap).items()}



def _emit(rows, keys, first_col_label, bold_best=True):
    """Rows = [(label, metricdict)]. With bold_best, mark the column-wise maximum, i.e.
    the best model (App C) or best architecture (App D) for that project on that metric.
    Appendix C sets bold_best=False: those tables are a plain per-project reference, and
    the winner-per-metric claim is already made in the main results table."""
    best = {}
    if bold_best:
        for k in keys:
            vs = [m.get(k) for _, m in rows
                  if m and m.get(k) == m.get(k) and m.get(k) is not None]
            if vs:
                best[k] = max(vs)
    out = []
    for lab, m in rows:
        cells = []
        for k in keys:
            v = m.get(k) if m else None
            s = fmt(v)
            if v is not None and v == v and k in best and v >= best[k] - 1e-12:
                s = r"\textbf{%s}" % s
            cells.append(s)
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    return out

def appendixC(folder, disp, gap):
    """Common-window: F+G and all baselines scored on [ev0, N) (identical commits)."""
    ff = OUTP / folder / "final_fusion_results.pkl"
    bx = OUTP / folder / "baseline_extra_results.pkl"
    if not (ff.exists() and bx.exists()):
        return None
    B = pickle.load(open(bx, "rb"))
    ev0 = cw.common_ev0(folder)
    rf = OUTP / folder / "raw_fusion_scores.pkl"
    if ev0 is None or not rf.exists():
        return None
    d = pickle.load(open(rf, "rb"))
    fg_m = _rescore(d["y"], d["scores"]["F+G"], gap)        # F+G on [ev0,N)
    rows = [("KG-Commit ($F{+}G$)", fg_m)]
    for k, lab in BASE5:
        if k in B.get("raws", {}):
            bm = cw.metrics_from_raw(B["raws"][k], ev0, gap=gap)  # baseline on [ev0,N)
            if bm:
                rows.append((lab, bm))
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         f"\\caption{{Appendix C: full 7-metric comparison on {disp} "
         r"(common evaluation window $[W{+}300,N)$).}",
         f"\\label{{tab:appC_{folder}}}",
         r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(M7) + "}", r"\toprule",
         "Model & " + " & ".join(PP[m] for m in M7) + r" \\ \midrule"]
    L += _emit(rows, M7, "Model", bold_best=False)
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def appendixD(folder, disp, gap):
    fe = OUTP / folder / "final_experiments_results.pkl"
    if not fe.exists():
        return None
    E = pickle.load(open(fe, "rb"))
    L = [r"\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
         f"\\caption{{Appendix D: per-layer 7-metric (5-method fusion F) on {disp}. "
         r"\textbf{Bold} marks the best representation per metric.}",
         f"\\label{{tab:appD_{folder}}}",
         r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(M7) + "}", r"\toprule",
         "Representation & " + " & ".join(PP[m] for m in M7) + r" \\ \midrule"]
    drows = []
    for g, lab in GRAPHS:
        node = E.get(g, {})
        fus = node.get("Fusion") if isinstance(node, dict) else None
        m = fus.get("metrics") if isinstance(fus, dict) else None
        drows.append((lab, m or {}))
    L += _emit(drows, M7, "Representation")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.40)
    args = ap.parse_args()
    suf = "g0" if args.gap == 0 else f"g{args.gap}"
    (PM / "appendices" / "C_perproject_baselines").mkdir(parents=True, exist_ok=True)
    (PM / "appendices" / "D_perproject_layers").mkdir(parents=True, exist_ok=True)
    for disp, folder in PROJECTS:
        c = appendixC(folder, disp, args.gap)          # gap-aware (F+G + baselines from raws)
        if c:
            (PM / "appendices" / "C_perproject_baselines" / f"appC__{folder}_{suf}.tex").write_text(c, encoding="utf-8")
        # Appendix D (per-graph fusion) is only emitted at gap=0: faithfully rescoring
        # each per-graph fusion at a gap needs re-running the per-graph stack (not just
        # cached p), so we do not emit a gap!=0 D to avoid a misleading duplicate.
        if args.gap == 0:
            d = appendixD(folder, disp, args.gap)
            if d:
                (PM / "appendices" / "D_perproject_layers" / f"appD__{folder}_{suf}.tex").write_text(d, encoding="utf-8")
        print(f"  {disp}: C={'ok' if c else '--'} D={'ok' if (args.gap==0 and c) else 'g0-only'}")
    print(f"wrote Appendix C ({suf})" + ("" if args.gap else " + D (g0)"))


if __name__ == "__main__":
    main()
