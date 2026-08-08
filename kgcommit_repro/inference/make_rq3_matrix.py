"""
RQ3/RQ4 Table 8: the comprehensive matrix.
Per project, a block of rows (RN, PPR, LP, DW, KGE, F=Fusion) x graph columns
(Core, +AST, +CFG, +DFG, +PDG, +CSTG) x metric sub-columns (Macro-F1, G-Mean, AUC).

Source: final_experiments_results.pkl -> [graph][method]['metrics'] and
        [graph]['Fusion']['metrics'] (the per-graph 5-method fusion = row 'F').
Cache-only. Projects lacking per-graph 'Fusion' (groovy/mapreduce until re-run) get
'--' in the F row for the affected graphs.

Run:  KGC_PROJECT=<any> python inference/make_rq3_matrix.py [--out DIR]
Out:  <out>/table8_matrix__<project>.tex   (one file per project; assemble as needed)
"""
import argparse
import pickle
from pathlib import Path
import numpy as np

import _kgc_paths  # noqa: F401
from online_jit import final_metrics
import run_final_fusion as rff
import run_final_experiments as rfe

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
GRAPHS = [("core", "Core"), ("ast", "+AST"), ("cfg", "+CFG"), ("dfg", "+DFG"),
          ("pdg", "+PDG"), ("final", "+CSTG")]
METHODS = ["RN", "PPR", "LP", "DW", "KGE", "Fusion"]
MLAB = {"RN": "RN", "PPR": "PPR", "LP": "LP", "DW": "DW", "KGE": "KGE", "Fusion": "F"}
PERF = ["Macro_F1", "G_Mean", "AUC"]
PP = {"Macro_F1": "MF1", "G_Mean": "GM", "AUC": "AUC"}


def _rescore_graph(folder, g, gap):
    """gap-aware per-method + per-graph-Fusion metrics for one graph, from cached raw
    per-method scores. Returns {method: metrics-dict} (incl. 'Fusion')."""
    rmp = OUTP / folder / "raw_method_scores.pkl"
    if not rmp.exists():
        return None
    rm = pickle.load(open(rmp, "rb"))
    if g not in rm.get("scores", {}):
        return None
    y = np.asarray(rm["y"], int); N = rm["N"]; W = rm["warmup"]
    ev = np.arange(W, N); yt = y[ev]
    out = {}
    for m in rfe.METHODS:
        p = np.asarray(rm["scores"][g][m], float)
        pt = np.clip(np.nan_to_num(p[ev], nan=float(yt.mean())), 0, 1)
        out[m] = {k: float(v) for k, v in final_metrics(yt, pt, gap=gap).items()}
    # per-graph 5-method fusion, re-stacked with the gap
    scores = {m: np.asarray(rm["scores"][g][m], float) for m in rfe.METHODS}
    mm, _, _, _ = rff.eval_subset(scores, list(rfe.METHODS), y, W, N, gap=gap)
    out["Fusion"] = {k: float(mm[k]) for k in mm}
    return out


def cell(fe, g, method, resc=None):
    if resc is not None:
        gm = resc.get(g)
        return gm.get(method) if gm else None
    node = fe.get(g, {})
    if method not in node:
        return None
    m = node[method].get("metrics") if isinstance(node[method], dict) else None
    return m


def render(folder, disp, gap=0):
    p = OUTP / folder / "final_experiments_results.pkl"
    if not p.exists():
        return None
    fe = pickle.load(open(p, "rb"))
    resc = None
    if gap != 0:
        resc = {}
        for g, _ in GRAPHS:
            gm = _rescore_graph(folder, g, gap)
            if gm:
                resc[g] = gm
        if not resc:
            return None
    ncol = len(GRAPHS) * len(PERF)
    L = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{2.2pt}",
         f"\\caption{{RQ3/RQ4 matrix for {disp}: inference methods (rows) $\\times$ "
         r"representation (columns) $\times$ \{Macro-F1, G-Mean, AUC\}. F is the "
         r"5-method fusion.}}",
         f"\\label{{tab:matrix_{folder}}}",
         r"\begin{tabular}{l" + ("c" * ncol) + "}", r"\toprule",
         "Method & " + " & ".join(r"\multicolumn{3}{c}{%s}" % lab for _, lab in GRAPHS) + r" \\"]
    cmids = []; s = 2
    for _ in GRAPHS:
        cmids.append(r"\cmidrule(lr){%d-%d}" % (s, s + 2)); s += 3
    L.append(" ".join(cmids))
    L.append(" & " + " & ".join([PP[m] for m in PERF] * len(GRAPHS)) + r" \\ \midrule")
    for method in METHODS:
        cells = [MLAB[method]]
        for g, _ in GRAPHS:
            m = cell(fe, g, method, resc)
            for mm in PERF:
                v = m.get(mm) if m else None
                cells.append("%.3f" % v if (v is not None and v == v) else "--")
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "Paper" / "paper_material" / "RQ3_subgraphs"))
    args = ap.parse_args()
    suf = "g0" if args.gap == 0 else f"g{args.gap}"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rq4 = ROOT / "Paper" / "paper_material" / "RQ4_inference"; rq4.mkdir(parents=True, exist_ok=True)
    for disp, folder in PROJECTS:
        t = render(folder, disp, args.gap)
        if t:
            (out / f"table8_matrix__{folder}_{suf}.tex").write_text(t, encoding="utf-8")
            (rq4 / f"table8_matrix__{folder}_{suf}.tex").write_text(t, encoding="utf-8")
            print(f"  {disp}: ok")
        else:
            print(f"  {disp}: MISSING (needs raw scores for gap!=0)" if args.gap else f"  {disp}: MISSING")
    print(f"wrote table8 matrices ({suf}) -> {out}")


if __name__ == "__main__":
    main()
