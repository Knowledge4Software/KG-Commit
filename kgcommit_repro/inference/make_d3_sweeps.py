"""
D3 sweeps (cache-only, per project): the deployed F+G re-scored over
  - M (block size)  in {10,25,50,100,200}
  - G (gap)         in {0,10,25,50,100}
from cached raw scores (raw_method_scores + online_jit_streams for the G channel).
Writes per-project robustness tables + a cross-project trend figure into
discussions/D3_gap_M and appendices/H_param_sensitivity.

Requires: raw_method_scores.pkl, online_jit_streams_v5.pkl, final_fusion_results.pkl.
Run per project: KGC_PROJECT=<p> python inference/make_d3_sweeps.py
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: F401
from config.project_config import OUT, PROJECT  # noqa: E402
import run_final_fusion as rff  # noqa: E402
import run_final_experiments as rfe  # noqa: E402
import online_infer as oi  # noqa: E402

M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
PP = {"Precision": "Prec.", "Recall": "Rec.", "Macro_F1": "Macro-F1", "Buggy_F1": "Buggy-F1",
      "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Acc."}
M_VALUES = [10, 25, 50, 100, 200]
G_VALUES = [0, 10, 25, 50, 100]
PM = Path(__file__).resolve().parent.parent.parent / "Paper" / "paper_material"


def load():
    sp5 = OUT / "raw_method_scores.pkl"; st = OUT / "online_jit_streams_v5.pkl"
    ff = OUT / "final_fusion_results.pkl"
    if not (sp5.exists() and st.exists() and ff.exists()):
        return None
    rm = pickle.load(open(sp5, "rb")); y = np.asarray(rm["y"]); N = rm["N"]; W = rm["warmup"]
    S = pickle.load(open(st, "rb"))
    if not np.array_equal(np.asarray(S["y"]), y):
        return None
    scores = {m: np.asarray(rm["scores"]["final"][m], float) for m in rfe.METHODS}
    G = sp.hstack([sp.csr_matrix(np.hstack([S["cstg_prior"][:, None], S["cstg_typed"],
                                            S["cstg_consist"]])), S["Xcstg"]]).tocsr()
    F = pickle.load(open(ff, "rb"))["F"]
    return scores, G, F, y, W, N


def sweep(kind):
    d = load()
    if not d:
        return None
    scores, Gfeat, F, y, W, N = d
    orig_block = oi.BLOCK
    rows = {}
    vals = M_VALUES if kind == "M" else G_VALUES
    for v in vals:
        if kind == "M":
            oi.BLOCK = v; rff.BLOCK = v; gap = 0
        else:
            oi.BLOCK = orig_block; rff.BLOCK = orig_block; gap = v
        sc = dict(scores); sc["G"] = rff.channel_score(Gfeat, y, W, N, sparse=True, gap=gap)
        m, _, _, _ = rff.eval_subset(sc, F + ["G"], y, W, N, gap=gap)
        rows[v] = {k: float(m[k]) for k in M7}
    oi.BLOCK = orig_block; rff.BLOCK = orig_block
    return rows


def write_table(kind, rows):
    lab = "block size $M$" if kind == "M" else "gap $G$"
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         f"\\caption{{Robustness of $F{{+}}G$ over {lab} on {PROJECT} (7 metrics; "
         r"last row = variance).}}",
         f"\\label{{tab:d3{kind.lower()}_{PROJECT}}}",
         r"\begin{tabular}{l" + "c" * len(M7) + "}", r"\toprule",
         f"{kind} & " + " & ".join(PP[m] for m in M7) + r" \\ \midrule"]
    for v in sorted(rows):
        L.append(str(v) + " & " + " & ".join("%.3f" % rows[v][k] for k in M7) + r" \\")
    var = {k: float(np.var([rows[v][k] for v in rows])) for k in M7}
    L.append(r"\midrule Var & " + " & ".join("%.4f" % var[k] for k in M7) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    d3 = PM / "discussions" / "D3_gap_M"; appH = PM / "appendices" / "H_param_sensitivity"
    for p in (d3, appH):
        p.mkdir(parents=True, exist_ok=True)
    for kind in ("M", "G"):
        rows = sweep(kind)
        if not rows:
            print(f"[{PROJECT}] {kind}-sweep: caches missing/misaligned; skip."); continue
        t = write_table(kind, rows)
        (d3 / f"d3_{kind}sweep__{PROJECT}.tex").write_text(t, encoding="utf-8")
        (appH / f"appH_{kind}sweep__{PROJECT}.tex").write_text(t, encoding="utf-8")
        print(f"[{PROJECT}] {kind}-sweep ok "
              f"(Macro-F1: {[round(rows[v]['Macro_F1'],3) for v in sorted(rows)]})")


if __name__ == "__main__":
    main()
