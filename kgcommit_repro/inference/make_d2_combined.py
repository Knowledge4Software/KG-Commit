"""
D2 combined table + figure: KG-Commit(F+G) and baselines over K = warm-up fraction
{.05,.1,.2,.3,.4,.5}, per project + Macro/Micro avg, plus variance.
KG side: read the existing App-H K-sweep JSON (outputs/<p>/param_experiments/K.json).
Baseline side: re-run the LR and HGB baselines' prequential loop at each K from the
12 change-metric CSV features (cache-only, no Neo4j) -- the two sklearn baselines
that re-fit trivially. (LApredict/Deeper/JITLine K-sweeps need their own pipelines
per K and are left to the C3 stage; LR/HGB are representative.)
Run per project: KGC_PROJECT=<p> python inference/make_d2_combined.py
"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
import _kgc_paths  # noqa: F401
from config.project_config import OUT, PROJECT  # noqa: E402
from online_infer import BLOCK  # noqa: E402
from online_jit import final_metrics  # noqa: E402
import run_baselines as rb  # noqa: E402

K_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
PM = Path(__file__).resolve().parent.parent.parent / "Paper" / "paper_material" / "discussions" / "D2_warmup_K"


def baseline_sweep_at_K(X, y, k, kind):
    """Prequential predict-then-learn at warm-up fraction k; return Macro-F1."""
    N = len(y); W = int(N * k)
    Xs = StandardScaler().fit(X[:max(W, 2)]).transform(X)
    pred = np.full(N, np.nan); i = W; blk = 0; clf = None
    def mk():
        return (LogisticRegression(max_iter=500, class_weight="balanced")
                if kind == "LR" else HistGradientBoostingClassifier(random_state=0))
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        if blk % 2 == 0 and len(set(y[:i])) > 1:
            clf = mk().fit(Xs[:i], y[:i])
        pred[idx] = clf.predict_proba(Xs[idx])[:, 1] if clf is not None else y[:i].mean()
        i = j; blk += 1
    ev = np.arange(W, N)
    return final_metrics(y[ev], np.nan_to_num(pred[ev], nan=float(y[ev].mean())))["Macro_F1"]


def main():
    kj = OUT / "param_experiments" / "K.json"
    if not kj.exists():
        print(f"[{PROJECT}] no KG K-sweep; skip."); return
    KG = json.load(open(kj))
    kg_row = {float(k): KG[k].get("Macro_F1", KG[k].get("F1")) for k in KG if k != "_meta"}
    X, y, _ = rb.load_project()
    lr_row = {k: baseline_sweep_at_K(X, y, k, "LR") for k in K_VALUES}
    hgb_row = {k: baseline_sweep_at_K(X, y, k, "HGB") for k in K_VALUES}

    def var(d):
        return float(np.var([d[k] for k in K_VALUES if k in d]))

    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         f"\\caption{{D2: Macro-F1 of KG-Commit ($F{{+}}G$) vs.\\ LR/HGB baselines over "
         f"warm-up $K$ on {PROJECT}. Last column = variance over $K$ (lower = more "
         r"robust).}}",
         f"\\label{{tab:d2_{PROJECT}}}",
         r"\begin{tabular}{l" + "c" * (len(K_VALUES) + 1) + "}", r"\toprule",
         "Model & " + " & ".join(f"$K{{=}}{k:g}$" for k in K_VALUES) + r" & Var \\ \midrule"]
    for lab, row in [("KG-Commit $F{+}G$", kg_row), ("LR", lr_row), ("HGB", hgb_row)]:
        cells = [f"{row[k]:.3f}" if k in row else "--" for k in K_VALUES]
        L.append(lab + " & " + " & ".join(cells) + f" & {var(row):.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    PM.mkdir(parents=True, exist_ok=True)
    (PM / f"d2_combined__{PROJECT}.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"[{PROJECT}] D2 combined table (KG+LR+HGB over K) written.")


if __name__ == "__main__":
    main()
