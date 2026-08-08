"""
D1 (why online evaluation): the leakage demonstration.
For each baseline, compare its metrics under (a) the honest ONLINE prequential
protocol vs (b) a RANDOM 70/30 split (which leaks future information). Random-split
scores are inflated -- that inflation is the leakage the paper's protocol avoids.

Cache-only: uses each baseline's per-commit raw scores (raws = {idx,y,pred}) from
baseline_extra_results.pkl. Online = the stored metrics; Random = re-score the SAME
per-commit predictions but pick the threshold on a random subset (oracle-ish),
approximating the optimism a non-chronological split introduces.

NOTE: this is a threshold-side approximation of leakage using cached predictions
(no re-training under a random split, which would need the features). It is labelled
as such. A fully-trained random-split variant is a C3 item (needs re-fit).

Writes discussions/D1_online_eval/{tab_leakage.tex, leakage_delta.pdf}.
"""
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "discussions" / "D1_online_eval"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
BASE5 = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
         ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]


def best_thr(y, p):
    order = np.argsort(-p)
    yy = y[order]; pp = p[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(yy.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return pp[int(np.argmax(f1))] if len(pp) else 0.5


def metrics_at(y, p, thr):
    yh = (p >= thr).astype(int)
    rec = recall_score(y, yh, pos_label=1, zero_division=0)
    spec = recall_score(y, yh, pos_label=0, zero_division=0)
    return {"Macro_F1": f1_score(y, yh, average="macro", zero_division=0),
            "G_Mean": float(np.sqrt(max(rec, 0) * max(spec, 0))),
            "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan}


def eval_random(y, p, rng, frac=0.3):
    """Random-split optimism: tune threshold on a random 'test' subset (leaks)."""
    idx = rng.permutation(len(y)); te = idx[:int(len(y)*frac)]
    thr = best_thr(y[te], p[te])            # threshold picked ON the test subset (leak)
    return metrics_at(y[te], p[te], thr)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # collect mean online vs random Macro-F1 per baseline across projects
    deltas = {lab: [] for _, lab in BASE5}
    rowsT = []   # (project, baseline, online MF1, random MF1)
    for disp, folder in PROJECTS:
        bx = OUTP / folder / "baseline_extra_results.pkl"
        if not bx.exists():
            continue
        B = pickle.load(open(bx, "rb"))
        for key, lab in BASE5:
            on = B["baselines"].get(key, {}).get("Macro_F1")
            r = B["raws"].get(key)
            if on is None or not r:
                continue
            y = np.asarray(r["y"], int); p = np.asarray(r["pred"], float)
            rnd = eval_random(y, p, rng)["Macro_F1"]
            deltas[lab].append(rnd - on)
            rowsT.append((disp, lab, on, rnd))
    if not rowsT:
        print("no baseline data"); return
    # figure: mean random-minus-online Macro-F1 per baseline (leakage inflation)
    labs = [l for _, l in BASE5]; means = [np.mean(deltas[l]) if deltas[l] else 0 for l in labs]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(labs, means, color="#d62728")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("$\Delta$ Macro-F1")
    ax.set_title("Leakage inflation", weight="bold")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"leakage_delta.{e}", bbox_inches="tight")
    plt.close(fig)
    # table (mean per baseline)
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\caption{D1: leakage from a non-chronological split. Mean Macro-F1 under "
         r"the honest online protocol vs.\ a random 70/30 split (threshold tuned on the "
         r"held-out subset), averaged over projects. The gap is optimism the online "
         r"protocol avoids. (Threshold-side approximation from cached predictions.)}",
         r"\label{tab:d1_leakage}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Baseline & Online Macro-F1 & Random-split Macro-F1 & Inflation \\ \midrule"]
    for key, lab in BASE5:
        ons = [o for (_, l, o, _) in rowsT if l == lab]
        rns = [r for (_, l, _, r) in rowsT if l == lab]
        if ons:
            L.append(f"{lab} & {np.mean(ons):.3f} & {np.mean(rns):.3f} & "
                     f"{np.mean(rns)-np.mean(ons):+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / "tab_leakage.tex").write_text("\n".join(L), encoding="utf-8")
    print("wrote D1 leakage table + figure")


if __name__ == "__main__":
    main()
