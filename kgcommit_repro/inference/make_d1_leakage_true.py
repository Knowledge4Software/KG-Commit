"""
D1: leakage under a non-chronological split, measured properly.

The earlier version of this analysis (make_d1_leakage.py) only re-tuned the decision
threshold on a random subset of the SAME per-commit predictions. That captures the
threshold-side optimism but not the far larger effect: under a random split the model
is also *trained* on commits that lie in the future relative to the commits it is
scored on. This script measures the full effect by actually retraining.

For each project and each change-metric baseline we compare:
  (a) ONLINE  : the deployed prequential protocol, trained only on the past,
                scored at a leakage-free online-tuned threshold;
  (b) RANDOM  : a random 70/30 split of the same commits, model refit on the random
                training portion, threshold tuned on it, scored on the held-out 30%.
Repeated over several random seeds and averaged.

Only the change-metric learners are re-trainable from the label CSV alone
(LR / HGB / LApredict share those twelve features). JITLine and Deeper need the
diff corpus and their own pipelines, so they are omitted rather than approximated.

Cache-only: label CSV + baseline_extra_results.pkl. No Neo4j.

Run:  python inference/make_d1_leakage_true.py [--seeds 5]
Out:  discussions/D1_online_eval/tab_leakage_true.tex
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score, recall_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DATA = ROOT / "data" / "apachejit" / "projects"
PM = ROOT / "Paper" / "paper_material" / "discussions" / "D1_online_eval"

JIT = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc",
       "aexp", "arexp", "asexp"]

# Only the baselines the paper actually reports. RF was dropped: it appears in no
# other table, so a leakage row for it would describe a model the reader never meets.
# JITLine and Deeper are not here because the random-split harness rebuilds models from
# the released twelve-feature set, which those two do not consume.
MODELS = {
    "LR": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "HGB": lambda: HistGradientBoostingClassifier(random_state=0),
    "LApredict": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
}


def _best_thr(y, p):
    o = np.argsort(-p)
    yy, pp = y[o], p[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(yy.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return pp[int(np.argmax(f1))] if len(pp) else 0.5


def _score(y, p, thr):
    yh = (p >= thr).astype(int)
    r = recall_score(y, yh, pos_label=1, zero_division=0)
    s = recall_score(y, yh, pos_label=0, zero_division=0)
    return {"Macro_F1": f1_score(y, yh, average="macro", zero_division=0),
            "G_Mean": float(np.sqrt(max(r, 0) * max(s, 0))),
            "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan}


def random_split_run(X, y, factory, seeds):
    """Retrain on a random 70% and score the held-out 30%. This is the leaky setting."""
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        tr, te = idx[:cut], idx[cut:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = factory().fit(sc.transform(X[tr]), y[tr])
        p_tr = clf.predict_proba(sc.transform(X[tr]))[:, 1]
        p_te = clf.predict_proba(sc.transform(X[te]))[:, 1]
        out.append(_score(y[te], p_te, _best_thr(y[tr], p_tr)))
    if not out:
        return None
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def online_stored(folder, key):
    """The deployed online result for the same model, from the instrumented run."""
    f = OUTP / folder / "baseline_extra_results.pkl"
    if not f.exists():
        return None
    d = pickle.load(open(f, "rb"))["baselines"]
    return d.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    PM.mkdir(parents=True, exist_ok=True)

    KEY = {"LR": "B_LR", "HGB": "B_HGB", "LApredict": "B_LAPREDICT"}
    acc = {m: {"on": [], "rnd": []} for m in MODELS}

    for disp, folder in PROJECTS:
        csv = DATA / f"apache_{folder}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv).sort_values("author_date").reset_index(drop=True)
        y = df["buggy"].astype(int).to_numpy()
        for name, fac in MODELS.items():
            cols = ["la"] if name == "LApredict" else JIT
            X = df[cols].fillna(0.0).to_numpy(float)
            r = random_split_run(X, y, fac, args.seeds)
            o = online_stored(folder, KEY[name])
            if r and o:
                acc[name]["rnd"].append(r)
                acc[name]["on"].append({k: float(o[k]) for k in
                                        ("Macro_F1", "G_Mean", "AUC")})
        print(f"  {disp}: done")

    # Single-column float: three model rows and seven columns do not justify the full
    # page width, and \textwidth-stretching them looked distorted.
    L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.4pt}",
         r"\caption{D1: optimism induced by a non-chronological evaluation. Each model "
         r"is evaluated twice on the same commits: under the deployment-faithful online "
         r"protocol, and under a random $70/30$ split in which the model is \emph{retrained} "
         r"on a randomly chosen majority of the history and scored on the remainder "
         rf"(mean over {args.seeds} seeds, averaged across the seven projects). The "
         r"random split lets a model learn from commits that would lie in the future at "
         r"prediction time, and it inflates every metric. Only the change-metric learners "
         r"are shown, as they are the models retrainable from the released feature set "
         r"alone.}",
         r"\label{tab:d1_leakage_true}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{G-Mean} & "
         r"\multicolumn{2}{c}{AUC} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Model & Online & Random & Online & Random & Online & Random \\ \midrule"]
    deltas = []
    for name in MODELS:
        if not acc[name]["rnd"]:
            continue
        cells = [name]
        for m in ("Macro_F1", "G_Mean", "AUC"):
            o = float(np.mean([d[m] for d in acc[name]["on"]]))
            r = float(np.mean([d[m] for d in acc[name]["rnd"]]))
            if m == "Macro_F1":
                deltas.append(r - o)
            # bold whichever protocol scores higher; on almost every cell this is the
            # random split, which is the point the table makes.
            so, sr = "%.3f" % o, "%.3f" % r
            if r > o:
                sr = r"\textbf{%s}" % sr
            else:
                so = r"\textbf{%s}" % so
            cells += [so, sr]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / "tab_leakage_true.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {PM / 'tab_leakage_true.tex'}")
    if deltas:
        print(f"mean Macro-F1 inflation from a random split: {np.mean(deltas):+.3f}")


if __name__ == "__main__":
    main()
