"""
Within-project JIT-defect-prediction BASELINES for the active project.
=====================================================================

Provides the standard change-metric baselines that the KG method is compared
against, evaluated under the SAME online prequential protocol as the KG models
(warm-up on the first WARMUP_FRAC of commits, then predict-then-learn in blocks),
so the numbers sit in the same table as final_experiments.

Baselines (all on the 12 ApacheJIT change metrics la,ld,nf,nd,ns,ent,ndev,age,
nuc,aexp,arexp,asexp -- NO knowledge graph, NO commit text):

  * B_LR    Logistic Regression (class-weighted)          -- linear JIT baseline
  * B_RF    Random Forest                                  -- non-linear JIT baseline
  * B_HGB   HistGradientBoosting                           -- strong tabular baseline
  * B_ALL1  "all buggy"   (predicts p=1 for everything)    -- naive reference
  * B_ALL0  "all benign"  (predicts p=0 for everything)    -- naive reference
  * B_RATE  historical base rate (past bug-rate as score)  -- trivial prior

Each is scored with the project's full metric suite (final_metrics: ROC/PR-AUC,
online-tuned Precision/Recall/Buggy-F1/Macro-F1/G-Mean/MCC/ACC) AND the
effort-aware metrics (Popt, ACC@20%LOC) using la+ld as inspection effort.

Reads the ApacheJIT label CSV (in place) -- needs NO Neo4j and NO KG build, so it
runs anywhere, even while another project's graph is building.

Out: outputs/<project>/baseline_results.pkl  + printed table
Run: KGC_PROJECT=zookeeper python baselines/run_baselines.py
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# baselines/ is a package subfolder; put the package root on path so _kgc_paths
# (and config.project_config) resolve when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import CSV_PATH, OUT, PROJECT  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# reuse the EXACT metric definitions the KG models use, so baselines are
# directly comparable (same online-tuned operating point, same effort metrics).
# IMPORTANT: warm-up fraction must match the KG experiments' protocol. The KG
# graph/fusion experiments use online_infer.WARMUP_FRAC (0.40); online_jit's is
# 0.30. Import the SAME 0.40 so the baseline is warmed/scored over an identical
# window and the per-project comparison figures line up on the x-axis.
from online_jit import final_metrics, BLOCK
from online_infer import WARMUP_FRAC
import effort_metrics as em
# reuse the SAME 7-metric online-trajectory helper + resolution the KG trends use,
# so the baseline stream trend is drawn identically (same stride/window/decisions).
from run_subgraph_rq import metric_trajectory, TRAJ_STRIDE, TRAJ_WINDOW

JIT_COLS = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc",
            "aexp", "arexp", "asexp"]
METRIC_KEYS = ["ROC_AUC", "PR_AUC", "F1", "MCC", "Brier", "Acc", "F1_online",
               "Precision", "Recall", "Buggy_F1", "Macro_F1", "G_Mean",
               "AUC", "ACC", "Popt", "ACC20"]


def _lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              solver="lbfgs")


def load_project():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("author_date").reset_index(drop=True)   # chronological
    y = df["buggy"].astype(int).to_numpy()
    X = df[JIT_COLS].fillna(0.0).to_numpy(float)
    effort = (df["la"].fillna(0) + df["ld"].fillna(0)).to_numpy(float) + 1.0
    return X, y, effort


def prequential_scores(X, y, model_factory, refit_every=3):
    """Predict-then-learn in blocks, refit on the expanding past window. Returns a
    per-commit probability array (nan before warm-up)."""
    N = len(y); W = int(N * WARMUP_FRAC)
    preds = np.full(N, np.nan)
    scaler = StandardScaler().fit(X[:W])
    clf = model_factory().fit(scaler.transform(X[:W]), y[:W])
    i = W; blk = 0
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        preds[idx] = clf.predict_proba(scaler.transform(X[idx]))[:, 1]
        if blk % refit_every == 0:
            scaler = StandardScaler().fit(X[:j])
            clf = model_factory().fit(scaler.transform(X[:j]), y[:j])
        i = j; blk += 1
    return preds, W


def naive_scores(y, kind):
    """Non-learned references over the same evaluated range."""
    N = len(y); W = int(N * WARMUP_FRAC)
    p = np.full(N, np.nan)
    ev = np.arange(W, N)
    if kind == "all1":
        p[ev] = 1.0
    elif kind == "all0":
        p[ev] = 0.0
    elif kind == "rate":
        # historical bug rate up to each commit (leakage-free prior)
        run = np.cumsum(y) - y            # count strictly before i
        denom = np.arange(N)
        rate = np.divide(run, np.maximum(denom, 1))
        p[ev] = rate[ev]
    return p, W


def evaluate(y, p, effort, W):
    ev = np.arange(W, len(y))
    yt = y[ev]; pt = np.nan_to_num(p[ev], nan=float(yt.mean()))
    leaf = final_metrics(yt, pt)
    leaf = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in leaf.items()}
    # effort-aware (uses per-commit inspection effort = la+ld)
    leaf["Popt"] = float(em.popt(yt, pt, effort[ev]))
    leaf["ACC20"] = float(em.recall_at_effort(yt, pt, effort[ev], 0.20))
    return leaf


def main():
    X, y, effort = load_project()
    N = len(y); W = int(N * WARMUP_FRAC)
    print(f"[{PROJECT}] baselines over {N} commits (warm-up {W}, "
          f"eval {N - W}), bug rate {y.mean():.3f}\n")

    results = {}
    preds_by = {}                    # keep per-commit preds for the trajectory
    learned = {
        "B_LR":  lambda: _lr(),
        "B_RF":  lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                n_jobs=-1, random_state=0),
        "B_HGB": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    for name, fac in learned.items():
        p, w = prequential_scores(X, y, fac)
        results[name] = evaluate(y, p, effort, w)
        preds_by[name] = p
    for name, kind in [("B_ALL1", "all1"), ("B_ALL0", "all0"), ("B_RATE", "rate")]:
        p, w = naive_scores(y, kind)
        results[name] = evaluate(y, p, effort, w)
        preds_by[name] = p

    # trajectory of the strongest learned baseline (by Buggy-F1), computed with the
    # SAME helper/resolution as the KG trends so the per-project stream figure
    # overlays them fairly. Also keep its raw per-commit stream for pooled re-plots.
    best = max(("B_LR", "B_RF", "B_HGB"), key=lambda b: results[b]["Buggy_F1"])
    ev = np.arange(W, N); yt = y[ev]
    p_best = np.nan_to_num(preds_by[best][ev], nan=float(yt.mean()))
    base_traj = metric_trajectory(yt, p_best, W)
    base_raw = {"idx": (W + np.arange(len(ev))).tolist(),
                "y": yt.astype(int).tolist(), "pred": p_best.astype(float).tolist(),
                "traj_stride": TRAJ_STRIDE, "traj_window": TRAJ_WINDOW}

    out = dict(project=PROJECT, N=N, warmup=W, n_eval=N - W,
               bug_rate=float(y.mean()), metric_keys=METRIC_KEYS,
               baselines=results, jit_cols=JIT_COLS,
               best_baseline=best, baseline_traj=base_traj, baseline_raw=base_raw)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / "baseline_results.pkl", "wb"))

    hdr = f"{'baseline':<10}{'PR_AUC':>8}{'Buggy_F1':>9}{'G_Mean':>8}{'AUC':>7}{'Popt':>7}{'ACC20':>7}"
    print(hdr); print("-" * len(hdr))
    for name, m in results.items():
        print(f"{name:<10}{m['PR_AUC']:8.3f}{m['Buggy_F1']:9.3f}{m['G_Mean']:8.3f}"
              f"{m['AUC']:7.3f}{m['Popt']:7.3f}{m['ACC20']:7.3f}")
    print(f"\nsaved -> {OUT / 'baseline_results.pkl'}")


if __name__ == "__main__":
    main()
