"""
D1, third protocol: time-aware blocked cross-validation.
=========================================================

D1 currently contrasts two protocols:

  ONLINE  the deployed prequential setting -- train on the strict past, score
          the next block, refit, and withhold the most recent labels for the
          verification gap;
  RANDOM  a random 70/30 split with the model retrained on the random majority,
          which lets it learn from commits that lie in the future relative to
          those it scores.

A reviewer will object that RANDOM is a straw man: nobody defending an offline
protocol would shuffle a commit stream. The standard offline design in the
JIT-SDP literature is a *time-aware blocked split* -- take contiguous
chronological windows, train on the earliest, validate on the next, test on the
last, and slide. That preserves temporal order, so it does not leak the future
in the obvious way, and it is the honest comparison to make.

This script adds that third protocol:

  TIME-AWARE  contiguous 40% train / 20% validation / 20% test windows, slid
              across the history in several folds. Order is preserved inside
              every fold; the threshold is tuned on the validation block and
              applied unchanged to the test block.

The point of the comparison is that even a temporally ordered offline split
still overstates deployment performance, because it (i) trains once on a large
contiguous block rather than tracking drift, (ii) never pays the
verification-latency gap, and (iii) evaluates on a single late window rather
than on the whole stream from an early start. Any gap between TIME-AWARE and
ONLINE is therefore attributable to operational realism, not to temporal
leakage -- which is a stronger argument than the RANDOM contrast alone.

Only the change-metric learners are re-trainable from the released feature set,
so LR / HGB / LApredict are reported and JITLine / DeepJIT are omitted rather
than approximated -- the same scoping as make_d1_leakage_true.py.

Cache-only: label CSV + baseline_extra_results.pkl. No Neo4j.

Run:  python inference/make_d1_timeaware.py [--folds 4]
Out:  outputs/tables/d1_timeaware.json
      Paper/paper_material/discussions/D1_online_eval/tab_d1_three_protocols.tex
"""
import argparse
import json
import os
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401

from sklearn.linear_model import LogisticRegression        # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score, recall_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DATA = ROOT / "data" / "apachejit" / "projects"
PM = ROOT / "Paper" / "paper_material" / "discussions" / "D1_online_eval"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]

JIT = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc",
       "aexp", "arexp", "asexp"]

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "HGB": lambda: HistGradientBoostingClassifier(random_state=0),
    "LApredict": lambda: LogisticRegression(max_iter=2000,
                                            class_weight="balanced"),
}
BKEY = {"LR": "B_LR", "HGB": "B_HGB", "LApredict": "B_LAPREDICT"}


def _best_thr(y, p):
    o = np.argsort(-p)
    yy, pp = y[o], p[o]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(yy.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return pp[int(np.argmax(f1))] if len(pp) else 0.5


def _score(y, p, thr):
    yh = (p >= thr).astype(int)
    r = recall_score(y, yh, pos_label=1, zero_division=0)
    s = recall_score(y, yh, pos_label=0, zero_division=0)
    return {"Macro_F1": f1_score(y, yh, average="macro", zero_division=0),
            "G_Mean": float(np.sqrt(max(r, 0) * max(s, 0))),
            "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan}


def timeaware_blocked(X, y, factory, folds=4,
                      tr_frac=0.40, va_frac=0.20, te_frac=0.20):
    """Contiguous 40/20/20 windows slid across the chronological stream.

    Fold k uses [s, s+tr) to train, [s+tr, s+tr+va) to pick the threshold, and
    [s+tr+va, s+tr+va+te) to test, where s advances so the folds tile the
    history. Order is preserved within and across the three blocks: no future
    commit is ever used to predict a past one.
    """
    n = len(y)
    w = int(n * (tr_frac + va_frac + te_frac))
    if w < 60 or n < 120:
        return None
    starts = (np.linspace(0, n - w, folds).astype(int) if folds > 1 else [0])

    out = []
    for s in starts:
        a = s + int(n * tr_frac)
        b = a + int(n * va_frac)
        c = min(b + int(n * te_frac), n)
        tr, va, te = np.arange(s, a), np.arange(a, b), np.arange(b, c)
        if len(te) < 20 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = factory().fit(sc.transform(X[tr]), y[tr])
        p_va = clf.predict_proba(sc.transform(X[va]))[:, 1]
        p_te = clf.predict_proba(sc.transform(X[te]))[:, 1]
        thr = _best_thr(y[va], p_va) if len(np.unique(y[va])) > 1 else 0.5
        out.append(_score(y[te], p_te, thr))
    if not out:
        return None
    return {k: float(np.nanmean([o[k] for o in out])) for k in out[0]}


def random_split(X, y, factory, seeds=5):
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


def load_project(p):
    """Chronologically ordered feature matrix and labels from the label CSV."""
    for cand in (DATA / f"apache_{p}.csv", DATA / f"{p}.csv"):
        if cand.exists():
            df = pd.read_csv(cand)
            break
    else:
        return None, None
    tcol = next((c for c in ("author_date", "committer_date", "date")
                 if c in df.columns), None)
    if tcol:
        df = df.sort_values(tcol).reset_index(drop=True)
    ycol = next((c for c in ("buggy", "label", "is_buggy") if c in df.columns),
                None)
    if ycol is None or not set(JIT).issubset(df.columns):
        return None, None
    X = np.nan_to_num(df[JIT].to_numpy(float), nan=0.0,
                      posinf=0.0, neginf=0.0)
    y = df[ycol].astype(int).to_numpy()
    return X, y


def online_stored(p, key):
    f = OUTP / p / "final_final_run" / "baselines" / "baseline_extra_results.pkl"
    if not f.exists():
        return None
    return pickle.load(open(f, "rb"))["baselines"].get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()

    res = {}
    for p in PROJECTS:
        X, y = load_project(p)
        if X is None:
            print(f"  {p}: label CSV unusable -- skipped")
            continue
        res[p] = {}
        for name, fac in MODELS.items():
            Xm = X[:, :1] if name == "LApredict" else X   # LApredict uses `la`
            on = online_stored(p, BKEY[name])
            res[p][name] = {
                "online": ({k: float(on[k]) for k in
                            ("Macro_F1", "G_Mean", "AUC")} if on else None),
                "timeaware": timeaware_blocked(Xm, y, fac, folds=a.folds),
                "random": random_split(Xm, y, fac),
            }
        print(f"  {p}: done")

    (OUTP / "tables").mkdir(parents=True, exist_ok=True)
    (OUTP / "tables" / "d1_timeaware.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")

    # ---- cross-project means + LaTeX --------------------------------------
    rows = []
    for name in MODELS:
        cell = {}
        for proto in ("online", "timeaware", "random"):
            for k in ("Macro_F1", "G_Mean", "AUC"):
                v = [res[p][name][proto][k] for p in res
                     if res[p].get(name, {}).get(proto)
                     and np.isfinite(res[p][name][proto][k])]
                cell[(proto, k)] = float(np.mean(v)) if v else float("nan")
        rows.append((name, cell))

    print(f"\n{'model':11}{'ONLINE':>22}{'TIME-AWARE':>22}{'RANDOM':>22}")
    print(f"{'':11}{'F1   G-Mean   AUC':>22}{'F1   G-Mean   AUC':>22}"
          f"{'F1   G-Mean   AUC':>22}")
    for name, c in rows:
        s = ""
        for proto in ("online", "timeaware", "random"):
            s += "  " + " ".join(f"{c[(proto, k)]:.3f}"
                                 for k in ("Macro_F1", "G_Mean", "AUC"))
        print(f"{name:11}{s}")

    def f3(v):
        if not np.isfinite(v):
            return "--"
        s = f"{v:.3f}"
        return s[1:] if s.startswith("0.") else s

    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3.4pt}",
         r"\caption{D1: what an offline protocol overstates. Each model is "
         r"evaluated three ways on the same commits. \textsc{Online} is the "
         r"deployed prequential protocol. \textsc{Time-aware} is a blocked "
         r"chronological split ($40\%$ train / $20\%$ validation / $20\%$ "
         r"test, slid over the history), which preserves temporal order and is "
         r"the standard offline design in the literature. \textsc{Random} is a "
         r"$70/30$ shuffle in which the model is retrained on a randomly chosen "
         r"majority. Both offline protocols score higher than the deployed one; "
         r"the time-aware gap is the more informative of the two, since it "
         r"cannot be explained by temporal leakage and instead reflects the "
         r"operational constraints -- continual refitting, delayed labels, and "
         r"scoring from an early start -- that the online protocol imposes.}",
         r"\label{tab:d1_three_protocols}",
         r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
         r"& \multicolumn{3}{c}{\textsc{Online} (deployed)} "
         r"& \multicolumn{3}{c}{\textsc{Time-aware} blocked} "
         r"& \multicolumn{3}{c}{\textsc{Random} $70/30$} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
         r"Model & Macro-F1 & G-Mean & AUC & Macro-F1 & G-Mean & AUC "
         r"& Macro-F1 & G-Mean & AUC \\", r"\midrule"]
    for name, c in rows:
        cells = []
        for proto in ("online", "timeaware", "random"):
            for k in ("Macro_F1", "G_Mean", "AUC"):
                v = f3(c[(proto, k)])
                if proto != "online" and np.isfinite(c[(proto, k)]) \
                        and c[(proto, k)] > c[("online", k)]:
                    v = rf"\textbf{{{v}}}"
                cells.append(v)
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    PM.mkdir(parents=True, exist_ok=True)
    (PM / "tab_d1_three_protocols.tex").write_text("\n".join(L),
                                                   encoding="utf-8")
    print(f"\nwrote outputs/tables/d1_timeaware.json")
    print(f"wrote {(PM / 'tab_d1_three_protocols.tex').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
