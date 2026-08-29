"""
Seed confidence for the change-metric baselines, cache-only.
============================================================

The paper reports 5-seed variance for DeepJIT only. The other baselines were
each fitted once with a hardcoded random_state=0, so no variance was ever
measured for them. Two of them are genuinely stochastic:

    RF   (RandomForestClassifier)          bootstrap sampling + feature subsets
    HGB  (HistGradientBoostingClassifier)  binning / subsampling

and two are not:

    LR         (lbfgs)      convex, deterministic solver
    LApredict  (lbfgs on a single feature)

This script re-runs the prequential evaluation of the stochastic ones under
protocol.SEEDS, from the cached Kamei metric block -- no Neo4j, no graph. The
deterministic ones are re-run too, to demonstrate rather than assume their
invariance.

JITLine-online is excluded: its cost is dominated by diff tokenisation that the
cache does not retain, so it cannot be re-fitted here.

Out: outputs/tables/seed_baselines.json
Run: python inference/seed_baselines.py
"""
import json
import os
# The config package binds one project at import time; these are
# cross-project drivers that read per-project files directly, so pin a
# placeholder to satisfy the import without constraining what we read.
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402
from online_jit import final_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
REPORT = ("Macro_F1", "G_Mean", "AUC")


def models(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    return {
        "LR": lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                         solver="lbfgs"),
        "HGB": lambda: HistGradientBoostingClassifier(random_state=seed),
        "RF": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                             class_weight="balanced",
                                             random_state=seed),
        # LApredict: same lbfgs LR, but on the single `la` column. Measured
        # rather than assumed deterministic.
        "LApredict": lambda: LogisticRegression(max_iter=2000,
                                                class_weight="balanced",
                                                solver="lbfgs"),
    }


def prequential(X, y, W, N, mk, block, gap):
    """Same prequential recipe as the baselines: fit on the strictly-past
    window, score the next block, refit every block."""
    p = np.full(N, np.nan)
    i, clf = W, None
    if len(set(y[:W])) > 1:
        clf = mk().fit(X[:W], y[:W])
    while i < N:
        j = min(N, i + block)
        p[i:j] = (clf.predict_proba(X[i:j])[:, 1] if clf
                  else float(y[:i].mean()))
        u = max(W, j - gap)
        if u > W and len(set(y[:u])) > 1:
            clf = mk().fit(X[:u], y[:u])
        i = j
    ev = np.arange(W, N)
    return np.clip(np.nan_to_num(p[ev], nan=float(y[ev].mean())), 0, 1), y[ev]


def jitline_matrix(project, y, W, top_k=2000):
    """[expert metrics || bag-of-token counts] for JITLine, from the on-disk
    diff CSV. Vocabulary is fit on the warm-up window only (leakage-free),
    mirroring run_extra_baselines._jitline_features."""
    import pandas as pd
    from collections import Counter

    diff = (ROOT / "data" / "apachejit" / "projects"
            / f"apache_{project}_diff.csv")
    if not diff.exists():
        return None
    S = pickle.load(open(OUTP / project / "online_jit_streams_v5.pkl", "rb"))
    X = np.nan_to_num(np.asarray(S["Xms"], float), nan=0.0,
                      posinf=0.0, neginf=0.0)

    sys.path.insert(0, str(ROOT / "kgcommit_repro"))
    from inference.cstg import parse_commit_text, commit_terms

    dd = pd.read_csv(diff)
    toks = []
    for txt in dd["diff_text"].fillna(""):
        try:
            toks.append(commit_terms(parse_commit_text(txt)))
        except Exception:
            toks.append([])
    if len(toks) < len(y):
        toks += [[]] * (len(y) - len(toks))
    toks = toks[:len(y)]

    vc = Counter()
    for t in toks[:W]:
        vc.update(t)
    vocab = [w for w, _ in vc.most_common(top_k)]
    vi = {w: i for i, w in enumerate(vocab)}
    T = np.zeros((len(y), len(vocab)), dtype=np.float32)
    for r, t in enumerate(toks):
        for w in t:
            j = vi.get(w)
            if j is not None:
                T[r, j] += 1.0
    return np.hstack([X, T])


def run_project(p):
    f = OUTP / p / "online_jit_streams_v5.pkl"
    if not f.exists():
        return None
    S = pickle.load(open(f, "rb"))
    X = np.asarray(S["Xms"], float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(S["y"], int)
    N = len(y)
    W = int(S.get("W", int(N * P.WARMUP_FRAC)))

    out = {}
    XJ = None
    try:
        XJ = jitline_matrix(p, y, W)
    except Exception as e:
        print(f"    {p}: JITLine features unavailable ({str(e)[:60]})")

    for name in ("LR", "HGB", "RF", "LApredict", "JITLine-online"):
        if name == "JITLine-online" and XJ is None:
            continue
        per_seed = {k: [] for k in REPORT}
        for sd in P.SEEDS:
            mk = (models(sd)["RF"] if name == "JITLine-online"
                  else models(sd)[name])
            if name == "LApredict":
                Xm = X[:, :1]                 # LApredict uses `la` only
            elif name == "JITLine-online":
                Xm = XJ                       # metrics || bag-of-tokens
            else:
                Xm = X
            pr, yy = prequential(Xm, y, W, N, mk, P.BLOCK, P.GAP)
            m = final_metrics(yy, pr, gap=P.GAP)
            for k in REPORT:
                per_seed[k].append(float(m[k]))
        out[name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                         "min": float(min(v)), "max": float(max(v)),
                         "values": v}
                     for k, v in per_seed.items()}
    return out


def main():
    allout = {}
    for p in PROJECTS:
        try:
            r = run_project(p)
        except Exception as e:
            print(f"  {p}: FAILED {str(e)[:90]}")
            continue
        if r is None:
            print(f"  {p}: no stream cache -- skipped")
            continue
        allout[p] = r
        s = "  ".join(f"{m} sd={r[m]['Macro_F1']['sd']:.5f}" for m in r)
        print(f"  {p:11} {s}")

    dst = OUTP / "tables" / "seed_baselines.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(allout, indent=1), encoding="utf-8")
    print(f"\nwrote {dst.relative_to(ROOT)}  ({len(allout)} projects)")


if __name__ == "__main__":
    main()
