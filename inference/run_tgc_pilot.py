#!/usr/bin/env python3
"""
TGC pilot --- the tier ladder, on all commits and on the Omega subset.
======================================================================

Turns the extracted tiers into history-aware features and runs the ladder that
is meant to become the paper's headline experiment:

    T0  ->  +T1  ->  +T2  ->  +TP  ->  +TC

Each rung adds graph reach. If the claim "KG-Commit supplies project context
beyond the commit's neighbourhood" is true, performance should rise along the
ladder, and it must rise on ALL COMMITS as well as on Omega -- a design that
lifts only Omega has overfit the motivating example.

Every feature is computed from strictly-past commits (expanding window), so the
prequential guarantee matches the deployed protocol exactly. No Neo4j on the
scoring path: tiers.pkl is read once, everything else is numpy.

Out: outputs/<project>/global_context/pilot_results.{pkl,csv}
Run: KGC_PROJECT=kafka python inference/run_tgc_pilot.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402
from online_jit import final_metrics  # noqa: E402
from protocol import WARMUP_FRAC, BLOCK  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEST = OUT / "global_context"
ROOT = HERE.parent.parent

# the ladder: each rung ADDS a tier to the previous one
LADDER = [
    ("T0",              ["T0"]),
    ("T0+T1",           ["T0", "T1"]),
    ("T0+T1+T2",        ["T0", "T1", "T2"]),
    ("T0+T1+T2+TP",     ["T0", "T1", "T2", "TP"]),
    ("T0+T1+T2+TP+TC",  ["T0", "T1", "T2", "TP", "TC"]),
]
# directional arm, tested separately (dependants vs dependencies)
DIRECTIONAL = ("T0+T1dep+T1dry", ["T0", "T1_dep", "T1_dry"])


class PastState:
    """Strictly-past, per-file history. Updated only after a commit is scored."""

    def __init__(self):
        self.n_changes = defaultdict(int)
        self.n_buggy = defaultdict(int)
        self.authors = defaultdict(set)
        self.last_seen = defaultdict(float)
        self.global_changes = 0
        self.global_buggy = 0

    def observe(self, files, buggy, ts, author=None):
        for f in files:
            self.n_changes[f] += 1
            self.n_buggy[f] += int(buggy)
            self.last_seen[f] = ts
            if author:
                self.authors[f].add(author)
        self.global_changes += 1
        self.global_buggy += int(buggy)

    def prior(self):
        return (self.global_buggy / self.global_changes) if self.global_changes else 0.0

    def summarise(self, files, ts):
        """Eight history-aware statistics for one tier. Identity is never used."""
        n = len(files)
        if n == 0:
            p = self.prior()
            return [0.0, p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        chg = np.array([self.n_changes[f] for f in files], float)
        bug = np.array([self.n_buggy[f] for f in files], float)
        seen = chg > 0
        # smoothed past defect density of the tier, backed off to the global prior
        p = self.prior()
        dens = (bug.sum() + 2.0 * p) / (chg.sum() + 2.0) if chg.sum() else p
        max_dens = float((bug[seen] / chg[seen]).max()) if seen.any() else p
        n_auth = np.array([len(self.authors[f]) for f in files], float)
        age = np.array([ts - self.last_seen[f] if self.last_seen[f] else 0.0
                        for f in files], float)
        return [
            float(np.log1p(n)),                 # tier size (blast radius)
            float(dens),                        # past defect density
            float(max_dens),                    # worst file in the tier
            float(np.log1p(chg.mean())),        # churn / volatility
            float(np.log1p(chg.max())),
            float(n_auth.mean()),               # ownership diffusion
            float(seen.mean()),                 # share with any history
            float(np.log1p(np.median(age)) if len(age) else 0.0),
        ]


def build_features(tiers_blob, tier_names):
    """Causal sweep: summarise tiers from past state, then fold the commit in."""
    commits = tiers_blob["commits"]
    tiers = tiers_blob["tiers"]
    st = PastState()
    X, y, ids = [], [], []
    for c in commits:
        cid, ts, buggy = c["id"], float(c["ts"]), int(c["buggy"])
        t = tiers.get(cid)
        row = []
        for name in tier_names:
            files = t.get(name, ()) if t else ()
            row.extend(st.summarise(files, ts))
        # relative reach: how much context lies OUTSIDE what the commit touched.
        # emitted for every commit (zero when the commit has no tiers) so the
        # design matrix stays rectangular
        n0 = max(len(t.get("T0", ())), 1) if t else 1
        for name in tier_names:
            if name != "T0":
                k = len(t.get(name, ())) if t else 0
                row.append(float(np.log1p(k / n0)))
        X.append(row)
        y.append(buggy)
        ids.append(cid)
        st.observe(t.get("T0", ()) if t else (), buggy, ts)
    return np.asarray(X, float), np.asarray(y, int), ids


def prequential(X, y, warmup_frac=WARMUP_FRAC, block=BLOCK):
    """Predict-then-learn on an expanding window -- the deployed protocol."""
    N = len(y)
    W = max(int(N * warmup_frac), 30)
    p = np.full(N, np.nan)
    i = W
    while i < N:
        j = min(N, i + block)
        past = np.arange(i)
        if len(np.unique(y[past])) < 2:
            p[i:j] = y[past].mean() if len(past) else 0.5
            i = j
            continue
        sc = StandardScaler().fit(X[past])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(sc.transform(X[past]), y[past])
        p[i:j] = clf.predict_proba(sc.transform(X[i:j]))[:, 1]
        i = j
    ev = np.arange(W, N)
    return ev, np.nan_to_num(p[ev], nan=float(y[ev].mean()))


def omega_ids():
    f = ROOT / "outputs" / "import_handling_check" / \
        f"{PROJECT}_orphaned_dependant_cases.csv"
    if not f.exists():
        return set()
    import pandas as pd
    return set(pd.read_csv(f)["commit"].astype(str))


def evaluate(y, p, ids, om):
    out = {}
    m = final_metrics(y, np.clip(p, 0, 1))
    out["all"] = {k: float(m[k]) for k in ("Macro_F1", "MCC", "AUC", "Buggy_F1", "G_Mean")}
    mask = np.array([i in om for i in ids])
    if mask.sum() >= 10 and len(np.unique(y[mask])) > 1:
        mo = final_metrics(y[mask], np.clip(p[mask], 0, 1))
        out["omega"] = {k: float(mo[k]) for k in ("Macro_F1", "MCC", "AUC", "Buggy_F1", "G_Mean")}
        out["omega"]["n"] = int(mask.sum())
        out["omega"]["n_buggy"] = int(y[mask].sum())
    else:
        out["omega"] = None
    return out


def main():
    blob = pickle.load(open(DEST / "tiers.pkl", "rb"))
    om = omega_ids()
    print(f"[{PROJECT}] TGC pilot  |  omega commits known: {len(om)}")
    print(f"tier stats: " + ", ".join(
        f"{k}={v['mean']:.0f}" for k, v in blob["tier_stats"].items()))

    results, rows = {}, []
    for label, names in LADDER + [DIRECTIONAL]:
        t0 = time.time()
        X, y, ids = build_features(blob, names)
        ev, p = prequential(X, y)
        r = evaluate(y[ev], p, [ids[i] for i in ev], om)
        r["n_features"] = int(X.shape[1])
        r["seconds"] = round(time.time() - t0, 1)
        results[label] = r
        a, o = r["all"], r["omega"]
        print(f"  {label:<18} feats={X.shape[1]:>3}  "
              f"ALL MacroF1={a['Macro_F1']:.4f} MCC={a['MCC']:.4f} AUC={a['AUC']:.4f}"
              + (f"   OMEGA(n={o['n']}) MacroF1={o['Macro_F1']:.4f} MCC={o['MCC']:.4f}"
                 if o else "   OMEGA n/a")
              + f"   [{r['seconds']}s]")
        rows.append([label, X.shape[1],
                     round(a["Macro_F1"], 4), round(a["MCC"], 4), round(a["AUC"], 4),
                     round(o["Macro_F1"], 4) if o else "",
                     round(o["MCC"], 4) if o else "",
                     o["n"] if o else ""])

    DEST.mkdir(parents=True, exist_ok=True)
    pickle.dump({"project": PROJECT, "results": results}, open(DEST / "pilot_results.pkl", "wb"))
    with (DEST / "pilot_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "n_features", "all_MacroF1", "all_MCC", "all_AUC",
                    "omega_MacroF1", "omega_MCC", "omega_n"])
        w.writerows(rows)
    print(f"\nsaved -> {DEST/'pilot_results.csv'}")


if __name__ == "__main__":
    main()
