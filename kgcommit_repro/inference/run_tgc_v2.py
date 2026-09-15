#!/usr/bin/env python3
"""
TGC v2 --- three fixes for the split pilot result.
==================================================

The v1 pilot found that stacking 44 TGC features onto the deployed F+G helps
overall (+0.020 Macro-F1, +0.066 MCC, McNemar p=4.7e-03) but HURTS the Omega
subset. Three hypotheses, each tested here as its own arm:

  (1) DILUTION.  44 features against 37 evaluated Omega commits swamps a sparse
      signal. Fix: L1 / small fixed subsets / stronger regularisation.

  (2) WRONG MECHANISM.  Summary statistics over tiers are a statistical proxy.
      Omega is a *mechanistic* claim: risk propagates along dependencies. Fix:
      diffuse strictly-past defect labels over the file import graph and read the
      value off the commit's files (arm C3). This is the strongest untested
      candidate and it targets Omega directly.

  (3) ADDITION vs REPLACEMENT.  TGC may be redundant with what F already
      carries. Fix: replace the graph channel rather than augment it.

All arms are scored on IDENTICAL rows (the v1 bug), with the deployed model on
the same rows as the baseline. Cache-only apart from tiers.pkl; no Neo4j.

Out: outputs/<project>/global_context/v2_results.csv
Run: KGC_PROJECT=kafka python inference/run_tgc_v2.py
"""
from __future__ import annotations

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402
from online_jit import final_metrics, online_decisions  # noqa: E402
from protocol import BLOCK  # noqa: E402
from run_tgc_pilot import build_features, omega_ids  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEST = OUT / "global_context"
ROOT = HERE.parent.parent


# ---------------------------------------------------------------- C3: diffusion
def diffusion_scores(blob, alpha=0.35, iters=12):
    """Propagate strictly-past defect mass over the file import graph.

    For each commit, we hold a per-file 'risk' vector accumulated from past
    buggy commits, diffuse it a few steps along IMPORTS (both directions), and
    read off the value at the files the commit touches. This is the mechanism
    the paper claims -- risk reaching a commit through dependencies it does not
    itself touch -- rather than a summary statistic about the neighbourhood.

    Causality: the risk vector is updated only AFTER a commit is scored.
    """
    commits = blob["commits"]
    tiers = blob["tiers"]

    # build a sparse file graph once from the tier structure's own adjacency
    # (T1_dep = importers, T1_dry = imports-of), symmetric for diffusion
    nbr = defaultdict(set)
    for t in tiers.values():
        if not t:
            continue
        for f in t["T0"]:
            for g in t.get("T1_dep", ()):
                nbr[f].add(g)
                nbr[g].add(f)
            for g in t.get("T1_dry", ()):
                nbr[f].add(g)
                nbr[g].add(f)

    risk = defaultdict(float)      # accumulated past defect mass per file
    out = np.zeros(len(commits))
    for k, c in enumerate(commits):
        t = tiers.get(c["id"])
        files = t["T0"] if t else []
        if files and risk:
            # local diffusion: seed at the commit's files' neighbourhoods
            frontier = {f: 1.0 for f in files}
            acc = 0.0
            seen = set(files)
            for step in range(3):
                nxt = defaultdict(float)
                for f, w in frontier.items():
                    ns = nbr.get(f)
                    if not ns:
                        continue
                    share = w * (1.0 - alpha) / len(ns)
                    for g in ns:
                        acc += share * risk.get(g, 0.0)
                        if g not in seen:
                            nxt[g] += share
                seen |= set(nxt)
                frontier = nxt
                if not frontier:
                    break
            out[k] = acc / max(len(files), 1)
        # fold this commit in afterwards -- strictly past
        if t and c["buggy"]:
            for f in t["T0"]:
                risk[f] += 1.0
    return np.log1p(out)


# ---------------------------------------------------------------- generic stack
def stack(cols, y, W, block=BLOCK, C=1.0, penalty="l2"):
    Z = np.column_stack(cols)
    N = len(y)
    p = np.full(N, np.nan)
    i = W
    while i < N:
        j = min(N, i + block)
        past = np.arange(i)
        if len(np.unique(y[past])) < 2:
            p[i:j] = float(y[past].mean()) if len(past) else 0.5
            i = j
            continue
        sc = StandardScaler().fit(Z[past])
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 C=C, penalty=penalty, solver=solver)
        clf.fit(sc.transform(Z[past]), y[past])
        p[i:j] = clf.predict_proba(sc.transform(Z[i:j]))[:, 1]
        i = j
    ev = np.arange(W, N)
    return np.nan_to_num(p[ev], nan=float(y[ev].mean()))


def boot(y, pa, pb, metric="Macro_F1", n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        a = final_metrics(y[s], np.clip(pa[s], 0, 1)).get(metric)
        b = final_metrics(y[s], np.clip(pb[s], 0, 1)).get(metric)
        if a is not None and b is not None:
            d.append(float(a) - float(b))
    if len(d) < 30:
        return None
    d = np.asarray(d)
    return {"diff": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def main():
    base = ROOT / "outputs" / PROJECT
    blob = pickle.load(open(DEST / "tiers.pkl", "rb"))
    fu = pickle.load(open(base / "raw_fusion_scores.pkl", "rb"))
    mp = pd.read_csv(base / "raw_method_scores.csv",
                     usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om = omega_ids()

    fidx = np.asarray(fu["commit_index"], int)
    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    F_only = np.asarray(fu["scores"]["F"], float)
    fids = [imap.get(int(i), "") for i in fidx]

    W = max(int(len(y) * 0.2), 50)
    ev = np.arange(W, len(y))
    yc = y[ev]
    idc = [fids[i] for i in ev]
    mask = np.array([i in om for i in idc])
    print(f"[{PROJECT}] TGC v2 | rows={len(ev)} omega={mask.sum()} buggy_omega={yc[mask].sum()}")

    # feature blocks, aligned to the deployed rows
    def align(names):
        X, _, ids_all = build_features(blob, names)
        pos = {c: k for k, c in enumerate(ids_all)}
        sel = np.array([pos[c] for c in fids])
        return X[sel]

    X_full = align(["T0", "T1_dep", "T1_dry", "TP"])
    X_small = align(["T0", "T1_dep", "T1_dry"])[:, :]      # smaller tier set
    diff_all = diffusion_scores(blob)
    dpos = {c["id"]: k for k, c in enumerate(blob["commits"])}
    diff = np.array([diff_all[dpos[c]] for c in fids])

    # a deliberately tiny TGC subset: the 6 most direct global signals
    # (dependant defect density / max / size, dependency density / size, reach)
    idx_small = [8, 9, 10, 16, 17, 24]
    idx_small = [i for i in idx_small if i < X_full.shape[1]]
    X_tiny = X_full[:, idx_small]

    arms = {
        "deployed (F+G)":            None,
        "+TGC full (L2)":            (X_full, 1.0, "l2"),
        "+TGC full (L1 sparse)":     (X_full, 0.05, "l1"),
        "+TGC tiny (6 feats)":       (X_tiny, 1.0, "l2"),
        "+diffusion only (C3)":      (diff.reshape(-1, 1), 1.0, "l2"),
        "+TGC tiny +diffusion":      (np.column_stack([X_tiny, diff]), 1.0, "l2"),
        "F + TGC tiny (replace G)":  ("REPLACE", 1.0, "l2"),
    }

    rows, base_all, base_om = [], None, None
    for label, spec in arms.items():
        if spec is None:
            p = dep[ev]
        elif isinstance(spec[0], str) and spec[0] == "REPLACE":
            # drop the CSTG channel G: use F alone plus global context
            p = stack([F_only, X_tiny, diff], y, W, C=spec[1], penalty=spec[2])
        else:
            p = stack([dep, spec[0]], y, W, C=spec[1], penalty=spec[2])

        m = final_metrics(yc, np.clip(p, 0, 1))
        mo = (final_metrics(yc[mask], np.clip(p[mask], 0, 1))
              if mask.sum() >= 10 and len(np.unique(yc[mask])) > 1 else None)
        if base_all is None:
            base_all, base_om = p.copy(), p.copy()
            sa = so = None
        else:
            sa = boot(yc, p, base_all, "Macro_F1")
            so = boot(yc[mask], p[mask], base_all[mask], "Macro_F1") if mo else None

        line = (f"  {label:<27} ALL F1={m['Macro_F1']:.4f} MCC={m['MCC']:.4f} "
                f"AUC={m['AUC']:.4f}")
        if mo:
            line += f" | OM F1={mo['Macro_F1']:.4f} MCC={mo['MCC']:.4f}"
        if sa:
            line += f" | dALL={sa['diff']:+.4f}(p={sa['p']:.3f})"
        if so:
            line += f" dOM={so['diff']:+.4f}(p={so['p']:.3f})"
        print(line, flush=True)
        rows.append([label, round(m["Macro_F1"], 4), round(m["MCC"], 4), round(m["AUC"], 4),
                     round(mo["Macro_F1"], 4) if mo else "",
                     round(mo["MCC"], 4) if mo else "",
                     round(sa["diff"], 4) if sa else "", round(sa["p"], 4) if sa else "",
                     round(so["diff"], 4) if so else "", round(so["p"], 4) if so else ""])

    with (DEST / "v2_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "all_MacroF1", "all_MCC", "all_AUC", "omega_MacroF1",
                    "omega_MCC", "d_all_MacroF1", "p_all", "d_omega_MacroF1", "p_omega"])
        w.writerows(rows)
    print(f"\nsaved -> {DEST/'v2_results.csv'}")


if __name__ == "__main__":
    main()
