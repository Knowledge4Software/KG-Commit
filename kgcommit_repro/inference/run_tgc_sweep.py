#!/usr/bin/env python3
"""
TGC exhaustive sweep --- every inference-level candidate, on the Kafka pilot.
============================================================================

v1 found global context helps overall but hurt Omega; v2 showed that was a
regularisation artifact (L1 recovers Omega). This script sweeps the whole
design space systematically so the choice of final methodology is evidence-based
rather than a guess, and so the negative results are on record too.

Axes swept:
  1. TIER SET        which tiers enter the feature block
  2. REGULARISATION  L1 / L2 across a C grid (the v2 finding, mapped properly)
  3. FEATURE COUNT   full (44) vs tiny (6) vs single summary
  4. MECHANISM       diffusion over IMPORTS, on its own and combined
  5. COMBINATION     stack onto F+G, onto F alone (replace G), or standalone
  6. AGGREGATION     defect-density only vs full statistic set

Every arm is scored on IDENTICAL rows against the same deployed baseline, with
a paired bootstrap on ALL commits and on the Omega subset.

Out: outputs/<project>/global_context/sweep_results.csv
Run: KGC_PROJECT=kafka python inference/run_tgc_sweep.py
"""
from __future__ import annotations

import csv
import itertools
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402
from online_jit import final_metrics  # noqa: E402
from protocol import BLOCK  # noqa: E402
from run_tgc_pilot import build_features, omega_ids  # noqa: E402
from run_tgc_v2 import diffusion_scores  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEST = OUT / "global_context"
ROOT = HERE.parent.parent

TIER_SETS = {
    "T0":                 ["T0"],
    "T0+T1":              ["T0", "T1"],
    "T0+T1dir":           ["T0", "T1_dep", "T1_dry"],
    "T0+T1dir+TP":        ["T0", "T1_dep", "T1_dry", "TP"],
    "T0+T1dir+TP+TC":     ["T0", "T1_dep", "T1_dry", "TP", "TC"],
    "T0+T1dir+T2+TP":     ["T0", "T1_dep", "T1_dry", "T2", "TP"],
    "T1dir_only":         ["T1_dep", "T1_dry"],
}

REG_GRID = [("l2", 1.0), ("l2", 0.1), ("l1", 0.05), ("l1", 0.2), ("l1", 0.5)]


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
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=C,
            solver="liblinear" if penalty == "l1" else "lbfgs",
            penalty=penalty)
        clf.fit(sc.transform(Z[past]), y[past])
        p[i:j] = clf.predict_proba(sc.transform(Z[i:j]))[:, 1]
        i = j
    return np.nan_to_num(p[np.arange(W, N)], nan=float(y[np.arange(W, N)].mean()))


def boot(y, pa, pb, metric="Macro_F1", n=300, seed=0):
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
    return {"diff": float(d.mean()), "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


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
    print(f"[{PROJECT}] SWEEP | rows={len(ev)} omega={mask.sum()} "
          f"buggy_omega={int(yc[mask].sum())}", flush=True)

    cache = {}

    def feats(names):
        key = tuple(names)
        if key not in cache:
            X, _, ids_all = build_features(blob, names)
            pos = {c: k for k, c in enumerate(ids_all)}
            cache[key] = X[np.array([pos[c] for c in fids])]
        return cache[key]

    diff_all = diffusion_scores(blob)
    dpos = {c["id"]: k for k, c in enumerate(blob["commits"])}
    diff = np.array([diff_all[dpos[c]] for c in fids])

    rows = []
    base_p = dep[ev]
    bm = final_metrics(yc, np.clip(base_p, 0, 1))
    bo = final_metrics(yc[mask], np.clip(base_p[mask], 0, 1))
    rows.append(["deployed (F+G)", "-", "-", "-", 0,
                 round(float(bm["Macro_F1"]), 4), round(float(bm["MCC"]), 4),
                 round(float(bm["AUC"]), 4),
                 round(float(bo["Macro_F1"]), 4), round(float(bo["MCC"]), 4),
                 "", "", "", ""])
    print(f"  BASELINE  ALL F1={bm['Macro_F1']:.4f} MCC={bm['MCC']:.4f} | "
          f"OM F1={bo['Macro_F1']:.4f} MCC={bo['MCC']:.4f}", flush=True)

    def record(label, tier, reg, C, block, nfeat):
        p = stack(block, y, W, C=C, penalty=reg)
        m = final_metrics(yc, np.clip(p, 0, 1))
        mo = final_metrics(yc[mask], np.clip(p[mask], 0, 1))
        sa = boot(yc, p, base_p, "Macro_F1")
        so = boot(yc[mask], p[mask], base_p[mask], "Macro_F1")
        both = (m["Macro_F1"] > bm["Macro_F1"]) and (mo["Macro_F1"] > bo["Macro_F1"])
        rows.append([label, tier, reg, C, nfeat,
                     round(float(m["Macro_F1"]), 4), round(float(m["MCC"]), 4),
                     round(float(m["AUC"]), 4),
                     round(float(mo["Macro_F1"]), 4), round(float(mo["MCC"]), 4),
                     round(sa["diff"], 4) if sa else "", round(sa["p"], 4) if sa else "",
                     round(so["diff"], 4) if so else "", round(so["p"], 4) if so else ""])
        print(f"  {label:<34} ALL F1={m['Macro_F1']:.4f} MCC={m['MCC']:.4f} | "
              f"OM F1={mo['Macro_F1']:.4f} MCC={mo['MCC']:.4f}"
              f"{'  <<< BOTH UP' if both else ''}", flush=True)

    # --- axis 1+2: tier set x regularisation -----------------------------
    for tname, tiers in TIER_SETS.items():
        X = feats(tiers)
        for reg, C in REG_GRID:
            record(f"F+G +{tname} [{reg},C={C}]", tname, reg, C,
                   [dep, X], X.shape[1])

    # --- axis 3: feature count -------------------------------------------
    Xfull = feats(TIER_SETS["T0+T1dir+TP"])
    tiny_idx = [i for i in (8, 9, 10, 16, 17, 24) if i < Xfull.shape[1]]
    Xtiny = Xfull[:, tiny_idx]
    for reg, C in REG_GRID:
        record(f"F+G +tiny6 [{reg},C={C}]", "tiny6", reg, C, [dep, Xtiny], Xtiny.shape[1])

    # --- axis 4: mechanism -------------------------------------------------
    record("F+G +diffusion", "diff", "l2", 1.0, [dep, diff.reshape(-1, 1)], 1)
    record("F+G +tiny6 +diffusion", "tiny6+diff", "l2", 1.0,
           [dep, Xtiny, diff], Xtiny.shape[1] + 1)
    record("F+G +full +diffusion [l1]", "full+diff", "l1", 0.05,
           [dep, Xfull, diff], Xfull.shape[1] + 1)

    # --- axis 5: combination mode -----------------------------------------
    record("F +tiny6 +diff (replace G)", "replaceG", "l2", 1.0,
           [F_only, Xtiny, diff], Xtiny.shape[1] + 1)
    record("F +full +diff (replace G,l1)", "replaceG", "l1", 0.05,
           [F_only, Xfull, diff], Xfull.shape[1] + 1)
    record("TGC standalone (no deployed)", "standalone", "l1", 0.05,
           [Xfull, diff], Xfull.shape[1] + 1)

    # --- axis 6: aggregation --- defect-density columns only --------------
    dens_idx = [i for i in range(Xfull.shape[1]) if i % 8 in (1, 2)]
    Xdens = Xfull[:, dens_idx]
    for reg, C in (("l2", 1.0), ("l1", 0.05)):
        record(f"F+G +density-only [{reg}]", "density", reg, C,
               [dep, Xdens], Xdens.shape[1])

    with (DEST / "sweep_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "tier_set", "penalty", "C", "n_features",
                    "all_MacroF1", "all_MCC", "all_AUC",
                    "omega_MacroF1", "omega_MCC",
                    "d_all_MacroF1", "p_all", "d_omega_MacroF1", "p_omega"])
        w.writerows(rows)
    print(f"\n{len(rows)-1} arms swept -> {DEST/'sweep_results.csv'}")


if __name__ == "__main__":
    main()
