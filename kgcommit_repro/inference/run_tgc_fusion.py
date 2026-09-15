#!/usr/bin/env python3
"""
TGC fusion --- does global context IMPROVE the deployed KG-Commit model?
========================================================================

The tier ladder showed that global context beats a commit-local baseline. The
question that decides the paper is different and harder: does adding global
context to the ALREADY DEPLOYED model (F+G) make it better -- on all commits
and on the Omega subset?

Arms:
  deployed          the current F+G score, unchanged (the baseline to beat)
  TGC               global-context features alone
  deployed+TGC      prequential stacking of the two
  deployed+TGC_dir  stacking with the directional split (dependants/dependencies)

Stacking is itself prequential: at each block the meta-learner is fitted only on
strictly-past commits, so the no-leakage guarantee is preserved end to end.

Significance: paired bootstrap over commits on the metric difference, plus a
McNemar test on the decisions, both computed on the same evaluated set.

Out: outputs/<project>/global_context/fusion_results.{pkl,csv}
Run: KGC_PROJECT=kafka python inference/run_tgc_fusion.py
"""
from __future__ import annotations

import csv
import pickle
import sys
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

TIERS_MERGED = ["T0", "T1", "TP"]
TIERS_DIRECTIONAL = ["T0", "T1_dep", "T1_dry", "TP"]


def stack(base_score, X, y, block=BLOCK, init=None):
    """Prequential stacking: meta-learner on [deployed score | TGC features]."""
    N = len(y)
    W = init if init is not None else max(int(N * 0.2), 50)
    Z = np.column_stack([base_score, X])
    p = np.full(N, np.nan)
    i = W
    while i < N:
        j = min(N, i + block)
        past = np.arange(i)
        if len(np.unique(y[past])) < 2:
            p[i:j] = base_score[i:j]
            i = j
            continue
        sc = StandardScaler().fit(Z[past])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(sc.transform(Z[past]), y[past])
        p[i:j] = clf.predict_proba(sc.transform(Z[i:j]))[:, 1]
        i = j
    ev = np.arange(W, N)
    return ev, np.nan_to_num(p[ev], nan=float(y[ev].mean()))


def paired_bootstrap(y, pa, pb, metric="Macro_F1", n=1000, seed=0):
    """P(metric(a) > metric(b)) and a 95% CI on the difference."""
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
    if len(d) < 50:
        return None
    d = np.asarray(d)
    return {"mean_diff": float(d.mean()),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def mcnemar(y, pa, pb):
    ya = online_decisions(np.clip(pa, 0, 1), y)
    yb = online_decisions(np.clip(pb, 0, 1), y)
    a = int(((ya == y) & (yb != y)).sum())
    b = int(((ya != y) & (yb == y)).sum())
    if a + b == 0:
        return None
    from scipy.stats import binomtest
    return {"a_only": a, "b_only": b, "p": float(binomtest(a, a + b, 0.5).pvalue)}


def report(y, p, ids, om):
    out = {}
    m = final_metrics(y, np.clip(p, 0, 1))
    out["all"] = {k: float(m[k]) for k in ("Macro_F1", "MCC", "AUC", "Buggy_F1", "G_Mean")}
    mask = np.array([i in om for i in ids])
    if mask.sum() >= 10 and len(np.unique(y[mask])) > 1:
        mo = final_metrics(y[mask], np.clip(p[mask], 0, 1))
        out["omega"] = {k: float(mo[k]) for k in ("Macro_F1", "MCC", "AUC", "Buggy_F1", "G_Mean")}
        out["omega"]["n"] = int(mask.sum())
    else:
        out["omega"] = None
    return out, mask


def main():
    base = ROOT / "outputs" / PROJECT
    blob = pickle.load(open(DEST / "tiers.pkl", "rb"))
    fu = pickle.load(open(base / "raw_fusion_scores.pkl", "rb"))
    mp = pd.read_csv(base / "raw_method_scores.csv",
                     usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om = omega_ids()

    fidx = np.asarray(fu["commit_index"], int)
    fy = np.asarray(fu["y"], int)
    deployed = np.asarray(fu["scores"]["F+G"], float)
    fids = [imap.get(int(i), "") for i in fidx]

    print(f"[{PROJECT}] TGC fusion  |  deployed commits={len(fidx)}  omega known={len(om)}")

    rows, results = [], {}

    def add(label, y, p, ids, ref=None):
        r, mask = report(y, p, ids, om)
        results[label] = r
        a, o = r["all"], r["omega"]
        line = (f"  {label:<22} ALL MacroF1={a['Macro_F1']:.4f} MCC={a['MCC']:.4f} "
                f"AUC={a['AUC']:.4f}")
        if o:
            line += f"   OMEGA(n={o['n']}) MacroF1={o['Macro_F1']:.4f} MCC={o['MCC']:.4f}"
        print(line)
        rows.append([label, round(a["Macro_F1"], 4), round(a["MCC"], 4), round(a["AUC"], 4),
                     round(o["Macro_F1"], 4) if o else "", round(o["MCC"], 4) if o else "",
                     o["n"] if o else ""])
        return r, mask

    # The stacking meta-learner needs its own warm-up, which shortens the
    # evaluated set. Every arm -- including the deployed baseline -- must be
    # scored on the SAME rows, or the Omega subset differs between arms and the
    # comparison is not like-for-like.
    STACK_W = max(int(len(fy) * 0.2), 50)
    ev_common = np.arange(STACK_W, len(fy))
    y_c = fy[ev_common]
    ids_c = [fids[i] for i in ev_common]
    print(f"  common evaluated rows: {len(ev_common)} (stacking warm-up {STACK_W})")

    # --- baseline: the deployed model, unchanged, on the common rows ------
    add("deployed (F+G)", y_c, deployed[ev_common], ids_c)

    # --- TGC features, aligned to the deployed evaluation set -------------
    for label, tier_names in (("deployed+TGC", TIERS_MERGED),
                              ("deployed+TGC_dir", TIERS_DIRECTIONAL)):
        X, y_all, ids_all = build_features(blob, tier_names)
        pos = {cid: k for k, cid in enumerate(ids_all)}
        sel = np.array([pos[c] for c in fids if c in pos])
        if len(sel) != len(fids):
            print(f"  !! alignment mismatch for {label}: {len(sel)}/{len(fids)}")
            continue
        Xs = X[sel]
        ev, p = stack(deployed, Xs, fy, init=STACK_W)
        assert np.array_equal(ev, ev_common), "arm evaluated on different rows"
        yy, ids = y_c, ids_c
        r, mask = add(label, yy, p, ids)

        # significance against the deployed model on the SAME evaluated rows
        base_p = deployed[ev_common]
        sig_all = paired_bootstrap(yy, p, base_p, "Macro_F1")
        mc = mcnemar(yy, p, base_p)
        results[label]["vs_deployed_all"] = {"bootstrap": sig_all, "mcnemar": mc}
        if sig_all:
            print(f"      vs deployed [ALL]   dMacroF1={sig_all['mean_diff']:+.4f} "
                  f"CI[{sig_all['ci_lo']:+.4f},{sig_all['ci_hi']:+.4f}] "
                  f"p={sig_all['p_two_sided']:.4f}"
                  + (f"  McNemar p={mc['p']:.3e} ({mc['a_only']}/{mc['b_only']})" if mc else ""))
        if mask.sum() >= 10 and len(np.unique(yy[mask])) > 1:
            sig_om = paired_bootstrap(yy[mask], p[mask], base_p[mask], "Macro_F1")
            results[label]["vs_deployed_omega"] = sig_om
            if sig_om:
                print(f"      vs deployed [OMEGA] dMacroF1={sig_om['mean_diff']:+.4f} "
                      f"CI[{sig_om['ci_lo']:+.4f},{sig_om['ci_hi']:+.4f}] "
                      f"p={sig_om['p_two_sided']:.4f}")

    DEST.mkdir(parents=True, exist_ok=True)
    pickle.dump({"project": PROJECT, "results": results},
                open(DEST / "fusion_results.pkl", "wb"))
    with (DEST / "fusion_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "all_MacroF1", "all_MCC", "all_AUC",
                    "omega_MacroF1", "omega_MCC", "omega_n"])
        w.writerows(rows)
    print(f"\nsaved -> {DEST/'fusion_results.csv'}")


if __name__ == "__main__":
    main()
