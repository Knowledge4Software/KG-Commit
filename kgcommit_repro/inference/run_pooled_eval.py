#!/usr/bin/env python3
"""
Pooled cross-project evaluation --- the correct test for the Omega claim.
=========================================================================

Every per-project Omega test in this investigation has had n = 37..67 evaluated
Omega commits. At that size a 0.05 Macro-F1 effect cannot be resolved: all 152
arms produced Omega p-values in [0.21, 0.82], positive and negative alike. A
non-significant negative on one project is NOT evidence against the claim, and
a non-significant positive is not evidence for it.

The claim must therefore be tested on the POOLED sample: ~821 evaluated Omega
commits across the eleven projects, roughly 20x the per-project size and the
only design with the power to resolve an effect of the size observed.

Protocol:
  * The design is FROZEN before this runs -- no per-project tuning.
  * Each project is scored independently under its own prequential protocol
    (no cross-project leakage; the models never see another project).
  * Scores are then POOLED for one significance test, with the project as a
    blocking factor.
  * Two complementary tests:
      - pooled paired bootstrap, resampling COMMITS (power)
      - stratified bootstrap, resampling PROJECTS (generalisation)
    Both are reported; they answer different questions and can disagree.

Out: outputs/_pooled/pooled_results.csv, pooled_significance.json
Run: python inference/run_pooled_eval.py            (no KGC_PROJECT needed)
"""
from __future__ import annotations

import csv
import json
import os
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
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import _kgc_paths  # noqa: E402,F401
from online_jit import final_metrics  # noqa: E402
from protocol import BLOCK  # noqa: E402
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = HERE.parent.parent
DEST = ROOT / "outputs" / "_pooled"

# ---- THE FROZEN DESIGN -------------------------------------------------------
# Fixed after three projects. Chosen for cross-project robustness, not for the
# best single-project number. L1 is mandatory (E3); tiers are one-hop (E1);
# M1 is the only topology block that survived (E5).
FROZEN_TIERS = ["T0", "T1_dep", "T1_dry", "TP"]
FROZEN_PENALTY = "l1"
FROZEN_C = 0.2
USE_M1 = True


def stack(cols, y, W):
    Z = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    N = len(y)
    p = np.full(N, np.nan)
    i = W
    while i < N:
        j = min(N, i + BLOCK)
        past = np.arange(i)
        if len(np.unique(y[past])) < 2:
            p[i:j] = float(y[past].mean()) if len(past) else 0.5
            i = j
            continue
        sc = StandardScaler().fit(Z[past])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 C=FROZEN_C, penalty=FROZEN_PENALTY,
                                 solver="liblinear")
        clf.fit(sc.transform(Z[past]), y[past])
        p[i:j] = clf.predict_proba(sc.transform(Z[i:j]))[:, 1]
        i = j
    return np.nan_to_num(p[np.arange(W, N)], nan=float(y[np.arange(W, N)].mean()))


def score_project(folder):
    """Run the frozen design on one project. Returns per-commit arrays."""
    base = ROOT / "outputs" / folder
    tiers_f = base / "global_context" / "tiers.pkl"
    fu_f = base / "raw_fusion_scores.pkl"
    ms_f = base / "raw_method_scores.csv"
    om_f = ROOT / "outputs" / "import_handling_check" / \
        f"{folder}_orphaned_dependant_cases.csv"
    if not all(f.exists() for f in (tiers_f, fu_f, ms_f)):
        return None

    os.environ["KGC_PROJECT"] = folder
    import importlib
    import run_tgc_pilot
    importlib.reload(run_tgc_pilot)

    blob = pickle.load(open(tiers_f, "rb"))
    fu = pickle.load(open(fu_f, "rb"))
    mp = pd.read_csv(ms_f, usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om = set(pd.read_csv(om_f)["commit"].astype(str)) if om_f.exists() else set()

    fidx = np.asarray(fu["commit_index"], int)
    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    fids = [imap.get(int(i), "") for i in fidx]

    X, _, ids_all = run_tgc_pilot.build_features(blob, FROZEN_TIERS)
    pos = {c: k for k, c in enumerate(ids_all)}
    if not all(c in pos for c in fids):
        return None
    Xs = X[np.array([pos[c] for c in fids])]

    cols = [dep, Xs]
    if USE_M1:
        from run_kg_topology import snapshot_features
        order_ids = [c["id"] for c in blob["commits"]]
        T = snapshot_features(blob, order_ids)
        tpos = {cid: k for k, cid in enumerate(order_ids)}
        cols.append(T["M1_position"][np.array([tpos[c] for c in fids])])

    W = max(int(len(y) * 0.2), 50)
    ev = np.arange(W, len(y))
    p_new = stack(cols, y, W)
    return dict(project=folder, y=y[ev], base=dep[ev], new=p_new,
                omega=np.array([fids[i] in om for i in ev]))


def pooled_metric(recs, key, omega_only=False):
    ys, ps = [], []
    for r in recs:
        m = r["omega"] if omega_only else np.ones(len(r["y"]), bool)
        if m.sum() == 0:
            continue
        ys.append(r["y"][m])
        ps.append(r[key][m])
    if not ys:
        return None
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    if len(np.unique(y)) < 2:
        return None
    return final_metrics(y, np.clip(p, 0, 1))


def bootstrap_commits(recs, metric="Macro_F1", omega_only=False, n=2000, seed=0):
    """Resample COMMITS within the pooled sample -- maximises power."""
    rng = np.random.default_rng(seed)
    ys, pa, pb = [], [], []
    for r in recs:
        m = r["omega"] if omega_only else np.ones(len(r["y"]), bool)
        if m.sum() == 0:
            continue
        ys.append(r["y"][m]); pa.append(r["new"][m]); pb.append(r["base"][m])
    if not ys:
        return None
    y = np.concatenate(ys); a = np.concatenate(pa); b = np.concatenate(pb)
    d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        ma = final_metrics(y[s], np.clip(a[s], 0, 1)).get(metric)
        mb = final_metrics(y[s], np.clip(b[s], 0, 1)).get(metric)
        if ma is not None and mb is not None:
            d.append(float(ma) - float(mb))
    if len(d) < 50:
        return None
    d = np.asarray(d)
    return {"n_commits": int(len(y)), "mean_diff": float(d.mean()),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def bootstrap_projects(recs, metric="Macro_F1", omega_only=False, n=2000, seed=0):
    """Resample PROJECTS -- tests generalisation to unseen projects."""
    rng = np.random.default_rng(seed)
    per = []
    for r in recs:
        m = r["omega"] if omega_only else np.ones(len(r["y"]), bool)
        if m.sum() < 10 or len(np.unique(r["y"][m])) < 2:
            continue
        a = final_metrics(r["y"][m], np.clip(r["new"][m], 0, 1)).get(metric)
        b = final_metrics(r["y"][m], np.clip(r["base"][m], 0, 1)).get(metric)
        if a is not None and b is not None:
            per.append(float(a) - float(b))
    if len(per) < 4:
        return None
    per = np.asarray(per)
    d = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    d = np.asarray(d)
    return {"n_projects": int(len(per)), "mean_diff": float(per.mean()),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "wins": int((per > 0).sum()), "losses": int((per < 0).sum())}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    recs, rows = [], []
    print("FROZEN DESIGN:", FROZEN_TIERS, f"[{FROZEN_PENALTY}, C={FROZEN_C}]",
          "+M1" if USE_M1 else "", flush=True)
    for disp, folder in PROJECTS:
        try:
            r = score_project(folder)
        except Exception as e:
            print(f"  {disp:<11} SKIP ({type(e).__name__}: {e})", flush=True)
            continue
        if r is None:
            print(f"  {disp:<11} SKIP (missing inputs)", flush=True)
            continue
        recs.append(r)
        mb = final_metrics(r["y"], np.clip(r["base"], 0, 1))
        mn = final_metrics(r["y"], np.clip(r["new"], 0, 1))
        om = r["omega"]
        ob = on = None
        if om.sum() >= 10 and len(np.unique(r["y"][om])) > 1:
            ob = final_metrics(r["y"][om], np.clip(r["base"][om], 0, 1))
            on = final_metrics(r["y"][om], np.clip(r["new"][om], 0, 1))
        rows.append([disp, len(r["y"]), int(om.sum()),
                     round(float(mb["Macro_F1"]), 4), round(float(mn["Macro_F1"]), 4),
                     round(float(mn["Macro_F1"] - mb["Macro_F1"]), 4),
                     round(float(ob["Macro_F1"]), 4) if ob else "",
                     round(float(on["Macro_F1"]), 4) if on else "",
                     round(float(on["Macro_F1"] - ob["Macro_F1"]), 4) if ob else ""])
        print(f"  {disp:<11} n={len(r['y']):>5} om={int(om.sum()):>4}  "
              f"ALL {mb['Macro_F1']:.4f}->{mn['Macro_F1']:.4f} "
              f"({mn['Macro_F1']-mb['Macro_F1']:+.4f})"
              + (f"   OM {ob['Macro_F1']:.4f}->{on['Macro_F1']:.4f} "
                 f"({on['Macro_F1']-ob['Macro_F1']:+.4f})" if ob else ""), flush=True)

    with (DEST / "pooled_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "n_eval", "n_omega", "base_ALL", "new_ALL", "d_ALL",
                    "base_OM", "new_OM", "d_OM"])
        w.writerows(rows)

    print("\n" + "=" * 66, flush=True)
    out = {"frozen_design": {"tiers": FROZEN_TIERS, "penalty": FROZEN_PENALTY,
                             "C": FROZEN_C, "M1": USE_M1},
           "n_projects": len(recs)}
    for label, om in (("ALL commits", False), ("OMEGA commits", True)):
        pm_b = pooled_metric(recs, "base", om)
        pm_n = pooled_metric(recs, "new", om)
        bc = bootstrap_commits(recs, "Macro_F1", om)
        bp = bootstrap_projects(recs, "Macro_F1", om)
        print(f"\n{label}")
        if pm_b and pm_n:
            print(f"  pooled Macro-F1  {pm_b['Macro_F1']:.4f} -> {pm_n['Macro_F1']:.4f} "
                  f"({pm_n['Macro_F1']-pm_b['Macro_F1']:+.4f})")
            print(f"  pooled MCC       {pm_b['MCC']:.4f} -> {pm_n['MCC']:.4f}")
        if bc:
            sig = "SIGNIFICANT" if bc["p"] < 0.05 else "n.s."
            print(f"  commit bootstrap n={bc['n_commits']}  d={bc['mean_diff']:+.4f} "
                  f"CI[{bc['ci_lo']:+.4f},{bc['ci_hi']:+.4f}] p={bc['p']:.4f}  {sig}")
        if bp:
            sig = "SIGNIFICANT" if bp["p"] < 0.05 else "n.s."
            print(f"  project bootstrap k={bp['n_projects']} d={bp['mean_diff']:+.4f} "
                  f"p={bp['p']:.4f} ({bp['wins']}W/{bp['losses']}L)  {sig}")
        out[label] = {"pooled_base": pm_b and {k: float(v) for k, v in pm_b.items()},
                      "pooled_new": pm_n and {k: float(v) for k, v in pm_n.items()},
                      "commit_bootstrap": bc, "project_bootstrap": bp}
    (DEST / "pooled_significance.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {DEST}", flush=True)


if __name__ == "__main__":
    main()
