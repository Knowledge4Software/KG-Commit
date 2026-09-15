#!/usr/bin/env python3
"""
The assembled final configuration, and a pooled test on correct-protocol scores.
===============================================================================

Two gaps remain after E13:

  1. Every component so far was evaluated INDIVIDUALLY against the reference
     block. The chosen configuration has never been run as one model.
  2. The headline pooled figures still come from the pre-audit protocol run.

Frozen configuration (from the E13 verdicts, fixed before this runs):
    features : T0 + T1_dep + T1_dry + TP, both raw and relative(z)-scored (W4),
               plus M1 architectural position
    head     : standardised L1, C=0.2, liblinear, balanced
    mode     : augment the deployed F+G (never replace)
    protocol : protocol.py exactly -- W=0.05N, adaptive init_window, GAP=50,
               REFIT_EVERY, and the u0 empty-window guard
    post-hoc : D2 abstention at 80% coverage

Out: outputs/_final/{final_config.csv, final_scores.pkl, pooled_correct.json}
Run: nohup python -u inference/run_final_config.py > ~/finalcfg.log 2>&1 &
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
import traceback
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
from protocol import WARMUP_FRAC, BLOCK, GAP, REFIT_EVERY, init_window  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = HERE.parent.parent
DEST = ROOT / "outputs" / "_final"
ORDER = ["zookeeper", "spark", "zeppelin", "kafka", "groovy", "activemq",
         "cassandra", "hive", "camel", "flink", "hbase"]


def eval_stack(Z, y, W, ev0):
    """final_final_run eval_subset loop with the adopted standardised-L1 head."""
    N = len(y)
    p = np.full(N, np.nan)
    i, blk = ev0, 0
    u0 = max(W, ev0 - GAP)
    if u0 - W < 2 or len(set(y[W:u0])) < 2:
        u0 = ev0
    sc = None

    def fit(a, b):
        nonlocal sc
        if b - a < 2 or len(set(y[a:b])) < 2:
            return None
        sc = StandardScaler().fit(Z[a:b])
        return LogisticRegression(max_iter=1500, class_weight="balanced",
                                  penalty="l1", C=0.2,
                                  solver="liblinear").fit(sc.transform(Z[a:b]), y[a:b])

    clf = fit(W, u0)
    while i < N:
        j = min(N, i + BLOCK)
        idx = np.arange(i, j)
        p[idx] = (clf.predict_proba(sc.transform(Z[idx]))[:, 1]
                  if clf is not None else y[:i].mean())
        uj = max(W, j - GAP)
        if blk % REFIT_EVERY == 0:
            c2 = fit(W, uj)
            if c2 is not None:
                clf = c2
        i = j
        blk += 1
    return p


def build(folder):
    base = ROOT / "outputs" / folder
    need = [base / "global_context" / "tiers.pkl", base / "raw_fusion_scores.pkl",
            base / "raw_method_scores.csv"]
    if not all(f.exists() for f in need):
        return None
    os.environ["KGC_PROJECT"] = folder
    import importlib
    import run_tgc_pilot
    import run_kg_topology
    for m in (run_tgc_pilot, run_kg_topology):
        importlib.reload(m)

    blob = pickle.load(open(need[0], "rb"))
    fu = pickle.load(open(need[1], "rb"))
    mp = pd.read_csv(need[2], usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    omf = ROOT / "outputs" / "import_handling_check" / f"{folder}_orphaned_dependant_cases.csv"
    om = set(pd.read_csv(omf)["commit"].astype(str)) if omf.exists() else set()

    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    fids = [imap.get(int(i), "") for i in np.asarray(fu["commit_index"], int)]

    X, _, ids = run_tgc_pilot.build_features(blob, ["T0", "T1_dep", "T1_dry", "TP"])
    pos = {c: k for k, c in enumerate(ids)}
    if not all(c in pos for c in fids):
        return None
    Xt = X[np.array([pos[c] for c in fids])]

    # W4: expanding-window z-scoring, strictly causal
    Z = np.zeros_like(Xt)
    mu = np.zeros(Xt.shape[1])
    s2 = np.ones(Xt.shape[1])
    c = 0
    for i in range(len(Xt)):
        if c > 30:
            Z[i] = (Xt[i] - mu) / np.sqrt(np.maximum(s2, 1e-9))
        c += 1
        d0 = Xt[i] - mu
        mu = mu + d0 / c
        s2 = s2 + (d0 * (Xt[i] - mu) - s2) / c

    oid = [cc["id"] for cc in blob["commits"]]
    tp = {cid: k for k, cid in enumerate(oid)}
    T = run_kg_topology.snapshot_features(blob, oid)
    M1 = T["M1_position"][np.array([tp[c] for c in fids])]

    return dict(y=y, dep=dep, Xt=Xt, Z=Z, M1=M1, fids=fids,
                omega=np.array([c in om for c in fids]))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    rows, recs = [], []
    for folder in ORDER:
        t0 = time.time()
        try:
            d = build(folder)
        except Exception as e:
            print(f"[{folder}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        if d is None:
            print(f"[{folder}] skip (missing inputs)", flush=True)
            continue

        y, dep = d["y"], d["dep"]
        N = len(y)
        W = int(N * WARMUP_FRAC)
        ev0 = W + init_window(y, W)
        ev = np.arange(ev0, N)
        yc, mask = y[ev], d["omega"][ev]
        bp = np.clip(np.nan_to_num(dep[ev], nan=float(yc.mean())), 0, 1)
        bm = final_metrics(yc, bp, gap=GAP)
        bo = (final_metrics(yc[mask], bp[mask], gap=GAP)
              if mask.sum() >= 10 and len(np.unique(yc[mask])) > 1 else None)

        Zc = np.nan_to_num(np.column_stack([dep, d["Xt"], d["Z"], d["M1"]]),
                           nan=float(y[:W].mean()), posinf=0.0, neginf=0.0)
        p = np.clip(np.nan_to_num(eval_stack(Zc, y, W, ev0)[ev],
                                  nan=float(yc.mean())), 0, 1)
        m = final_metrics(yc, p, gap=GAP)
        mo = final_metrics(yc[mask], p[mask], gap=GAP) if bo else None
        dA = float(m["Macro_F1"] - bm["Macro_F1"])
        dO = float(mo["Macro_F1"] - bo["Macro_F1"]) if mo else None

        conf = np.abs(p - 0.5)
        k = conf >= np.quantile(conf, 0.2)
        ma = final_metrics(yc[k], p[k], gap=GAP)
        km = k & mask
        moa = (final_metrics(yc[km], p[km], gap=GAP)
               if bo and km.sum() >= 10 and len(np.unique(yc[km])) > 1 else None)

        rows.append([folder, round(float(bm["Macro_F1"]), 4), round(float(m["Macro_F1"]), 4),
                     round(dA, 4),
                     round(float(bo["Macro_F1"]), 4) if bo else "",
                     round(float(mo["Macro_F1"]), 4) if mo else "",
                     round(dO, 4) if dO is not None else "",
                     round(float(ma["Macro_F1"]), 4),
                     round(float(ma["Macro_F1"] - bm["Macro_F1"]), 4),
                     round(float(moa["Macro_F1"] - bo["Macro_F1"]), 4) if moa else ""])
        recs.append(dict(project=folder, y=yc, base=bp, new=p, omega=mask))
        print(f"[{folder}] ALL {bm['Macro_F1']:.4f}->{m['Macro_F1']:.4f} ({dA:+.4f})"
              + (f"  OM {bo['Macro_F1']:.4f}->{mo['Macro_F1']:.4f} ({dO:+.4f})" if mo else "")
              + f"  | abstain80 ALL={ma['Macro_F1']:.4f}"
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    with (DEST / "final_config.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "base_ALL", "new_ALL", "d_ALL", "base_OM", "new_OM",
                    "d_OM", "abst80_ALL", "d_abst80_ALL", "d_abst80_OM"])
        w.writerows(rows)
    pickle.dump(recs, open(DEST / "final_scores.pkl", "wb"))

    print("\n=== POOLED (correct protocol) ===", flush=True)
    out = {}
    for lab, om in (("ALL", False), ("OMEGA", True)):
        ys, pa, pb = [], [], []
        for r in recs:
            mk = r["omega"] if om else np.ones(len(r["y"]), bool)
            if mk.sum() == 0 or len(np.unique(r["y"][mk])) < 2:
                continue
            ys.append(r["y"][mk]); pa.append(r["new"][mk]); pb.append(r["base"][mk])
        if not ys:
            continue
        y = np.concatenate(ys); a = np.concatenate(pa); b = np.concatenate(pb)
        rng = np.random.default_rng(0)
        dd = []
        for _ in range(2000):
            s = rng.integers(0, len(y), len(y))
            if len(np.unique(y[s])) < 2:
                continue
            x = final_metrics(y[s], np.clip(a[s], 0, 1)).get("Macro_F1")
            z = final_metrics(y[s], np.clip(b[s], 0, 1)).get("Macro_F1")
            if x is not None and z is not None:
                dd.append(float(x) - float(z))
        dd = np.asarray(dd)
        mb = final_metrics(y, np.clip(b, 0, 1))
        mn = final_metrics(y, np.clip(a, 0, 1))
        pv = float(2 * min((dd <= 0).mean(), (dd >= 0).mean()))
        print(f"{lab}: n={len(y)}  {mb['Macro_F1']:.4f} -> {mn['Macro_F1']:.4f} "
              f"({mn['Macro_F1']-mb['Macro_F1']:+.4f})  p={pv:.4f} "
              f"CI[{np.percentile(dd,2.5):+.4f},{np.percentile(dd,97.5):+.4f}]", flush=True)
        out[lab] = {"n": int(len(y)), "base": float(mb["Macro_F1"]),
                    "new": float(mn["Macro_F1"]), "p": pv,
                    "ci_lo": float(np.percentile(dd, 2.5)),
                    "ci_hi": float(np.percentile(dd, 97.5))}
    (DEST / "pooled_correct.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {DEST}", flush=True)


if __name__ == "__main__":
    main()
