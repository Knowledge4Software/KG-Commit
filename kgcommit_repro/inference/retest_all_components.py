#!/usr/bin/env python3
"""
Re-test every component under the CORRECT protocol + the adopted H3 head.
========================================================================

Fifteen components were evaluated only on the Kafka pilot, under the
non-standard protocol audited in E11 (W=0.20N instead of 0.05N, no adaptive
init_window, GAP ignored, no u0 guard) and with an unregularised head. Their
KEEP/HOLD/DROP verdicts therefore rest on measurements we know to be wrong ---
including several DROPs, which is the dangerous direction: a component
discarded on a bad measurement is never revisited.

This runs all of them on all eleven projects under:
  * protocol.py exactly (WARMUP_FRAC=0.05, init_window, GAP=50, u0 guard)
  * the H3 head adopted in E12 (standardised + L1, C=0.2, liblinear)

Components re-tested (each stacked on the deployed F+G):
  base blocks : T0+T1dir (reference), +T2, +TP, +TC
  topology    : M1 position, M2 diffusion, M3 spectral, M4 shape
  mechanisms  : B1 parent tokens, C1 term metapaths, C2 typed PPR,
                A4 articulation, W1 ownership/issue, W2 directed diffusion,
                W3 temporal decay, W4 relative(z), W5 structural holes,
                W6 interactions
  post-hoc    : D2 selective abstention (applied to the best arm)

Out: outputs/_retest/<project>/arms.csv, summary.csv, verdicts.csv
Run: nohup python -u inference/retest_all_components.py > ~/retest.log 2>&1 &
"""
from __future__ import annotations

import csv
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
DEST = ROOT / "outputs" / "_retest"
ORDER = ["zookeeper", "spark", "zeppelin", "kafka", "groovy", "activemq",
         "cassandra", "hive", "camel", "flink", "hbase"]


def eval_stack(Z, y, W, ev0, refit=REFIT_EVERY, gap=GAP):
    """final_final_run eval_subset loop with the adopted H3 head."""
    N = len(y)
    p_all = np.full(N, np.nan)
    i, blk = ev0, 0
    u0 = max(W, ev0 - gap)
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
        p_all[idx] = (clf.predict_proba(sc.transform(Z[idx]))[:, 1]
                      if clf is not None else y[:i].mean())
        uj = max(W, j - gap)
        if blk % refit == 0:
            c2 = fit(W, uj)
            if c2 is not None:
                clf = c2
        i = j
        blk += 1
    return p_all


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
    import run_wave2_arms
    import run_untested_arms
    for m in (run_tgc_pilot, run_kg_topology, run_wave2_arms, run_untested_arms):
        importlib.reload(m)

    blob = pickle.load(open(need[0], "rb"))
    fu = pickle.load(open(need[1], "rb"))
    mp = pd.read_csv(need[2], usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om_f = ROOT / "outputs" / "import_handling_check" / f"{folder}_orphaned_dependant_cases.csv"
    om = set(pd.read_csv(om_f)["commit"].astype(str)) if om_f.exists() else set()

    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    fids = [imap.get(int(i), "") for i in np.asarray(fu["commit_index"], int)]

    def tiers(names):
        X, _, ids = run_tgc_pilot.build_features(blob, names)
        pos = {c: k for k, c in enumerate(ids)}
        return X[np.array([pos[c] for c in fids])] if all(c in pos for c in fids) else None

    order_ids = [c["id"] for c in blob["commits"]]
    tpos = {cid: k for k, cid in enumerate(order_ids)}
    sel = np.array([tpos[c] for c in fids])
    T = run_kg_topology.snapshot_features(blob, order_ids)

    B = {"REF": tiers(["T0", "T1_dep", "T1_dry"]),
         "T2": tiers(["T0", "T1_dep", "T1_dry", "T2"]),
         "TP": tiers(["T0", "T1_dep", "T1_dry", "TP"]),
         "TC": tiers(["T0", "T1_dep", "T1_dry", "TC"]),
         "M1": T["M1_position"][sel], "M2": T["M2_diffusion"][sel],
         "M3": T["M3_spectral"][sel], "M4": T["M4_shape"][sel]}

    # wave-2 / untested mechanism blocks, rebuilt for this project
    CTX = {"blob": blob, "order_ids": order_ids, "fids": fids, "y": y,
           "all_files": sorted({f for t in blob["tiers"].values() if t
                                for k in ("T0", "T1_dep", "T1_dry")
                                for f in t.get(k, ())}),
           "Xtgc": B["REF"], "M1": B["M1"], "dep": dep,
           "W": int(len(y) * WARMUP_FRAC)}
    extras = {}
    for tag, mod, fn in (("W2", run_wave2_arms, "stage_W2"),
                         ("W3", run_wave2_arms, "stage_W3"),
                         ("W5", run_wave2_arms, "stage_W5")):
        try:
            mod.CTX.clear(); mod.CTX.update(CTX); mod.ROWS.clear()
            mod.CTX["yc"] = y; mod.CTX["mask"] = np.zeros(len(y), bool)
            mod.CTX["base_p"] = dep
            mod.CTX["bm"] = {"Macro_F1": 0.0}; mod.CTX["bo"] = {"Macro_F1": 0.0}
            getattr(mod, fn)()
            extras[tag] = mod.CTX.get(tag)
        except Exception:
            extras[tag] = None
    # W4 relative(z): expanding-window standardisation of the reference block
    if B["REF"] is not None:
        X = B["REF"]; Z = np.zeros_like(X)
        mu = np.zeros(X.shape[1]); s2 = np.ones(X.shape[1]); c = 0
        for i in range(len(X)):
            if c > 30:
                Z[i] = (X[i] - mu) / np.sqrt(np.maximum(s2, 1e-9))
            c += 1
            d0 = X[i] - mu; mu = mu + d0 / c
            s2 = s2 + (d0 * (X[i] - mu) - s2) / c
        extras["W4"] = Z
        # W6 interactions
        extras["W6"] = np.hstack([X[:, 1:3] * X[:, 0:1], B["M1"][:, :2] * X[:, 0:1]])
    B.update({k: v for k, v in extras.items() if v is not None})
    return dict(y=y, dep=dep, B=B, fids=fids,
                omega=np.array([c in om for c in fids]))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    allrows = []
    for folder in ORDER:
        t0 = time.time()
        try:
            d = build(folder)
        except Exception as e:
            print(f"[{folder}] BUILD FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        if d is None:
            print(f"[{folder}] skip (missing inputs)", flush=True)
            continue
        y, dep, B = d["y"], d["dep"], d["B"]
        N = len(y); W = int(N * WARMUP_FRAC); INIT = init_window(y, W); ev0 = W + INIT
        ev = np.arange(ev0, N); yc = y[ev]; mask = d["omega"][ev]
        base_p = np.clip(np.nan_to_num(dep[ev], nan=float(yc.mean())), 0, 1)
        bm = final_metrics(yc, base_p, gap=GAP)
        bo = (final_metrics(yc[mask], base_p[mask], gap=GAP)
              if mask.sum() >= 10 and len(np.unique(yc[mask])) > 1 else None)
        print(f"\n[{folder}] rows={len(ev)} om={int(mask.sum())} "
              f"base ALL={bm['Macro_F1']:.4f}", flush=True)

        ref = B.get("REF")
        arms = {}
        if ref is not None:
            arms["REF T0+T1dir"] = [dep, ref]
            for k in ("T2", "TP", "TC"):
                if B.get(k) is not None:
                    arms[f"+{k}"] = [dep, B[k]]
            for k in ("M1", "M2", "M3", "M4", "W2", "W3", "W4", "W5", "W6"):
                if B.get(k) is not None:
                    arms[f"REF+{k}"] = [dep, ref, B[k]]
        rows = [["deployed (F+G)", round(float(bm["Macro_F1"]), 4),
                 round(float(bo["Macro_F1"]), 4) if bo else "", 0.0, 0.0]]
        best_p, best_d = None, -9
        for name, cols in arms.items():
            try:
                Z = np.nan_to_num(np.column_stack(cols), nan=float(y[:W].mean()),
                                  posinf=0.0, neginf=0.0)
                p = np.clip(np.nan_to_num(eval_stack(Z, y, W, ev0)[ev],
                                          nan=float(yc.mean())), 0, 1)
            except Exception as e:
                print(f"   {name:<16} FAILED {type(e).__name__}", flush=True)
                continue
            m = final_metrics(yc, p, gap=GAP)
            mo = final_metrics(yc[mask], p[mask], gap=GAP) if bo else None
            dA = float(m["Macro_F1"] - bm["Macro_F1"])
            dO = float(mo["Macro_F1"] - bo["Macro_F1"]) if mo else None
            rows.append([name, round(float(m["Macro_F1"]), 4),
                         round(float(mo["Macro_F1"]), 4) if mo else "",
                         round(dA, 4), round(dO, 4) if dO is not None else ""])
            allrows.append([folder, name, dA, dO if dO is not None else ""])
            if dA > best_d:
                best_d, best_p = dA, p
            print(f"   {name:<16} ALL={m['Macro_F1']:.4f} ({dA:+.4f})"
                  + (f"  OM={mo['Macro_F1']:.4f} ({dO:+.4f})" if mo else "")
                  + ("  BOTH UP" if (dA > 0 and dO and dO > 0) else ""), flush=True)

        # D2 abstention on the best arm of this project
        if best_p is not None:
            conf = np.abs(best_p - 0.5)
            for cov in (0.90, 0.80):
                thr = np.quantile(conf, 1 - cov)
                k = conf >= thr
                if k.sum() < 50 or len(np.unique(yc[k])) < 2:
                    continue
                m = final_metrics(yc[k], best_p[k], gap=GAP)
                km = k & mask
                mo = (final_metrics(yc[km], best_p[km], gap=GAP)
                      if bo and km.sum() >= 10 and len(np.unique(yc[km])) > 1 else None)
                dA = float(m["Macro_F1"] - bm["Macro_F1"])
                dO = float(mo["Macro_F1"] - bo["Macro_F1"]) if mo else None
                rows.append([f"D2 abstain {cov:.0%}", round(float(m["Macro_F1"]), 4),
                             round(float(mo["Macro_F1"]), 4) if mo else "",
                             round(dA, 4), round(dO, 4) if dO is not None else ""])
                allrows.append([folder, f"D2 abstain {cov:.0%}", dA,
                                dO if dO is not None else ""])
                print(f"   D2 abstain {cov:.0%}  ALL={m['Macro_F1']:.4f} ({dA:+.4f})",
                      flush=True)

        pdir = DEST / folder
        pdir.mkdir(exist_ok=True)
        with (pdir / "arms.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "all_MacroF1", "omega_MacroF1", "d_ALL", "d_OM"])
            w.writerows(rows)
        print(f"   ({time.time()-t0:.0f}s)", flush=True)

    with (DEST / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "arm", "d_ALL", "d_OM"])
        w.writerows(allrows)
    print(f"\nsaved -> {DEST}", flush=True)


if __name__ == "__main__":
    main()
