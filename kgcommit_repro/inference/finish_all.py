#!/usr/bin/env python3
"""
Recovering ALL while keeping the Omega gain: head + gating variants.
====================================================================

Under the correct final_final_run protocol the global-context arms give a
significant Omega gain (TGCdirTP: 9/10 projects, +0.130, Wilcoxon p=0.0039) but
LOSE on ALL commits (mean -0.023). The loss is present in EVERY arm, including
"M1 only" (2 features), so it is not the feature block -- it is the head.

Diagnosis: run_final_fusion's _lr() is a plain unregularised lbfgs LR. That is
correct for stacking 2-6 method scores, which is what it was built for. Here it
stacks 26-44 dense context features on streams whose evaluated window now
starts at 0.05*N -- so the early blocks fit a wide model on very little data.
E3 already established on Kafka that L1 fixes exactly this (Omega MCC
0.108 -> 0.404 on identical features).

Variants tested (protocol otherwise IDENTICAL to run_final_fusion.eval_subset):

  H1  L1 head            sparsity instead of unregularised lbfgs
  H2  L2 head, C=0.1     shrinkage without sparsity
  H3  standardised L1    scaler fitted on the past only
  H4  omega-gated        deployed score everywhere EXCEPT commits with real
                         cross-file context (|T1_dep|>0), where the global
                         model is used. Directly encodes the paper's claim:
                         global context is applied where global context exists.
  H5  confidence-gated   global model only where it is confident, else deployed

Out: outputs/_finish/<project>/arms.csv + summary.csv
Run: nohup python -u inference/finish_all.py > ~/finish.log 2>&1 &
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
DEST = ROOT / "outputs" / "_finish"
ORDER = ["zookeeper", "spark", "zeppelin", "kafka", "groovy", "activemq",
         "cassandra", "hive", "camel", "flink", "hbase"]


def head(kind):
    if kind == "l1":
        return LogisticRegression(max_iter=1500, class_weight="balanced",
                                  penalty="l1", C=0.2, solver="liblinear")
    if kind == "l2c":
        return LogisticRegression(max_iter=1500, class_weight="balanced",
                                  C=0.1, solver="lbfgs")
    return LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")


def eval_stack(Z, y, W, ev0, kind="plain", scale=False, refit=REFIT_EVERY, gap=GAP):
    """run_final_fusion.eval_subset stacking loop, with a pluggable head."""
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
        Zt = Z[a:b]
        if scale:
            sc = StandardScaler().fit(Zt)
            Zt = sc.transform(Zt)
        return head(kind).fit(Zt, y[a:b])

    clf = fit(W, u0)
    while i < N:
        j = min(N, i + BLOCK)
        idx = np.arange(i, j)
        if clf is not None:
            Zi = sc.transform(Z[idx]) if (scale and sc is not None) else Z[idx]
            p_all[idx] = clf.predict_proba(Zi)[:, 1]
        else:
            p_all[idx] = y[:i].mean()
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
    t_f, fu_f, ms_f = (base / "global_context" / "tiers.pkl",
                       base / "raw_fusion_scores.pkl", base / "raw_method_scores.csv")
    if not all(f.exists() for f in (t_f, fu_f, ms_f)):
        return None
    os.environ["KGC_PROJECT"] = folder
    import importlib
    import run_tgc_pilot
    import run_kg_topology
    for m in (run_tgc_pilot, run_kg_topology):
        importlib.reload(m)

    blob = pickle.load(open(t_f, "rb"))
    fu = pickle.load(open(fu_f, "rb"))
    mp = pd.read_csv(ms_f, usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om_f = ROOT / "outputs" / "import_handling_check" / f"{folder}_orphaned_dependant_cases.csv"
    om = set(pd.read_csv(om_f)["commit"].astype(str)) if om_f.exists() else set()

    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    fids = [imap.get(int(i), "") for i in np.asarray(fu["commit_index"], int)]
    X, _, ids = run_tgc_pilot.build_features(blob, ["T0", "T1_dep", "T1_dry", "TP"])
    pos = {c: k for k, c in enumerate(ids)}
    if not all(c in pos for c in fids):
        return None
    sel = np.array([pos[c] for c in fids])
    Xt = X[sel]
    # has this commit any REAL cross-file context? (untouched dependants)
    tiers = blob["tiers"]
    has_ctx = np.array([bool(tiers.get(c) and len(tiers[c].get("T1_dep", ())) > 0)
                        for c in fids])
    return dict(y=y, dep=dep, X=Xt, fids=fids, has_ctx=has_ctx,
                omega=np.array([c in om for c in fids]))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    allrows = []
    for folder in ORDER:
        t0 = time.time()
        try:
            d = build(folder)
        except Exception as e:
            print(f"[{folder}] BUILD FAILED {e}", flush=True)
            traceback.print_exc()
            continue
        if d is None:
            print(f"[{folder}] skip", flush=True)
            continue
        y, dep, X = d["y"], d["dep"], d["X"]
        N = len(y)
        W = int(N * WARMUP_FRAC)
        INIT = init_window(y, W)
        ev0 = W + INIT
        ev = np.arange(ev0, N)
        yc, mask = y[ev], d["omega"][ev]
        ctx = d["has_ctx"][ev]
        base_p = np.clip(np.nan_to_num(dep[ev], nan=float(yc.mean())), 0, 1)
        bm = final_metrics(yc, base_p, gap=GAP)
        bo = (final_metrics(yc[mask], base_p[mask], gap=GAP)
              if mask.sum() >= 10 and len(np.unique(yc[mask])) > 1 else None)
        print(f"\n[{folder}] rows={len(ev)} om={int(mask.sum())} ctx={ctx.mean():.0%} "
              f"base ALL={bm['Macro_F1']:.4f}" + (f" OM={bo['Macro_F1']:.4f}" if bo else ""),
              flush=True)

        Z = np.nan_to_num(np.column_stack([dep, X]), nan=float(y[:W].mean()),
                          posinf=0.0, neginf=0.0)
        variants = {}
        for tag, kw in (("H1 L1", dict(kind="l1")),
                        ("H2 L2c", dict(kind="l2c")),
                        ("H3 stdL1", dict(kind="l1", scale=True))):
            variants[tag] = np.clip(np.nan_to_num(
                eval_stack(Z, y, W, ev0, **kw)[ev], nan=float(yc.mean())), 0, 1)

        # H4: gate on REAL cross-file context -- the paper's claim, operationalised
        best_global = variants["H3 stdL1"]
        g = base_p.copy()
        g[ctx] = best_global[ctx]
        variants["H4 ctx-gated"] = g

        # H5: gate on confidence of the global model
        conf = np.abs(best_global - 0.5)
        thr = np.quantile(conf, 0.5)
        g2 = base_p.copy()
        sel2 = conf >= thr
        g2[sel2] = best_global[sel2]
        variants["H5 conf-gated"] = g2

        rows = [["deployed (F+G)", round(float(bm["Macro_F1"]), 4),
                 round(float(bm["MCC"]), 4),
                 round(float(bo["Macro_F1"]), 4) if bo else "",
                 round(float(bo["MCC"]), 4) if bo else "", 0.0, 0.0]]
        for tag, p in variants.items():
            m = final_metrics(yc, p, gap=GAP)
            mo = final_metrics(yc[mask], p[mask], gap=GAP) if bo else None
            dA = float(m["Macro_F1"] - bm["Macro_F1"])
            dO = float(mo["Macro_F1"] - bo["Macro_F1"]) if mo else None
            rows.append([tag, round(float(m["Macro_F1"]), 4), round(float(m["MCC"]), 4),
                         round(float(mo["Macro_F1"]), 4) if mo else "",
                         round(float(mo["MCC"]), 4) if mo else "",
                         round(dA, 4), round(dO, 4) if dO is not None else ""])
            allrows.append([folder, tag, dA, dO if dO is not None else ""])
            print(f"   {tag:<15} ALL={m['Macro_F1']:.4f} ({dA:+.4f})"
                  + (f"  OM={mo['Macro_F1']:.4f} ({dO:+.4f})" if mo else "")
                  + ("  BOTH UP" if (dA > 0 and dO and dO > 0) else ""), flush=True)

        pdir = DEST / folder
        pdir.mkdir(exist_ok=True)
        with (pdir / "arms.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "all_MacroF1", "all_MCC", "omega_MacroF1",
                        "omega_MCC", "d_ALL", "d_OM"])
            w.writerows(rows)
        pickle.dump({"project": folder, "y": yc, "base": base_p, "omega": mask,
                     "ctx": ctx, **variants}, open(pdir / "scores.pkl", "wb"))
        print(f"   ({time.time()-t0:.0f}s)", flush=True)

    with (DEST / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "arm", "d_ALL", "d_OM"])
        w.writerows(allrows)
    print(f"\nsaved -> {DEST}", flush=True)


if __name__ == "__main__":
    main()
