#!/usr/bin/env python3
"""
Multi-project global-context evaluation, under the EXACT final_final_run protocol.
=================================================================================

An audit found that every earlier script in this investigation invented its own
evaluation protocol instead of using the project's. The differences were not
cosmetic:

  quantity        earlier scripts        final_final_run (protocol.py)
  --------------  ---------------------  ---------------------------------------
  warm-up W       0.20 * N               WARMUP_FRAC = 0.05 * N
  stack start     max(0.2*N, 50)         W + init_window(y, W)   [ADAPTIVE]
  gap G           0 (ignored)            GAP = 50, in fit AND threshold tuning
  refit cadence   every block            REFIT_EVERY = 1 block, u0 fix applied
  head            plain LR               _lr(): max_iter=1500, balanced, lbfgs

The adaptive init_window exists precisely because a flat window "quietly
defeated K = 0.05" and produced degenerate fits on small projects -- Spark is
named in its docstring. Our hardcoded 0.2*N burned 243 extra commits on Spark
(293 vs 50) and is the likely cause of its -0.187 collapse, which was wrongly
reported as an applicability limit.

This script changes ONLY the inference layer (adding global-context features on
top of the deployed score) and reproduces everything else exactly:
eval_subset's stacking loop, the u0 empty-window fix, GAP-respecting refits, and
gap-aware final_metrics.

Out: outputs/_allproj_protocol/<project>/{arms.csv,scores.pkl}, progress.json
Run: nohup python -u inference/run_allproj_protocol.py > ~/allproj_proto.log 2>&1 &
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
from protocol import (WARMUP_FRAC, BLOCK, GAP, REFIT_EVERY,  # noqa: E402
                      init_window)
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = HERE.parent.parent
DEST = ROOT / "outputs" / "_allproj_protocol"
MIN_PROJECTS_TO_DROP = 3
ORDER = ["zookeeper", "spark", "zeppelin", "kafka", "groovy", "activemq",
         "cassandra", "hive", "camel", "flink", "hbase"]


def _lr():
    """Identical head to run_final_fusion._lr()."""
    return LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")


def eval_stack(Z, y, W, ev0, refit=REFIT_EVERY, gap=GAP):
    """A faithful copy of run_final_fusion.eval_subset's stacking branch.

    Includes the documented u0 fix: when the adaptive init equals the gap the
    initial window Z[W:u0] can collapse to zero rows, leaving clf None and
    making the whole first block a constant -- the exact defect that produced
    Spark's negative score in the original code.
    """
    N = len(y)
    p_all = np.full(N, np.nan)
    i, blk = ev0, 0
    u0 = max(W, ev0 - gap)
    if u0 - W < 2 or len(set(y[W:u0])) < 2:
        u0 = ev0                       # past-only widening, no leakage
    clf = (_lr().fit(Z[W:u0], y[W:u0])
           if u0 - W >= 2 and len(set(y[W:u0])) > 1 else None)
    while i < N:
        j = min(N, i + BLOCK)
        idx = np.arange(i, j)
        p_all[idx] = clf.predict_proba(Z[idx])[:, 1] if clf is not None else y[:i].mean()
        uj = max(W, j - gap)
        if blk % refit == 0 and uj - W >= 2 and len(set(y[W:uj])) > 1:
            clf = _lr().fit(Z[W:uj], y[W:uj])
        i = j
        blk += 1
    return p_all


def build_blocks(folder):
    base = ROOT / "outputs" / folder
    tiers_f = base / "global_context" / "tiers.pkl"
    fu_f = base / "raw_fusion_scores.pkl"
    ms_f = base / "raw_method_scores.csv"
    if not all(f.exists() for f in (tiers_f, fu_f, ms_f)):
        return None

    os.environ["KGC_PROJECT"] = folder
    import importlib
    import run_tgc_pilot
    import run_kg_topology
    for m in (run_tgc_pilot, run_kg_topology):
        importlib.reload(m)

    blob = pickle.load(open(tiers_f, "rb"))
    fu = pickle.load(open(fu_f, "rb"))
    mp = pd.read_csv(ms_f, usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om_f = ROOT / "outputs" / "import_handling_check" / \
        f"{folder}_orphaned_dependant_cases.csv"
    om = set(pd.read_csv(om_f)["commit"].astype(str)) if om_f.exists() else set()

    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    F_only = np.asarray(fu["scores"]["F"], float)
    fids = [imap.get(int(i), "") for i in np.asarray(fu["commit_index"], int)]

    def tiers(names):
        X, _, ids = run_tgc_pilot.build_features(blob, names)
        pos = {c: k for k, c in enumerate(ids)}
        if not all(c in pos for c in fids):
            return None
        return X[np.array([pos[c] for c in fids])]

    order_ids = [c["id"] for c in blob["commits"]]
    tpos = {cid: k for k, cid in enumerate(order_ids)}
    sel = np.array([tpos[c] for c in fids])
    T = run_kg_topology.snapshot_features(blob, order_ids)

    return dict(y=y, dep=dep, F=F_only, fids=fids,
                omega=np.array([c in om for c in fids]),
                B={"TGCdir": tiers(["T0", "T1_dep", "T1_dry"]),
                   "TGCdirTP": tiers(["T0", "T1_dep", "T1_dry", "TP"]),
                   "TGCfull": tiers(["T0", "T1_dep", "T1_dry", "TP", "TC"]),
                   "M1": T["M1_position"][sel], "M2": T["M2_diffusion"][sel],
                   "M4": T["M4_shape"][sel]})


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    prog_f, drop_f = DEST / "progress.json", DEST / "dropped_arms.json"
    progress = json.loads(prog_f.read_text()) if prog_f.exists() else {}
    dropped = json.loads(drop_f.read_text()) if drop_f.exists() else {}
    history = {}

    print(f"PROTOCOL: WARMUP_FRAC={WARMUP_FRAC} BLOCK={BLOCK} GAP={GAP} "
          f"REFIT_EVERY={REFIT_EVERY}, adaptive init_window", flush=True)

    for folder in ORDER:
        if progress.get(folder, {}).get("status") == "done":
            print(f"[{folder}] done already, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"\n[{folder}] {'='*48}", flush=True)
        try:
            d = build_blocks(folder)
        except Exception as e:
            print(f"  BUILD FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            progress[folder] = {"status": "failed", "error": str(e)}
            prog_f.write_text(json.dumps(progress, indent=2))
            continue
        if d is None:
            progress[folder] = {"status": "skipped", "reason": "missing inputs"}
            prog_f.write_text(json.dumps(progress, indent=2))
            print("  SKIP: missing inputs", flush=True)
            continue

        y, dep, F, B = d["y"], d["dep"], d["F"], d["B"]
        N = len(y)
        W = int(N * WARMUP_FRAC)                 # 0.05, not 0.2
        INIT = init_window(y, W)                 # ADAPTIVE
        ev0 = W + INIT
        ev = np.arange(ev0, N)
        yc = y[ev]
        mask = d["omega"][ev]
        base_p = np.clip(np.nan_to_num(dep[ev], nan=float(yc.mean())), 0, 1)
        bm = final_metrics(yc, base_p, gap=GAP)
        bo = (final_metrics(yc[mask], np.clip(dep[ev][mask], 0, 1), gap=GAP)
              if mask.sum() >= 10 and len(np.unique(yc[mask])) > 1 else None)
        print(f"  N={N} W={W} init={INIT} ev0={ev0} rows={len(ev)} "
              f"omega={int(mask.sum())}", flush=True)
        print(f"  baseline ALL={bm['Macro_F1']:.4f}"
              + (f" OM={bo['Macro_F1']:.4f}" if bo else " OM=n/a"), flush=True)

        arms = {"deployed (F+G)": None}
        if B["TGCdir"] is not None:
            arms["TGCdir"] = [dep, B["TGCdir"]]
            arms["TGCdir+M1"] = [dep, B["TGCdir"], B["M1"]]
            arms["TGCdir+M1+M2"] = [dep, B["TGCdir"], B["M1"], B["M2"]]
            arms["F+TGCdir+M1 (drop G)"] = [F, B["TGCdir"], B["M1"]]
        if B["TGCdirTP"] is not None:
            arms["TGCdirTP"] = [dep, B["TGCdirTP"]]
            arms["TGCdirTP+M1"] = [dep, B["TGCdirTP"], B["M1"]]
        if B["TGCfull"] is not None:
            arms["TGCfull+M1"] = [dep, B["TGCfull"], B["M1"]]
        arms["M1 only"] = [dep, B["M1"]]
        arms["M1+M2+M4"] = [dep, B["M1"], B["M2"], B["M4"]]

        rows, scores = [], {"y": yc, "base": base_p, "omega": mask,
                            "fids": [d["fids"][i] for i in ev]}
        for name, cols in arms.items():
            if name in dropped:
                continue
            try:
                if cols is None:
                    p = base_p
                else:
                    Z = np.nan_to_num(np.column_stack(cols), nan=float(y[:W].mean()),
                                      posinf=0.0, neginf=0.0)
                    p = np.clip(np.nan_to_num(eval_stack(Z, y, W, ev0)[ev],
                                              nan=float(yc.mean())), 0, 1)
            except Exception as e:
                print(f"    {name:<26} FAILED ({type(e).__name__}: {e})", flush=True)
                continue
            m = final_metrics(yc, p, gap=GAP)
            mo = final_metrics(yc[mask], p[mask], gap=GAP) if bo else None
            dA = float(m["Macro_F1"] - bm["Macro_F1"])
            dO = float(mo["Macro_F1"] - bo["Macro_F1"]) if mo else None
            rows.append([name, round(float(m["Macro_F1"]), 4), round(float(m["MCC"]), 4),
                         round(float(m["AUC"]), 4),
                         round(float(mo["Macro_F1"]), 4) if mo else "",
                         round(float(mo["MCC"]), 4) if mo else "",
                         round(dA, 4), round(dO, 4) if dO is not None else "",
                         int(mask.sum())])
            scores[name] = p
            if name != "deployed (F+G)":
                history.setdefault(name, []).append((folder, dA))
            print(f"    {name:<26} ALL={m['Macro_F1']:.4f} ({dA:+.4f})"
                  + (f"  OM={mo['Macro_F1']:.4f} ({dO:+.4f})" if mo else "")
                  + ("  BOTH UP" if (dA > 0 and dO and dO > 0) else ""), flush=True)

        pdir = DEST / folder
        pdir.mkdir(exist_ok=True)
        with (pdir / "arms.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "all_MacroF1", "all_MCC", "all_AUC",
                        "omega_MacroF1", "omega_MCC", "d_ALL", "d_OM", "n_omega"])
            w.writerows(rows)
        pickle.dump({"project": folder, **scores}, open(pdir / "scores.pkl", "wb"))

        for arm, hist in history.items():
            if arm not in dropped and len(hist) >= MIN_PROJECTS_TO_DROP \
                    and all(x <= 0 for _, x in hist):
                dropped[arm] = {"dropped_after": folder, "n_projects": len(hist),
                                "d_ALL_history": {p: round(x, 4) for p, x in hist},
                                "reason": "never improved ALL Macro-F1"}
                print(f"  >> DROPPING '{arm}'", flush=True)
        drop_f.write_text(json.dumps(dropped, indent=2))
        progress[folder] = {"status": "done", "W": W, "init": INIT, "ev0": ev0,
                            "n_eval": int(len(ev)), "n_omega": int(mask.sum()),
                            "seconds": round(time.time() - t0, 1)}
        prog_f.write_text(json.dumps(progress, indent=2))
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    print(f"\nCOMPLETE -> {DEST}\ndropped: {sorted(dropped)}", flush=True)


if __name__ == "__main__":
    main()
