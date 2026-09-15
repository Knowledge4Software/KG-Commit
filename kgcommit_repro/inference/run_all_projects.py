#!/usr/bin/env python3
"""
Full multi-project run --- protected, chained, with early-drop and persistence.
==============================================================================

Runs the whole arm suite over every project whose dump is available, in
ascending order of graph size (cheapest first, so a failure late costs least).

TWO POLICIES, as instructed:

  1. EARLY DROP. After each project the accumulated record is re-scored. An arm
     that has been evaluated on >= MIN_PROJECTS_TO_DROP projects and has NEVER
     improved ALL Macro-F1 on any of them is dropped: it is skipped on all
     remaining projects. Dropped arms and the evidence are written to
     dropped_arms.json, so the decision is auditable and reversible.

  2. PERSIST EVERYTHING. Per project we save not only the metric rows but the
     raw per-commit score vectors of every surviving arm, so future work
     (plots, new metrics, effort-aware analysis, abstention curves,
     re-pooling) needs no re-run.

Resumable: a project whose done-marker exists is skipped, so the chain can be
re-launched after an interruption without losing work.

Out: outputs/_allproj/
       <project>/arms.csv          metric row per arm
       <project>/scores.pkl        y, base, per-arm score vectors, omega mask
       dropped_arms.json           early-drop decisions + evidence
       progress.json               per-project status
Run: nohup python -u inference/run_all_projects.py > ~/allproj.log 2>&1 &
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
from protocol import BLOCK  # noqa: E402
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = HERE.parent.parent
DEST = ROOT / "outputs" / "_allproj"
MIN_PROJECTS_TO_DROP = 3          # never drop on fewer than this many projects

# cheapest first: a late failure then costs the least
ORDER = ["zookeeper", "spark", "zeppelin", "kafka", "groovy", "activemq",
         "cassandra", "hive", "camel", "flink", "hbase"]


def stack(cols, y, W, C=0.2, penalty="l1"):
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
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C,
                                 penalty=penalty,
                                 solver="liblinear" if penalty == "l1" else "lbfgs")
        clf.fit(sc.transform(Z[past]), y[past])
        p[i:j] = clf.predict_proba(sc.transform(Z[i:j]))[:, 1]
        i = j
    return np.nan_to_num(p[np.arange(W, N)], nan=float(y[np.arange(W, N)].mean()))


def build_blocks(folder):
    """Every feature block for one project, aligned to the deployed rows."""
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
    import run_tgc_v2
    import run_wave2_arms
    for m in (run_tgc_pilot, run_kg_topology, run_tgc_v2, run_wave2_arms):
        importlib.reload(m)

    blob = pickle.load(open(tiers_f, "rb"))
    fu = pickle.load(open(fu_f, "rb"))
    mp = pd.read_csv(ms_f, usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om_f = ROOT / "outputs" / "import_handling_check" / \
        f"{folder}_orphaned_dependant_cases.csv"
    om = set(pd.read_csv(om_f)["commit"].astype(str)) if om_f.exists() else set()

    fidx = np.asarray(fu["commit_index"], int)
    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    F_only = np.asarray(fu["scores"]["F"], float)
    fids = [imap.get(int(i), "") for i in fidx]

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

    B = {
        "TGCdir": tiers(["T0", "T1_dep", "T1_dry"]),
        "TGCdirTP": tiers(["T0", "T1_dep", "T1_dry", "TP"]),
        "TGCfull": tiers(["T0", "T1_dep", "T1_dry", "TP", "TC"]),
        "M1": T["M1_position"][sel],
        "M2": T["M2_diffusion"][sel],
        "M4": T["M4_shape"][sel],
    }
    return dict(y=y, dep=dep, F=F_only, fids=fids,
                omega=np.array([c in om for c in fids]), blocks=B, blob=blob)


def arm_specs(B, dep, F):
    """The arms carried forward. Kept deliberately broad -- early-drop prunes."""
    A = {}
    A["deployed (F+G)"] = None                      # baseline
    if B["TGCdir"] is not None:
        A["TGCdir"] = [dep, B["TGCdir"]]
        A["TGCdir+M1"] = [dep, B["TGCdir"], B["M1"]]
        A["TGCdir+M1+M2"] = [dep, B["TGCdir"], B["M1"], B["M2"]]
        A["TGCdir+M1+M4"] = [dep, B["TGCdir"], B["M1"], B["M4"]]
        A["F+TGCdir+M1 (drop G)"] = [F, B["TGCdir"], B["M1"]]
    if B["TGCdirTP"] is not None:
        A["TGCdirTP"] = [dep, B["TGCdirTP"]]
        A["TGCdirTP+M1"] = [dep, B["TGCdirTP"], B["M1"]]
    if B["TGCfull"] is not None:
        A["TGCfull"] = [dep, B["TGCfull"]]
        A["TGCfull+M1"] = [dep, B["TGCfull"], B["M1"]]
    A["M1 only"] = [dep, B["M1"]]
    A["M1+M2+M4"] = [dep, B["M1"], B["M2"], B["M4"]]
    return A


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    drop_f = DEST / "dropped_arms.json"
    prog_f = DEST / "progress.json"
    dropped = json.loads(drop_f.read_text()) if drop_f.exists() else {}
    progress = json.loads(prog_f.read_text()) if prog_f.exists() else {}
    history = {}                        # arm -> list of (project, dALL)

    # rebuild history from any completed projects (resume support)
    for pdir in sorted(DEST.glob("*/arms.csv")):
        proj = pdir.parent.name
        for r in csv.DictReader(open(pdir)):
            if r["arm"] == "deployed (F+G)":
                continue
            try:
                history.setdefault(r["arm"], []).append((proj, float(r["d_ALL"])))
            except (ValueError, KeyError):
                pass

    disp_of = {f: d for d, f in PROJECTS}
    for folder in ORDER:
        if folder in progress and progress[folder].get("status") == "done":
            print(f"[{folder}] already done, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"\n[{folder}] {'='*50}", flush=True)
        try:
            data = build_blocks(folder)
        except Exception as e:
            print(f"  BUILD FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            progress[folder] = {"status": "failed", "error": str(e)}
            prog_f.write_text(json.dumps(progress, indent=2))
            continue
        if data is None:
            print("  SKIP: missing tiers.pkl or score caches", flush=True)
            progress[folder] = {"status": "skipped", "reason": "missing inputs"}
            prog_f.write_text(json.dumps(progress, indent=2))
            continue

        y, dep, F = data["y"], data["dep"], data["F"]
        W = max(int(len(y) * 0.2), 50)
        ev = np.arange(W, len(y))
        yc = y[ev]
        mask = data["omega"][ev]
        base_p = dep[ev]
        bm = final_metrics(yc, np.clip(base_p, 0, 1))
        bo = (final_metrics(yc[mask], np.clip(base_p[mask], 0, 1))
              if mask.sum() >= 10 and len(np.unique(yc[mask])) > 1 else None)
        print(f"  rows={len(ev)} omega={int(mask.sum())}  "
              f"baseline ALL={bm['Macro_F1']:.4f}"
              + (f" OM={bo['Macro_F1']:.4f}" if bo else " OM=n/a"), flush=True)

        specs = arm_specs(data["blocks"], dep, F)
        rows, scores = [], {"y": yc, "base": base_p, "omega": mask}
        for name, cols in specs.items():
            if name in dropped:
                continue
            try:
                p = base_p if cols is None else stack(cols, y, W)
            except Exception as e:
                print(f"    {name:<26} FAILED ({type(e).__name__})", flush=True)
                continue
            m = final_metrics(yc, np.clip(p, 0, 1))
            mo = (final_metrics(yc[mask], np.clip(p[mask], 0, 1))
                  if bo else None)
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
            flag = "  BOTH UP" if (dA > 0 and dO is not None and dO > 0) else ""
            print(f"    {name:<26} ALL={m['Macro_F1']:.4f} ({dA:+.4f})"
                  + (f"  OM={mo['Macro_F1']:.4f} ({dO:+.4f})" if mo else "")
                  + flag, flush=True)

        pd_ = DEST / folder
        pd_.mkdir(exist_ok=True)
        with (pd_ / "arms.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "all_MacroF1", "all_MCC", "all_AUC",
                        "omega_MacroF1", "omega_MCC", "d_ALL", "d_OM", "n_omega"])
            w.writerows(rows)
        # PERSIST raw per-commit vectors for future plots / metrics / pooling
        pickle.dump({"project": folder, "fids": [data["fids"][i] for i in ev],
                     **scores}, open(pd_ / "scores.pkl", "wb"))

        # ---- EARLY DROP -------------------------------------------------
        for arm, hist in history.items():
            if arm in dropped or len(hist) < MIN_PROJECTS_TO_DROP:
                continue
            if all(d <= 0 for _, d in hist):
                dropped[arm] = {"dropped_after": folder,
                                "n_projects": len(hist),
                                "d_ALL_history": {p: round(d, 4) for p, d in hist},
                                "reason": "never improved ALL Macro-F1"}
                print(f"  >> DROPPING '{arm}': no ALL gain on "
                      f"{len(hist)} projects", flush=True)
        drop_f.write_text(json.dumps(dropped, indent=2))
        progress[folder] = {"status": "done", "n_arms": len(rows),
                            "seconds": round(time.time() - t0, 1),
                            "n_eval": int(len(ev)), "n_omega": int(mask.sum())}
        prog_f.write_text(json.dumps(progress, indent=2))
        print(f"  [{folder}] done in {time.time()-t0:.0f}s, "
              f"{len(rows)} arms, {len(dropped)} dropped so far", flush=True)

    print(f"\nALL PROJECTS COMPLETE -> {DEST}", flush=True)
    print(f"dropped arms: {sorted(dropped)}", flush=True)


if __name__ == "__main__":
    main()
