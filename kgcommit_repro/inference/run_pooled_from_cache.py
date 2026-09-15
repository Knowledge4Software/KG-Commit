#!/usr/bin/env python3
"""
Pooled evaluation from the Stage-2 cached score vectors.
========================================================

run_pooled_eval.py recomputed every project from scratch and silently dropped
the four largest (Camel, Flink, HBase, Hive) -- almost certainly OOM on a 23 GB
box during snapshot_features. Those four hold ~75% of all Omega commits, so the
pooled Omega test ran on n=135 instead of the ~500 available.

Stage 2 (run_all_projects.py) already computed and PERSISTED the per-commit
score vectors for every arm on all eleven projects. This script pools from
those caches: no recomputation, no Neo4j, seconds instead of hours, and every
project included.

Failures are reported explicitly -- nothing is silently skipped.

Out: outputs/_pooled/pooled_cache_<arm>.json  + pooled_cache_summary.csv
Run: python inference/run_pooled_from_cache.py [--arm "TGCdirTP+M1"]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import _kgc_paths  # noqa: E402,F401
from online_jit import final_metrics  # noqa: E402

ROOT = HERE.parent.parent
SRC = ROOT / "outputs" / "_allproj"
DEST = ROOT / "outputs" / "_pooled"


def load_all():
    recs, missing = [], []
    for d in sorted(SRC.iterdir()):
        f = d / "scores.pkl"
        if not f.is_dir() and f.exists():
            try:
                recs.append(pickle.load(open(f, "rb")))
            except Exception as e:
                missing.append((d.name, f"unreadable: {e}"))
        elif d.is_dir():
            missing.append((d.name, "no scores.pkl"))
    return recs, missing


def pooled(recs, arm, omega_only=False):
    ys, pa, pb = [], [], []
    used, skipped = [], []
    for r in recs:
        if arm not in r:
            skipped.append((r["project"], "arm absent"))
            continue
        m = r["omega"].astype(bool) if omega_only else np.ones(len(r["y"]), bool)
        if m.sum() == 0:
            skipped.append((r["project"], "no omega commits"))
            continue
        if len(np.unique(r["y"][m])) < 2:
            skipped.append((r["project"], "single class"))
            continue
        ys.append(r["y"][m]); pa.append(np.asarray(r[arm])[m]); pb.append(r["base"][m])
        used.append(r["project"])
    if not ys:
        return None
    return (np.concatenate(ys), np.concatenate(pa), np.concatenate(pb),
            used, skipped)


def boot_commits(y, a, b, metric="Macro_F1", n=2000, seed=0):
    rng = np.random.default_rng(seed)
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
    return {"n": int(len(y)), "diff": float(d.mean()),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def boot_projects(recs, arm, omega_only=False, n=2000, seed=0):
    per, names = [], []
    for r in recs:
        if arm not in r:
            continue
        m = r["omega"].astype(bool) if omega_only else np.ones(len(r["y"]), bool)
        if m.sum() < 10 or len(np.unique(r["y"][m])) < 2:
            continue
        a = final_metrics(r["y"][m], np.clip(np.asarray(r[arm])[m], 0, 1)).get("Macro_F1")
        b = final_metrics(r["y"][m], np.clip(r["base"][m], 0, 1)).get("Macro_F1")
        if a is None or b is None:
            continue
        per.append(float(a) - float(b)); names.append(r["project"])
    if len(per) < 4:
        return None
    per = np.asarray(per)
    rng = np.random.default_rng(seed)
    d = np.asarray([per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)])
    return {"k": len(per), "diff": float(per.mean()),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "wins": int((per > 0).sum()), "losses": int((per < 0).sum()),
            "per_project": dict(zip(names, [round(float(x), 4) for x in per]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="evaluate one arm only")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    recs, missing = load_all()
    print(f"projects loaded: {len(recs)}  -> {[r['project'] for r in recs]}", flush=True)
    for p, why in missing:
        print(f"  !! MISSING {p}: {why}", flush=True)

    arms = ([args.arm] if args.arm else
            sorted({k for r in recs for k in r
                    if k not in ("project", "y", "base", "omega", "fids")}))
    rows = []
    for arm in arms:
        out = {"arm": arm}
        print(f"\n{'='*66}\n{arm}", flush=True)
        for label, om in (("ALL", False), ("OMEGA", True)):
            got = pooled(recs, arm, om)
            if got is None:
                print(f"  {label}: no data", flush=True)
                continue
            y, a, b, used, skipped = got
            mb = final_metrics(y, np.clip(b, 0, 1))
            ma = final_metrics(y, np.clip(a, 0, 1))
            bc = boot_commits(y, a, b)
            bp = boot_projects(recs, arm, om)
            print(f"  {label}: n={len(y)} from {len(used)} projects", flush=True)
            if skipped:
                print(f"    excluded: {[(p, w) for p, w in skipped]}", flush=True)
            print(f"    Macro-F1 {mb['Macro_F1']:.4f} -> {ma['Macro_F1']:.4f} "
                  f"({ma['Macro_F1']-mb['Macro_F1']:+.4f})   "
                  f"MCC {mb['MCC']:.4f} -> {ma['MCC']:.4f}", flush=True)
            if bc:
                print(f"    commit bootstrap  d={bc['diff']:+.4f} "
                      f"CI[{bc['lo']:+.4f},{bc['hi']:+.4f}] p={bc['p']:.4f}  "
                      f"{'SIGNIFICANT' if bc['p']<0.05 else 'n.s.'}", flush=True)
            if bp:
                print(f"    project bootstrap k={bp['k']} d={bp['diff']:+.4f} "
                      f"p={bp['p']:.4f} ({bp['wins']}W/{bp['losses']}L)  "
                      f"{'SIGNIFICANT' if bp['p']<0.05 else 'n.s.'}", flush=True)
            out[label] = {"n": int(len(y)), "projects": used,
                          "base_MacroF1": float(mb["Macro_F1"]),
                          "new_MacroF1": float(ma["Macro_F1"]),
                          "base_MCC": float(mb["MCC"]), "new_MCC": float(ma["MCC"]),
                          "commit_bootstrap": bc, "project_bootstrap": bp}
            rows.append([arm, label, len(y), len(used),
                         round(float(mb["Macro_F1"]), 4), round(float(ma["Macro_F1"]), 4),
                         round(float(ma["Macro_F1"] - mb["Macro_F1"]), 4),
                         round(bc["p"], 4) if bc else "",
                         round(bp["p"], 4) if bp else "",
                         f"{bp['wins']}W/{bp['losses']}L" if bp else ""])
        safe = arm.replace("/", "_").replace(" ", "_").replace("+", "p")
        (DEST / f"pooled_cache_{safe}.json").write_text(json.dumps(out, indent=2, default=str))

    with (DEST / "pooled_cache_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "population", "n_commits", "n_projects",
                    "base_MacroF1", "new_MacroF1", "delta",
                    "p_commit_bootstrap", "p_project_bootstrap", "wins_losses"])
        w.writerows(rows)
    print(f"\nsaved -> {DEST/'pooled_cache_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
