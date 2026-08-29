"""
Rescore the RQ1 baselines onto KG-Commit's deployed protocol.
=============================================================

RQ1 compares KG-Commit against the baselines, but the two were not scored the
same way in `final_final_run`:

                       span              verification gap
    KG-Commit          [ev0, N)          G = 50      (build_final_final_run,
                                                      final_metrics(..., gap=P.GAP))
    baselines          [W,   N)          G = 0       (run_final_experiments imports
                                                      GAP but never passes it, and
                                                      final_metrics defaults gap=0)

So the baselines were evaluated on `init` extra commits and without the
verification latency every KG-Commit number pays. The caption's claim that every
model is "scored on identical commits under one strictly chronological protocol"
was therefore not true of the gap or the span.

This script rescores each baseline from its persisted per-commit predictions
onto KG-Commit's exact span and gap, and writes the corrected metrics alongside
the originals. DeepJIT is deliberately NOT touched: it comes from its own
grid (`outputs/DeepJIT_baseline_results/grid_summary.csv`), not from this
pickle, and is left exactly as published.

Note the direction of the correction: matching the protocol *lowers* the
baselines, because they lose the easy early commits and gain the gap. The
change works against KG-Commit's favour in no case, which is precisely why it
should be stated in the paper rather than applied silently.

Cache-only: reads `baseline_extra_results.pkl` (`raws` holds each baseline's
per-commit `pred`, `y` and `idx`) and the fusion pickle's span metadata. No
Neo4j, no graph, no refit -- the stored predictions are simply re-measured.

Run:  python inference/rq1_rescore_baselines.py           # report only
      python inference/rq1_rescore_baselines.py --apply   # write the JSON
Out:  outputs/tables/rq1_baselines_matched.json
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
DEST = OUTP / "tables" / "rq1_baselines_matched.json"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
KEYS = ("Macro_F1", "G_Mean", "AUC")


def one_project(p):
    import numpy as np
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent))
    import _kgc_paths  # noqa: F401
    import protocol as P
    from online_jit import final_metrics

    ffr = OUTP / p / "final_final_run"
    B = pickle.load(open(ffr / "baselines" / "baseline_extra_results.pkl", "rb"))
    F = pickle.load(open(ffr / "fusion" / "final_fusion_results.pkl", "rb"))

    W, ev0, N = F["meta"]["W"], F["meta"]["ev0"], F["meta"]["N"]
    off = ev0 - W                     # baselines start at W; KG starts at ev0

    out = {"meta": {"W": W, "ev0": ev0, "N": N, "gap": int(P.GAP),
                    "n_eval_published": int(B["n_eval"]),
                    "n_eval_matched": int(N - ev0)},
           "baselines": {}}

    for name, d in B["raws"].items():
        y = np.asarray(d["y"], int)
        pr = np.asarray(d["pred"], float)
        if len(y) <= off:
            continue
        y2, p2 = y[off:], pr[off:]     # restrict to KG-Commit's span
        mm = final_metrics(y2, p2, gap=P.GAP)
        pub = B["baselines"][name]
        out["baselines"][name] = {
            "published": {k: float(pub[k]) for k in KEYS},
            "matched": {k: float(mm[k]) for k in KEYS},
            "n_matched": int(len(y2)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the JSON; without it, report only")
    ap.add_argument("project", nargs="?")
    a = ap.parse_args()

    if a.project:
        print(json.dumps({a.project: one_project(a.project)}))
        return

    allout = {}
    for p in PROJECTS:
        env = dict(os.environ, KGC_PROJECT=p)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), p],
                           capture_output=True, text=True, env=env, cwd=str(HERE))
        line = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
        if not line.startswith("{"):
            print(f"  {p}: FAILED\n{r.stderr[-400:]}")
            continue
        allout.update(json.loads(line))

    print(f"{'project':11}{'baseline':20}{'published':>10}{'matched':>10}{'delta':>9}")
    for p in PROJECTS:
        if p not in allout:
            continue
        for n, d in allout[p]["baselines"].items():
            a_, b_ = d["published"]["Macro_F1"], d["matched"]["Macro_F1"]
            print(f"{p:11}{n:20}{a_:10.4f}{b_:10.4f}{b_-a_:+9.4f}")

    if a.apply:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps(allout, indent=1), encoding="utf-8")
        print(f"\nwrote {DEST.relative_to(ROOT)}  ({len(allout)} projects)")
    else:
        print("\n(report only -- pass --apply to write the JSON)")


if __name__ == "__main__":
    main()
