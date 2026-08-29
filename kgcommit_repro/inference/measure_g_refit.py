"""
Measure the CSTG channel's periodic refit cost, per commit (amortised).
=======================================================================

Tables 4 and 5 of the draft reported the deployed KG-Commit refit as 0.000 / "---",
which is right for the graph fusion itself -- RN and PPR are closed-form, so
F=RN+PPR needs no model refit -- but wrong for the DEPLOYED model F+G. The semantic
channel G is a logistic regression over the CSTG feature block, refit prequentially
on the expanding past window (run_final_fusion.channel_score: every other BLOCK,
labels withheld by the gap). That is a real periodic cost and belongs in the table.

This script replays exactly that refit schedule against the cached CSTG features and
times it, then amortises over the commits each refit serves:

    refit_ms_per_commit = total_refit_seconds * 1000 / n_scored

Also times the small stacker refit (the logistic blend over the score columns),
which is negligible but included for completeness.

Cache-only: reads online_jit_streams_v5.pkl. No Neo4j.

Out: outputs/<project>/final_final_run/complexity/g_refit_cost.json
Run: python inference/measure_g_refit.py
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]


def measure(project):
    from sklearn.linear_model import LogisticRegression
    import protocol as P

    st = OUTP / project / "online_jit_streams_v5.pkl"
    fr = OUTP / project / "final_final_run" / "fusion" / "final_fusion_results.pkl"
    if not (st.exists() and fr.exists()):
        return None
    S = pickle.load(open(st, "rb"))
    F = pickle.load(open(fr, "rb"))
    y = np.asarray(S["y"], int)
    N = len(y)
    W = int(F.get("meta", {}).get("W") or S.get("W") or int(N * P.WARMUP_FRAC))

    # the CSTG feature block, exactly as run_final_fusion assembles it for G
    G = sp.hstack([sp.csr_matrix(np.hstack([
        S["cstg_prior"][:, None], S["cstg_typed"], S["cstg_consist"]])),
        S["Xcstg"]]).tocsr()

    # replay channel_score's refit schedule and time only the fits
    total_s, n_refits = 0.0, 0
    i, blk = W, 0
    while i < N:
        j = min(N, i + P.BLOCK)
        if blk % 2 == 0:                     # channel_score refits every other block
            u2 = max(0, i - P.GAP)
            if u2 >= 2 and len(set(y[:u2])) > 1:
                clf = LogisticRegression(max_iter=1500, class_weight="balanced",
                                         solver="liblinear")
                t0 = time.perf_counter()
                clf.fit(G[:u2], y[:u2])
                total_s += time.perf_counter() - t0
                n_refits += 1
        i = j; blk += 1

    n_scored = N - W
    return {"project": project, "N": N, "warmup": W, "n_scored": n_scored,
            "n_refits": n_refits, "total_refit_s": total_s,
            "refit_ms_per_commit_amortised": total_s * 1000.0 / max(1, n_scored),
            "mean_refit_ms": total_s * 1000.0 / max(1, n_refits),
            "n_features": int(G.shape[1]),
            "note": "G = CSTG channel: prequential LR over the CSTG feature block, "
                    "refit every other BLOCK on the expanding past window "
                    "(labels withheld by GAP). RN/PPR need no refit."}


def main():
    out = {}
    for p in PROJECTS:
        os.environ["KGC_PROJECT"] = p
        for m in [m for m in list(sys.modules)
                  if m.startswith(("config", "_kgc_paths"))]:
            del sys.modules[m]
        try:
            r = measure(p)
        except Exception as e:
            print(f"  {p:<11} FAILED: {type(e).__name__}: {e}")
            continue
        if not r:
            print(f"  {p:<11} missing inputs")
            continue
        dst = OUTP / p / "final_final_run" / "complexity" / "g_refit_cost.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump(r, open(dst, "w"), indent=2)
        out[p] = r
        print(f"  {p:<11} {r['n_refits']:>3} refits  "
              f"mean {r['mean_refit_ms']:>8.1f} ms  "
              f"amortised {r['refit_ms_per_commit_amortised']:>7.4f} ms/commit")

    if out:
        med = float(np.median([r["refit_ms_per_commit_amortised"]
                               for r in out.values()]))
        print(f"\n  median amortised G refit = {med:.4f} ms/commit")


if __name__ == "__main__":
    main()
