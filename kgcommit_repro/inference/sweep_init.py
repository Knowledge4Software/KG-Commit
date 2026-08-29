"""
Sensitivity of the deployed model to the stacking-head initialisation rule.
==========================================================================

INIT is the one deployed constant with no published sweep. It is the window of
labelled commits the fusion head is allowed after the warm-up before it may
score anything, and it is chosen ADAPTIVELY (protocol.init_window): advance from
INIT_FLOOR in steps of INIT_STEP until the window holds at least
INIT_MIN_MINORITY examples of the rarer class, capped at INIT_CAP.

Two questions a reviewer will ask, answered here from cache:

  A. Is the adaptive rule better than the flat INIT it replaced?
     -> compare adaptive against a grid of fixed values.
  B. Is the result sensitive to INIT_MIN_MINORITY, the only free parameter of
     the rule?
     -> sweep it and report the spread.

Cache-only: replays the deployed fusion from raw_method_scores.pkl and the
cached CSTG block. No Neo4j.

Out: outputs/tables/sweep_init.json
Run: python inference/sweep_init.py
"""
import json
import os
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402
import run_final_fusion as rff  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]

FIXED = [50, 100, 150, 200, 300, 400]      # flat alternatives
MINORITY = [5, 10, 15, 20, 30]             # the rule's own free parameter


def load(project):
    ffr = OUTP / project / "final_final_run"
    R = pickle.load(open(ffr / "experiments" / "raw_method_scores.pkl", "rb"))
    F = pickle.load(open(ffr / "fusion" / "final_fusion_results.pkl", "rb"))
    st = OUTP / project / "online_jit_streams_v5.pkl"
    S = pickle.load(open(st, "rb")) if st.exists() else None
    return R, F, S


_GCACHE = {}


def score(R, F, S, y, W, N, init, project=None):
    """Deployed fusion F+G at a given init.

    The CSTG channel G does not depend on init, so it is computed once per
    project and cached; recomputing it per sweep point was the entire cost.
    """
    sc = {m: np.asarray(R["scores"]["final"][m], float)
          for m in ("RN", "PPR", "LP", "DW", "KGE")}
    if S is not None:
        if project not in _GCACHE:
            G = sp.hstack([sp.csr_matrix(np.hstack([
                S["cstg_prior"][:, None], S["cstg_typed"], S["cstg_consist"]])),
                S["Xcstg"]]).tocsr()
            _GCACHE[project] = rff.channel_score(G, y, W, N, sparse=True)
        sc["G"] = _GCACHE[project]
    subset = [m for m in F["chosen_overall"].split("+") if m]
    if "G" in sc:
        subset = subset + ["G"]
    m, _, _, _ = rff.eval_subset(sc, subset, y, W, N, init=init,
                                 refit=P.REFIT_EVERY, gap=P.GAP)
    return float(m["Macro_F1"])


def main():
    out = {}
    for p in PROJECTS:
        try:
            R, F, S = load(p)
        except Exception as e:
            print(f"  {p}: skipped ({str(e)[:60]})")
            continue
        y = np.asarray(R["y"], int)
        N, W = len(y), int(R["warmup"])

        rec = {"W": W, "N": N,
               "adaptive_init": int(P.init_window(y, W)),
               "fixed": {}, "minority": {}}
        rec["adaptive_score"] = score(R, F, S, y, W, N, rec["adaptive_init"], p)
        for v in FIXED:
            if W + v < N:
                rec["fixed"][str(v)] = score(R, F, S, y, W, N, v, p)
        for mm in MINORITY:
            iv = int(P.init_window(y, W, min_minority=mm))
            rec["minority"][str(mm)] = {"init": iv,
                                        "score": score(R, F, S, y, W, N, iv, p)}
        out[p] = rec
        fx = rec["fixed"]
        best = max(fx, key=fx.get) if fx else "-"
        print(f"  {p:11} adaptive init={rec['adaptive_init']:3} "
              f"MF1={rec['adaptive_score']:.4f} | best fixed={best} "
              f"({fx.get(best, float('nan')):.4f})")

    dst = OUTP / "tables" / "sweep_init.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dst.relative_to(ROOT)}  ({len(out)} projects)")


if __name__ == "__main__":
    main()
