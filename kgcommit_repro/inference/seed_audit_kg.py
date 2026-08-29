"""
Cache-only audit of KG-Commit's seed sensitivity.
=================================================

`run_seed_robustness.py` reports sd = 0.000000 on every project. This script
establishes -- without Neo4j -- what that number does and does not show.

Three questions, answered from cached artifacts:

  Q1  Does the seed reach the stochastic components at all?
      Measured directly on the two RNG call sites.

  Q2  With the DW/KGE scores held at their cached values, does anything
      DOWNSTREAM of them introduce seed variance? (the prequential stacker,
      the CSTG channel G, and the threshold tuner)
      Re-runs the full deployed fusion under every protocol seed, from
      raw_method_scores.pkl. Cache-only.

  Q3  What WOULD vary if the seed were plumbed through? Bounded from below by
      re-fitting the stacker on perturbed embedding channels, which is the only
      part of the DW/KGE path reachable without re-deriving the embeddings.

Out: outputs/tables/seed_audit_kg.json
Run: python inference/seed_audit_kg.py
"""
import json
import os
# The config package binds one project at import time; these are
# cross-project drivers that read per-project files directly, so pin a
# placeholder to satisfy the import without constraining what we read.
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]


# ---------------------------------------------------------------- Q1 ------
def q1_rng_reachability():
    """Do the two stochastic call sites respond to np.random.seed()?"""
    from sklearn.utils.extmath import randomized_svd
    import scipy.sparse as sp

    M = sp.random(300, 120, density=0.08, random_state=7, format="csr")
    svd, rng_draw = [], []
    for sd in P.SEEDS:
        np.random.seed(sd)                       # what run_seed_robustness sets
        U, S, _ = randomized_svd(M, n_components=8, random_state=0)
        svd.append(U[:4, :3].copy())
        np.random.seed(sd)
        rng_draw.append(np.random.default_rng(0).standard_normal(5))

    return {
        "dw_randomized_svd_identical_across_seeds":
            bool(all(np.allclose(svd[0], s) for s in svd[1:])),
        "kge_default_rng_identical_across_seeds":
            bool(all(np.allclose(rng_draw[0], g) for g in rng_draw[1:])),
        "explanation":
            "dw_embed calls randomized_svd(..., random_state=0) and kge_embed "
            "calls np.random.default_rng(seed) with its default seed=0. Neither "
            "consults the legacy global RNG that np.random.seed() sets, so the "
            "seed loop never changed the embeddings.",
    }


# ---------------------------------------------------------------- Q2 ------
def q2_downstream(project):
    """Holding DW/KGE at their cached values, is the rest seed-invariant?"""
    import run_final_fusion as rff

    ffr = OUTP / project / "final_final_run"
    R = pickle.load(open(ffr / "experiments" / "raw_method_scores.pkl", "rb"))
    F = pickle.load(open(ffr / "fusion" / "final_fusion_results.pkl", "rb"))
    y = np.asarray(R["y"], int)
    N, W = len(y), int(R["warmup"])
    init = P.init_window(y, W)
    subset = [m for m in F["chosen_overall"].split("+") if m]

    vals = []
    for sd in P.SEEDS:
        np.random.seed(sd)
        sc = {m: np.asarray(R["scores"]["final"][m], float) for m in METHODS}
        m, _, _, _ = rff.eval_subset(sc, subset, y, W, N, init=init,
                                     refit=P.REFIT_EVERY, gap=P.GAP)
        vals.append(float(m["Macro_F1"]))
    return {"seeds": list(P.SEEDS), "macro_f1": vals,
            "sd": float(np.std(vals)), "spread": float(max(vals) - min(vals))}


# ---------------------------------------------------------------- Q3 ------
def q3_embedding_sensitivity(project, rel=0.02, reps=5):
    """Lower bound on the variance the seed WOULD induce.

    We cannot re-derive DW/KGE without the graph, so we perturb their cached
    per-commit scores by a small multiplicative noise and re-run the deployed
    fusion. This is a proxy, not a replication: it shows how much the fusion's
    output moves when the embedding channels move slightly, which is the
    mechanism by which a real seed change would propagate.
    """
    import run_final_fusion as rff

    ffr = OUTP / project / "final_final_run"
    R = pickle.load(open(ffr / "experiments" / "raw_method_scores.pkl", "rb"))
    F = pickle.load(open(ffr / "fusion" / "final_fusion_results.pkl", "rb"))
    y = np.asarray(R["y"], int)
    N, W = len(y), int(R["warmup"])
    init = P.init_window(y, W)
    subset = [m for m in F["chosen_overall"].split("+") if m]

    base = {m: np.asarray(R["scores"]["final"][m], float) for m in METHODS}
    if not ({"DW", "KGE"} & set(subset)):
        return {"note": ("deployed fusion is %s -- it contains neither DW nor "
                         "KGE, so embedding noise cannot reach it"
                         % F["chosen_overall"]),
                "sd": 0.0, "deployed": F["chosen_overall"]}

    vals = []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        sc = dict(base)
        for m in ("DW", "KGE"):
            if m in sc:
                sc[m] = np.clip(sc[m] * (1 + rel * rng.standard_normal(len(y))),
                                0, 1)
        mm, _, _, _ = rff.eval_subset(sc, subset, y, W, N, init=init,
                                      refit=P.REFIT_EVERY, gap=P.GAP)
        vals.append(float(mm["Macro_F1"]))
    return {"macro_f1": vals, "sd": float(np.std(vals)),
            "deployed": F["chosen_overall"], "rel_noise": rel}


def main():
    out = {"Q1_rng_reachability": q1_rng_reachability(),
           "Q2_downstream_invariance": {}, "Q3_embedding_sensitivity": {}}
    for p in PROJECTS:
        try:
            out["Q2_downstream_invariance"][p] = q2_downstream(p)
        except Exception as e:
            out["Q2_downstream_invariance"][p] = {"error": str(e)[:120]}
        try:
            out["Q3_embedding_sensitivity"][p] = q3_embedding_sensitivity(p)
        except Exception as e:
            out["Q3_embedding_sensitivity"][p] = {"error": str(e)[:120]}
        print(f"  {p}: done")

    dst = OUTP / "tables" / "seed_audit_kg.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
