"""
Seed robustness for the deployed fusion (light, proof-of-stability only).
=========================================================================

WHY
---
Every headline number in the paper comes from a single run. A reviewer will ask
whether it is stable. Most of the pipeline is deterministic -- RN, PPR and LP are
closed-form over the graph and return the same scores every time -- so a full
multi-seed re-run would be almost entirely wasted compute.

The stochastic parts are exactly two:
  * the fusion head (LogisticRegression / SGD initialisation and solver path), and
  * the DW / KGE embeddings (random walks, negative sampling, SGD init).

This script re-evaluates ONLY the deployed fusion under protocol.SEEDS and reports
mean +/- sd per metric, which is enough to state "results are stable to within
+/- x" without pretending to a full replication study.

NEEDS THE GRAPH: re-deriving DW/KGE under each seed calls run_final_experiments.
run_graph, which reads Neo4j. Run it while THIS project's graph is resident, in the
same slot as the other graph-bound steps.

Out: outputs/<project>/final_run/seeds/seed_robustness.json (+ .csv)
Run: KGC_PROJECT=zookeeper python inference/run_seed_robustness.py
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402
from protocol import SEEDS, OVERALL_F  # noqa: E402

REPORT = ["PR_AUC", "ROC_AUC", "Buggy_F1", "Macro_F1", "G_Mean", "MCC"]


def _load_fusion():
    f = OUT / "final_fusion_results.pkl"
    if not f.exists():
        sys.exit(f"missing {f} -- run run_final_fusion.py first")
    return pickle.load(open(f, "rb"))


def main():
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler
    import run_final_fusion as rff
    import run_final_experiments as rfe

    res = _load_fusion()
    outdir = OUT / "final_run" / "seeds"
    outdir.mkdir(parents=True, exist_ok=True)

    # the two selection rules; evaluate both so either can headline the paper
    rules = {"per_project": res.get("F"), "overall": res.get("F_overall")}

    commits, cids, y, files, devs, tok, cstg = rfe.load_all()
    N = len(cids); W = int(N * rfe.WARMUP_FRAC)
    S = pickle.load(open(OUT / "online_jit_streams_v5.pkl", "rb"))

    # The G channel is deterministic given the streams; build it once.
    Gfeat = sp.hstack([sp.csr_matrix(np.hstack([
        S["cstg_prior"][:, None], S["cstg_typed"], S["cstg_consist"]])),
        S["Xcstg"]]).tocsr()

    all_rows = []
    summary = {}
    per_seed_scores = {}
    for sd in SEEDS:
        # Re-derive the five graph-native method scores under this seed. RN/PPR/LP
        # are deterministic; DW/KGE are not, which is the point of the experiment.
        np.random.seed(sd)
        _, preds = rfe.run_graph("final", cids, y, files, devs, tok, cstg)
        sc = {m: preds[m] for m in rff.METHODS}
        sc["G"] = rff.channel_score(Gfeat, y, W, N, sparse=True)
        per_seed_scores[sd] = sc

    for rule, F in rules.items():
        if not F:
            continue
        per_seed = {m: [] for m in REPORT}
        for sd in SEEDS:
            m, _, _, _ = rff.eval_subset(per_seed_scores[sd], list(F) + ["G"],
                                         y, W, N)
            for k in REPORT:
                if k in m:
                    per_seed[k].append(float(m[k]))
            all_rows.append({"rule": rule, "seed": sd,
                             **{k: float(m.get(k, float("nan"))) for k in REPORT}})
            print(f"  [{rule}] seed={sd} " +
                  " ".join(f"{k}={m.get(k, float('nan')):.4f}" for k in REPORT))

        summary[rule] = {
            "F": list(F), "n_seeds": len(SEEDS),
            **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                   "min": float(np.min(v)), "max": float(np.max(v))}
               for k, v in per_seed.items() if v},
        }

    json.dump({"project": PROJECT, "seeds": list(SEEDS), "summary": summary},
              open(outdir / "seed_robustness.json", "w"), indent=2)
    if all_rows:
        with open(outdir / "seed_robustness.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)

    print(f"\nsaved -> {outdir}/seed_robustness.{{json,csv}}")
    for rule, s in summary.items():
        mf = s.get("Macro_F1", {})
        if mf:
            print(f"  {rule:<12} Macro-F1 = {mf['mean']:.4f} +/- {mf['sd']:.4f}")


if __name__ == "__main__":
    main()
