"""
Effort-aware evaluation of the KG models for the active project.
================================================================

JIT-defect-prediction is usually judged not only by classification quality but by
INSPECTION EFFORT: if a reviewer can only read a fraction of the changed lines,
how many bugs do they catch? This script reports the two standard effort-aware
metrics for the KG method's predictions:

  * Popt        normalised effort-vs-bugs curve area (Kamei et al.); 0.5 random
  * ACC@20%LOC  recall of buggy commits after inspecting commits (risk-ranked)
                until 20% of the total changed-lines effort is spent

for the DEPLOYED fusion (F+G) and each of the individual feature channels, using
la+ld as the per-commit effort. Runs entirely from the cached feature streams
(online_jit_streams_v5.pkl) + the label CSV -- NO Neo4j needed, so it can run
while another project's graph is building.

Out: outputs/<project>/effort_results.pkl  + printed table
Run: KGC_PROJECT=zookeeper python inference/run_effort_eval.py
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, CSV_PATH, PROJECT  # noqa: E402
import online_jit as OJ
import effort_metrics as em

JIT_COLS = ["la", "ld"]


def load_effort():
    """Per-commit inspection effort (la+ld), chronologically aligned to the
    stream (which is ordered by author_date, same as the CSV sort)."""
    df = pd.read_csv(CSV_PATH).sort_values("author_date").reset_index(drop=True)
    return (df["la"].fillna(0) + df["ld"].fillna(0)).to_numpy(float) + 1.0


def main():
    cache = OUT / "online_jit_streams_v5.pkl"
    if not cache.exists():
        raise SystemExit(f"no cached streams at {cache}; run experiments first "
                         f"(precompute_streams) for {PROJECT}.")
    S = pickle.load(open(cache, "rb"))
    N, W = S["N"], S["W"]
    effort = load_effort()
    if len(effort) != N:
        # streams and CSV should align; guard against off-by-project mismatch
        print(f"  !! effort length {len(effort)} != stream N {N}; truncating to min")
        m = min(len(effort), N); effort = effort[:m]

    ev = np.arange(W, N)
    y_ev = S["y"][ev]
    eff_ev = effort[ev]

    # the deployed model + each channel (same subsets online_jit uses)
    subsets = {
        "Fusion(F+G)": dict(metrics=True, tfidf=True, priors=True, ppr=True, cstg=True),
        "M_metrics":   dict(metrics=True),
        "T_tfidf":     dict(tfidf=True),
        "R_priors":    dict(priors=True),
        "P_ppr":       dict(ppr=True),
        "G_cstg":      dict(cstg=True),
    }
    results = {}
    print(f"[{PROJECT}] effort-aware eval over {len(ev)} commits "
          f"(bug rate {y_ev.mean():.3f})\n")
    hdr = f"{'model':<14}{'Popt':>8}{'ACC@20':>9}{'Buggy_F1':>9}{'PR_AUC':>8}"
    print(hdr); print("-" * len(hdr))
    for name, mask in subsets.items():
        r = OJ.run_subset(S, block=OJ.BLOCK, **mask)
        p = np.clip(np.nan_to_num(r["p"], nan=float(y_ev.mean())), 0, 1)
        popt = float(em.popt(y_ev, p, eff_ev))
        acc20 = float(em.recall_at_effort(y_ev, p, eff_ev, 0.20))
        cm = r["cum"]
        results[name] = dict(Popt=popt, ACC20=acc20,
                             Buggy_F1=float(cm["Buggy_F1"]),
                             PR_AUC=float(cm["PR_AUC"]),
                             Macro_F1=float(cm["Macro_F1"]),
                             G_Mean=float(cm["G_Mean"]), AUC=float(cm["AUC"]))
        print(f"{name:<14}{popt:8.3f}{acc20:9.3f}{cm['Buggy_F1']:9.3f}{cm['PR_AUC']:8.3f}")

    out = dict(project=PROJECT, N=N, warmup=W, n_eval=len(ev),
               bug_rate=float(y_ev.mean()), effort_metric="la+ld",
               models=results)
    pickle.dump(out, open(OUT / "effort_results.pkl", "wb"))
    print(f"\nsaved -> {OUT / 'effort_results.pkl'}")


if __name__ == "__main__":
    main()
