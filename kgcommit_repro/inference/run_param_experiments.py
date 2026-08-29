"""
Parameter-sensitivity experiments for the active project.
=========================================================

Reproduces the first-phase (groovy) parameter sweeps per project, to confirm the
tuned choices transfer. Sweeps the online-protocol knobs and re-evaluates the
DEPLOYED fusion (F+G) at each setting:

  * WARMUP_FRAC (K)   0.20 0.30 0.40 0.50   initial-fit fraction
  * BLOCK       (M)   25 50 100 200         predict/learn block granularity
  * ROLL              200 400 800 1600      rolling-metric window (visualisation)

Runs entirely from the cached feature streams (online_jit_streams_v5.pkl) -- NO
Neo4j -- so it is safe to run while another project's graph is building. The
WARMUP sweep re-derives W from the stream; BLOCK re-runs the prequential loop.

Out: outputs/<project>/param_experiments/{K,M,ROLL}.json  + printed tables
Run: KGC_PROJECT=zookeeper python inference/run_param_experiments.py
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402
import online_jit as OJ  # noqa: E402
import pickle

DEPLOYED = dict(metrics=True, tfidf=True, priors=True, ppr=True, cstg=True)
REPORT = ["PR_AUC", "Buggy_F1", "Macro_F1", "G_Mean", "AUC", "MCC"]
PARAM_DIR = OUT / "param_experiments"


def _leaf(S, **over):
    r = OJ.run_subset(S, **DEPLOYED, **over)
    cm = r["cum"]
    return {k: float(cm[k]) for k in REPORT}


def sweep_block(S, values=(25, 50, 100, 200)):
    return {str(v): _leaf(S, block=v) for v in values}


def sweep_warmup(S, values=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50)):
    """Re-derive the warm-up split from the stream and re-run. We rebuild a
    shallow copy of S with W overridden so run_subset scores from that point."""
    out = {}
    N = S["N"]
    for k in values:
        S2 = dict(S); S2["W"] = int(N * k)
        out[str(k)] = _leaf(S2)
    return out


def sweep_roll(S, values=(200, 400, 800, 1600)):
    # ROLL only affects the rolling-metric trajectory, not the end-of-stream
    # cum metrics; we report the LAST rolling-window PR/ROC at each setting.
    out = {}
    r = OJ.run_subset(S, **DEPLOYED)
    p, y, W, N = r["p"], r["y"], S["W"], S["N"]
    for roll in values:
        lo = max(0, len(y) - roll)
        cm = OJ.cum_metrics(y[lo:], np.clip(np.nan_to_num(p[lo:], nan=float(y.mean()), posinf=1.0, neginf=0.0), 0, 1))
        out[str(roll)] = {k: float(cm.get(k, float("nan")))
                          for k in ("ROC_AUC", "PR_AUC", "F1", "MCC")}
    return out


def main():
    cache = OUT / "online_jit_streams_v5.pkl"
    if not cache.exists():
        raise SystemExit(f"no cached streams at {cache}; run experiments first.")
    S = pickle.load(open(cache, "rb"))
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{PROJECT}] parameter sweeps (deployed F+G, from cache)\n")

    sweeps = {
        "K":    ("WARMUP_FRAC", sweep_warmup(S)),
        "M":    ("BLOCK",       sweep_block(S)),
        "ROLL": ("ROLL",        sweep_roll(S)),
    }
    for name, (knob, res) in sweeps.items():
        res["_meta"] = dict(project=PROJECT, knob=knob, metric_keys=REPORT)
        json.dump(res, open(PARAM_DIR / f"{name}.json", "w"), indent=2)
        print(f"=== {name}  ({knob}) ===")
        vals = [k for k in res if k != "_meta"]
        cols = REPORT if name != "ROLL" else ["ROC_AUC", "PR_AUC", "F1", "MCC"]
        print("  value   " + "".join(f"{c:>9}" for c in cols))
        for v in vals:
            print(f"  {v:<7} " + "".join(f"{res[v].get(c, float('nan')):9.3f}" for c in cols))
        print()
    print(f"saved -> {PARAM_DIR}/{{K,M,ROLL}}.json")


if __name__ == "__main__":
    main()
