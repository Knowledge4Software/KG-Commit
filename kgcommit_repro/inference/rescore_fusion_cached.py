"""
Re-score the fusion results from CACHED artifacts only -- no Neo4j, no re-inference.
====================================================================================

WHY
---
`eval_subset` had a degenerate initial-fit window: u0 = max(W, ev0-gap). When the
adaptive INIT equals the gap (spark: W=73, init=50, gap=50 -> ev0-gap = 73 = W) the
stacker's first fit window Z[W:u0] is EMPTY, clf stays None, and all of block 0
falls back to the constant y[:i].mean(). On spark that constant is 0.2195 < 0.5, so
every commit in block 0 was predicted benign while 61% were buggy (macro-F1 0.281 on
that block, -0.042 on the project). Single-method subsets skip the stacker, which is
why F (=PPR) was fine and F+G was not.

The fix widens u0 to ev0 when the gap-respecting window is unusable. This is still
strictly past-only: ev0 is where scoring begins, so no future label is consulted.

WHAT THIS SCRIPT DOES
---------------------
Rebuilds part1 (all 31 combos), part2 and part2_overall from:
  final_run/experiments/raw_method_scores.pkl   (the 5 method scores, final graph)
  online_jit_streams_v5.pkl                     (the M and G channels)
and rewrites final_fusion_results.pkl + raw_fusion_scores{,_overall}.{pkl,csv}.

Everything is read from disk. No graph, no Neo4j, no re-inference.

Run:  python inference/rescore_fusion_cached.py --project spark
      python inference/rescore_fusion_cached.py            # all ready projects
"""
import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from protocol import PROJECTS, OVERALL_F, GAP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
TOL = 0.005


def load_inputs(p):
    fr = OUTP / p / "final_run"
    rm = fr / "experiments" / "raw_method_scores.pkl"
    ff = fr / "fusion" / "final_fusion_results.pkl"
    st = OUTP / p / "online_jit_streams_v5.pkl"
    if not (rm.exists() and ff.exists() and st.exists()):
        return None
    R = pickle.load(open(rm, "rb"))
    prev = pickle.load(open(ff, "rb"))
    S = pickle.load(open(st, "rb"))
    y = np.asarray(R["y"], int)
    if len(np.asarray(S["y"])) != len(y):
        print(f"  {p}: stream cache misaligned ({len(S['y'])} vs {len(y)}) -- skipped")
        return None
    return R, prev, S, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import run_final_fusion as rff

    projects = [a.project] if a.project else PROJECTS
    for p in projects:
        got = load_inputs(p)
        if not got:
            continue
        R, prev, S, y = got
        N = len(y)
        W = int(R["warmup"])
        meta = prev.get("meta", {})
        INIT_P = int(meta.get("init", 300))

        scores = {m: np.asarray(R["scores"]["final"][m], float) for m in METHODS}
        Xms = StandardScaler().fit(S["Xms"][:W]).transform(S["Xms"])
        scores["M"] = rff.channel_score(Xms, y, W, N)
        Gfeat = sp.hstack([sp.csr_matrix(np.hstack([
            S["cstg_prior"][:, None], S["cstg_typed"], S["cstg_consist"]])),
            S["Xcstg"]]).tocsr()
        scores["G"] = rff.channel_score(Gfeat, y, W, N, sparse=True)

        # ---- part1: all 31 combinations -------------------------------------
        part1 = {}
        for r in range(1, 6):
            for combo in itertools.combinations(METHODS, r):
                m, tr, _, _ = rff.eval_subset(scores, list(combo), y, W, N,
                                              init=INIT_P)
                part1["+".join(combo)] = dict(methods=list(combo), n=r,
                                              metrics=m, traj=tr)

        def sc(v): return v["metrics"]["Macro_F1"]
        best = max(sc(v) for v in part1.values())
        cands = [(k, v) for k, v in part1.items() if sc(v) >= best - TOL]
        chosen_key, chosen = min(cands, key=lambda kv: (kv[1]["n"], -sc(kv[1])))

        def part2_for(F):
            out, raws, ev0r = {}, {}, None
            for name, extra in [("F", []), ("F+G", ["G"]),
                                ("F+M", ["M"]), ("F+G+M", ["G", "M"])]:
                m, tr, p_ev, ev0 = rff.eval_subset(scores, F + extra, y, W, N,
                                                   init=INIT_P)
                out[name] = dict(methods=F + extra, metrics=m, traj=tr)
                raws[name] = p_ev.astype(np.float32); ev0r = ev0
            return out, raws, ev0r

        F = chosen["methods"]
        part2, raw_per, ev0 = part2_for(F)
        F_ov = [m for m in OVERALL_F if m in METHODS]
        part2_ov, raw_ov, _ = part2_for(F_ov)

        old_fg = prev["part2"]["F+G"]["metrics"]["Macro_F1"]
        new_fg = part2["F+G"]["metrics"]["Macro_F1"]
        old_ov = prev.get("part2_overall", {}).get("F+G", {}).get("metrics", {}).get("Macro_F1", float("nan"))
        new_ov = part2_ov["F+G"]["metrics"]["Macro_F1"]
        print(f"  {p:<11} F+G {old_fg:.4f} -> {new_fg:.4f} ({new_fg-old_fg:+.4f})   "
              f"RN+PPR {old_ov:.4f} -> {new_ov:.4f} ({new_ov-old_ov:+.4f})   "
              f"[{prev['chosen']} -> {chosen_key}]")

        if a.dry_run:
            continue
        fr = OUTP / p / "final_run" / "fusion"
        pickle.dump(dict(part1=part1, chosen=chosen_key, part2=part2, F=F,
                         chosen_overall="+".join(F_ov), part2_overall=part2_ov,
                         F_overall=F_ov,
                         meta=dict(meta, rescored_initfit_fix=True)),
                    open(fr / "final_fusion_results.pkl", "wb"))
        for suf, raws in (("", raw_per), ("_overall", raw_ov)):
            ev = np.arange(ev0, N)
            pickle.dump(dict(names=list(raws), ev0=ev0, N=N, commit_index=ev,
                             y=np.asarray(y[ev], dtype=np.int8), scores=raws),
                        open(fr / f"raw_fusion_scores{suf}.pkl", "wb"))


if __name__ == "__main__":
    main()
