"""
Build outputs/<project>/final_final_run/ : fusion re-scored with the INIT-fit fix.
=================================================================================

WHAT CHANGED
------------
eval_subset's initial stacker window was u0 = max(W, ev0 - gap). When the adaptive
INIT equals the gap (spark: W=73, init=50, gap=50 -> ev0-gap = 73 = W) that window
is EMPTY, the stacker never fits, and all of block 0 falls back to the constant
y[:i].mean(). On spark that constant (0.2195) is below the 0.5 threshold, so every
commit in block 0 was predicted benign while 61% were buggy.

The fix widens u0 to ev0 when the gap-respecting window is unusable -- still strictly
past-only (ev0 is where scoring begins), so no future label is consulted.

CACHE-ONLY. Reads raw_method_scores.pkl (the five per-commit method scores on the
final graph, already persisted) plus the streams cache for M and G. The graph is NOT
traversed: a fusion is a logistic stack over columns that already exist, which is why
all 31 combinations can be evaluated without 31 graph reads.

Writes, per project, under final_final_run/:
    fusion/final_fusion_results.pkl        part1 (31 combos), part2, part2_overall
    fusion/raw_fusion_scores{,_overall}.pkl
    switch/switch_results.json             F / F+G / switch at every S
    config.json                            frozen protocol + fix marker
and a cross-project summary at outputs/final_final_run_summary.json.

Run:  python inference/build_final_final_run.py
      python inference/build_final_final_run.py --project spark
"""
import argparse
import itertools
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
TOL = 0.005
SWITCH_GRID = [0, 100, 200, 250, 300, 400, 500, 600, 750, 1000, 1500, 2000, 3000]
REPORT = ["Macro_F1", "G_Mean", "ROC_AUC", "PR_AUC", "Buggy_F1", "MCC",
          "Precision", "Recall", "ACC"]


def load(p):
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
        print(f"  {p}: stream cache misaligned -- skipped")
        return None
    return R, prev, S, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    a = ap.parse_args()

    import run_final_fusion as rff
    from online_jit import final_metrics

    def met(y, pr):
        return {k: float(v) for k, v in
                final_metrics(np.asarray(y), np.asarray(pr), gap=P.GAP).items()}

    projects = [a.project] if a.project else P.PROJECTS
    summary = {}

    for proj in projects:
        got = load(proj)
        if not got:
            continue
        R, prev, S, y = got
        N = len(y); W = int(R["warmup"])
        INIT_P = int(prev.get("meta", {}).get("init", 300))

        scores = {m: np.asarray(R["scores"]["final"][m], float) for m in METHODS}
        Xms = StandardScaler().fit(S["Xms"][:W]).transform(S["Xms"])
        scores["M"] = rff.channel_score(Xms, y, W, N)
        Gf = sp.hstack([sp.csr_matrix(np.hstack([
            S["cstg_prior"][:, None], S["cstg_typed"], S["cstg_consist"]])),
            S["Xcstg"]]).tocsr()
        scores["G"] = rff.channel_score(Gf, y, W, N, sparse=True)

        # ---- part1: all 31 combinations ----
        part1 = {}
        for r in range(1, 6):
            for c in itertools.combinations(METHODS, r):
                m, tr, _, _ = rff.eval_subset(scores, list(c), y, W, N, init=INIT_P)
                part1["+".join(c)] = dict(methods=list(c), n=r, metrics=m, traj=tr)

        def sc(v): return v["metrics"]["Macro_F1"]
        best = max(sc(v) for v in part1.values())
        cands = [(k, v) for k, v in part1.items() if sc(v) >= best - TOL]
        ckey, chosen = min(cands, key=lambda kv: (kv[1]["n"], -sc(kv[1])))

        def part2_for(F):
            out, raws, ev0r = {}, {}, None
            for name, extra in [("F", []), ("F+G", ["G"]),
                                ("F+M", ["M"]), ("F+G+M", ["G", "M"])]:
                m, tr, pev, ev0 = rff.eval_subset(scores, F + extra, y, W, N,
                                                  init=INIT_P)
                out[name] = dict(methods=F + extra, metrics=m, traj=tr)
                raws[name] = pev.astype(np.float32); ev0r = ev0
            return out, raws, ev0r

        F = chosen["methods"]
        part2, raw_per, ev0 = part2_for(F)
        F_ov = [m for m in P.OVERALL_F if m in METHODS]
        part2_ov, raw_ov, _ = part2_for(F_ov)

        # ---- switch: F for the first S, then F+G ----
        def switch_block(raws):
            pF = np.asarray(raws["F"], float); pFG = np.asarray(raws["F+G"], float)
            yv = y[np.arange(ev0, N)]
            d = np.abs(pF - pFG)
            out = {"first_diff_idx": int(np.argmax(d > 1e-12)) if (d > 1e-12).any() else -1,
                   "n_eval": len(yv), "F_only": met(yv, pF), "FG_only": met(yv, pFG),
                   "grid": {}}
            for s in SWITCH_GRID:
                if s >= len(yv):
                    continue
                q = pFG.copy()
                if s > 0:
                    q[:s] = pF[:s]
                out["grid"][str(s)] = met(yv, q)
            out["grid"]["inf"] = out["F_only"]
            return out

        sw = {"per_project": switch_block(raw_per),
              "overall": switch_block(raw_ov)}

        # ---- write ----
        dst = OUTP / proj / "final_final_run"
        (dst / "fusion").mkdir(parents=True, exist_ok=True)
        (dst / "switch").mkdir(parents=True, exist_ok=True)
        pickle.dump(dict(part1=part1, chosen=ckey, part2=part2, F=F,
                         chosen_overall="+".join(F_ov), part2_overall=part2_ov,
                         F_overall=F_ov,
                         meta=dict(prev.get("meta", {}), initfit_fix=True,
                                   source="final_run raw_method_scores (cache-only)")),
                    open(dst / "fusion" / "final_fusion_results.pkl", "wb"))
        ev = np.arange(ev0, N)
        for suf, raws in (("", raw_per), ("_overall", raw_ov)):
            pickle.dump(dict(names=list(raws), ev0=ev0, N=N, commit_index=ev,
                             y=np.asarray(y[ev], np.int8), scores=raws),
                        open(dst / "fusion" / f"raw_fusion_scores{suf}.pkl", "wb"))
        json.dump(sw, open(dst / "switch" / "switch_results.json", "w"), indent=2)
        cfg = P.as_dict(); cfg["INITFIT_FIX"] = True; cfg["INIT_used"] = INIT_P
        json.dump(cfg, open(dst / "config.json", "w"), indent=2)

        # carry the untouched artifacts across so the folder is self-contained
        src = OUTP / proj / "final_run"
        for sub in ("baselines", "experiments", "complexity", "params", "seeds"):
            if (src / sub).is_dir():
                shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)

        summary[proj] = {
            "chosen": ckey, "chosen_overall": "+".join(F_ov),
            "per_project": {k: part2[k]["metrics"] for k in part2},
            "overall": {k: part2_ov[k]["metrics"] for k in part2_ov},
            "switch": sw,
        }
        o = prev["part2"]["F+G"]["metrics"]["Macro_F1"]
        nn = part2["F+G"]["metrics"]["Macro_F1"]
        print(f"  {proj:<11} F+G {o:.4f} -> {nn:.4f} ({nn-o:+.4f})  F={ckey}")

    json.dump(summary, open(OUTP / "final_final_run_summary.json", "w"), indent=2)
    print(f"\nsaved -> {OUTP/'final_final_run_summary.json'}")


if __name__ == "__main__":
    main()
