"""
"G after a while": use F alone for the first S scored commits, then switch to F+G.
=================================================================================

MOTIVATION
----------
The final run showed a systematic pattern: the CSTG channel G *helps* on the four
largest projects and *hurts* on the three smallest (spark -0.136, zookeeper -0.085).
CSTG must accumulate term statistics before it is informative; at K=0.05 the small
projects never give it enough. So G is not uniformly good -- it is good *once enough
text has been seen*.

This experiment tests the obvious remedy: score with F until commit S of the
evaluation stream, then score with F+G from S+1 onward. If the hypothesis is right,
a switch point exists that is at least as good as both endpoints on every project,
and strictly better on the small ones.

CACHE-ONLY: reads raw_fusion_scores{,_overall}.pkl, which store the per-commit
probability of F and F+G over the same aligned evaluation span. Splicing two score
vectors needs no Neo4j and no re-inference -- S is a post-hoc decision.

Note S is measured in commits *from the start of the evaluation span* (ev0), not
from commit 0, so it is comparable across projects.

Out: outputs/<project>/final_run/switch/switch_results.json
Run: python inference/run_switch_experiment.py            # all projects
     python inference/run_switch_experiment.py --project flink
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from protocol import PROJECTS, GAP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"

# Switch points to sweep, in commits from the start of the evaluation span.
# 0 = pure F+G (switch immediately); inf = pure F (never switch).
#
# IMPORTANT: on projects whose scoring starts inside the first block, the first
# BLOCK(=200) commits carry a single constant fallback score (y[:i].mean()) for BOTH
# F and F+G, because the fusion head has not refit yet. Any S <= 200 is therefore a
# no-op there -- splicing a constant over the same constant. S=200 in particular sat
# exactly on the block boundary and made the switch look inert on four projects.
# The grid now includes points clearly inside the region where F and F+G differ.
GRID = [0, 100, 200, 250, 300, 400, 500, 600, 750, 1000, 1500, 2000, 3000, 4000, 6000]
REPORT = ["Macro_F1", "G_Mean", "ROC_AUC", "PR_AUC", "Buggy_F1", "MCC"]


def metrics(y, p, gap=GAP):
    from online_jit import final_metrics
    return {k: float(v) for k, v in final_metrics(np.asarray(y), np.asarray(p),
                                                  gap=gap).items()}


def spliced(pF, pFG, s):
    """F for the first s scored commits, then F+G."""
    p = np.asarray(pFG, dtype=float).copy()
    s = min(int(s), len(p))
    if s > 0:
        p[:s] = np.asarray(pF, dtype=float)[:s]
    return p


def run_project(project, rule_suffix=""):
    """rule_suffix: '' = per-project F, '_overall' = the fixed overall F."""
    # ALWAYS read final_run/. The project root still holds pre-final-run copies
    # (some months old); preferring them silently mixed old and new results.
    f = OUTP / project / "final_run" / "fusion" / f"raw_fusion_scores{rule_suffix}.pkl"
    if not f.exists():
        return None
    d = pickle.load(open(f, "rb"))
    if "F" not in d["scores"] or "F+G" not in d["scores"]:
        return None
    y = np.asarray(d["y"], dtype=int)
    pF, pFG = d["scores"]["F"], d["scores"]["F+G"]
    n = len(y)

    # Where do F and F+G actually start to differ? Below this index the switch is a
    # no-op (both models are still emitting the constant warm-up prior), so a switch
    # point smaller than this tests nothing.
    diff = np.abs(np.asarray(pF, float) - np.asarray(pFG, float))
    first_diff = int(np.argmax(diff > 1e-12)) if (diff > 1e-12).any() else -1

    out = {"n_eval": n, "ev0": int(d["ev0"]), "first_diff_idx": first_diff,
           "grid": {}}
    out["F_only"] = metrics(y, pF)
    out["FG_only"] = metrics(y, pFG)
    for s in GRID:
        if s >= n:
            continue
        out["grid"][str(s)] = metrics(y, spliced(pF, pFG, s))
    # also the "never switch" endpoint, for completeness
    out["grid"]["inf"] = out["F_only"]

    # NOTE: a per-project argmax is recorded only as an upper bound / diagnostic.
    # The DEPLOYED S is a single global constant chosen in main() from the
    # cross-project aggregate -- a per-project S would be another tuned knob and
    # would reintroduce exactly the selection objection the fixed overall F removes.
    best_s, best_v = None, -1e9
    for s, m in out["grid"].items():
        if m["Macro_F1"] > best_v:
            best_v, best_s = m["Macro_F1"], s
    out["best_S_perproject"] = best_s          # diagnostic ONLY, not deployed
    out["best_Macro_F1_perproject"] = best_v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    a = ap.parse_args()
    projects = [a.project] if a.project else PROJECTS

    allres = {}
    for p in projects:
        res = {}
        for tag, suf in (("per_project", ""), ("overall", "_overall")):
            r = run_project(p, suf)
            if r:
                res[tag] = r
        if not res:
            print(f"  {p}: no raw fusion scores -- skipped")
            continue
        allres[p] = res
        d = res.get("per_project")
        if d:
            print(f"  {p:<11} n={d['n_eval']:<6} F={d['F_only']['Macro_F1']:.3f} "
                  f"F+G={d['FG_only']['Macro_F1']:.3f} "
                  f"(per-proj argmax S={d['best_S_perproject']})")
        outdir = OUTP / p / "final_run" / "switch"
        outdir.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(outdir / "switch_results.json", "w"), indent=2)

    # ---- choose ONE global S from the cross-project aggregate ----------------
    print("\n" + "=" * 74)
    print("GLOBAL S selection (macro-mean Macro-F1 over all projects)")
    print("=" * 74)
    summary = {}
    for tag in ("per_project", "overall"):
        have = [p for p in allres if tag in allres[p]]
        if not have:
            continue
        # A project shorter than S never reaches the switch, so it is simply F for
        # its whole stream -- that is the correct behaviour of a single global S,
        # not a missing value. Fall back to the F-only score rather than dropping
        # the grid point (which previously truncated the sweep at S=500 because
        # zookeeper has only 748 scored commits).
        keys = list(map(str, GRID)) + ["inf"]
        means = {}
        for k in keys:
            vals = []
            for p in have:
                g = allres[p][tag]["grid"]
                vals.append(g[k]["Macro_F1"] if k in g
                            else allres[p][tag]["F_only"]["Macro_F1"])
            means[k] = sum(vals) / len(vals)
        best = max(means, key=means.get)
        summary[tag] = {"n_projects": len(have), "means": means, "best_S": best,
                        "best_mean": means[best]}
        print(f"\n[{tag}]  ({len(have)} projects)")
        for k in keys:
            mark = "  <== best" if k == best else ""
            print(f"   S={k:<6} mean Macro-F1 = {means[k]:.4f}{mark}")

    js = OUTP / "switch_experiment_all.json"
    json.dump({"per_project_results": allres, "global_S": summary,
               "grid": GRID, "note": "S is ONE global constant, in commits from ev0"},
              open(js, "w"), indent=2)
    print(f"\nsaved -> {js}")


if __name__ == "__main__":
    main()
