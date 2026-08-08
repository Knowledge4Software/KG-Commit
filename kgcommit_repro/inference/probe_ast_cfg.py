"""
PROBE: does adding a SECOND structural subgraph on top of AST help, and if so
which one? Compares, under the identical online protocol / fusion-selection rule
as run_final_fusion, the deployed-style fusion (F, then F+G) on:

  core          Core only (files+dev hubs; no subgraph, no CSTG)
  final         Core + AST        + CSTG   (the current deployed graph)
  final_astcfg  Core + AST + CFG  + CSTG
  final_astdfg  Core + AST + DFG  + CSTG
  final_astpdg  Core + AST + PDG  + CSTG

For each recipe it computes the five graph-method scores (RN/PPR/LP/DW/KGE) on that
graph, adds the CSTG channel G (same as the main pipeline; for `core` the CSTG
channel is still added so the comparison is apples-to-apples on F+G), runs the
Macro-F1 parsimony-aware fusion selection to get F, then evaluates F and F+G on all
seven metrics. Decision: pick the 2nd subgraph (if any) whose F+G Macro-F1 beats
AST-only by > 0.005; otherwise keep AST-only (3-tier methodology).

Needs the project's graph resident in Neo4j (AST+CFG+DFG+PDG layers built).

Run:  KGC_PROJECT=zookeeper python inference/probe_ast_cfg.py
Out:  outputs/<project>/probe_ast_cfg.{pkl,txt,csv}
"""
import pickle
import sys
import csv
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402

import run_final_experiments as rfe  # noqa: E402
import run_final_fusion as rff  # noqa: E402

M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]

# label -> graph key in run_final_experiments.build_graph
RECIPES = [
    ("Core",         "core"),
    ("AST-only",     "final"),
    ("AST+CFG",      "final_astcfg"),
    ("AST+DFG",      "final_astdfg"),
    ("AST+PDG",      "final_astpdg"),
]


def method_scores_for_graph(g, cids, y, files, devs, tok, cstg):
    _res, pred = rfe.run_graph(g, cids, y, files, devs, tok, cstg, limit=0)
    return {m: np.asarray(pred[m], dtype=float) for m in rfe.METHODS}


def deployed_style_fusion(scores5, gscore, y, W, N):
    """run_final_fusion Part-1 selection (Macro-F1, parsimony tie-break) then F, F+G."""
    import itertools
    scores = dict(scores5); scores["G"] = gscore
    part1 = {}
    for r in range(1, 6):
        for combo in itertools.combinations(rfe.METHODS, r):
            m, _tr, _p, _e = rff.eval_subset(scores, list(combo), y, W, N)
            part1["+".join(combo)] = dict(n=r, metrics=m, methods=list(combo))
    best = max(v["metrics"]["Macro_F1"] for v in part1.values())
    cands = [(k, v) for k, v in part1.items()
             if v["metrics"]["Macro_F1"] >= best - rff.TOL]
    chosen_key, chosen = min(cands, key=lambda kv: (kv[1]["n"], -kv[1]["metrics"]["Macro_F1"]))
    F = chosen["methods"]
    out = {"chosen": chosen_key}
    for name, extra in [("F", []), ("F+G", ["G"])]:
        m, _tr, _p, _e = rff.eval_subset(scores, F + extra, y, W, N)
        out[name] = {k: float(m[k]) for k in M7}
    return out


def main():
    commits, cids, y, files, devs, tok, cstg = rfe.load_all()
    y = np.asarray(y); N = len(cids); W = int(N * rfe.WARMUP_FRAC)

    # CSTG channel G, aligned to cids/W, exactly as run_final_fusion builds it
    S = pickle.load(open(OUT / "online_jit_streams_v5.pkl", "rb"))
    assert np.array_equal(np.asarray(S["y"]), y), "streams_v5 not aligned to cids"
    Gfeat = sp.hstack([sp.csr_matrix(np.hstack([S["cstg_prior"][:, None], S["cstg_typed"],
                                                S["cstg_consist"]])), S["Xcstg"]]).tocsr()
    gscore = rff.channel_score(Gfeat, y, W, N, sparse=True)

    results = {}
    for label, g in RECIPES:
        print(f"\n=== {label}  ({g}) ===")
        s5 = method_scores_for_graph(g, cids, y, files, devs, tok, cstg)
        res = deployed_style_fusion(s5, gscore, y, W, N)
        results[label] = res
        print(f"  chosen F = {res['chosen']}")
        for name in ("F", "F+G"):
            r = res[name]
            print("  " + name.ljust(4) + "  " +
                  "  ".join(f"{k.split('_')[0][:4]}={r[k]:.3f}" for k in M7))

    # ---- table: F+G, all 7 metrics, one row per recipe ----
    base = results["AST-only"]["F+G"]["Macro_F1"]
    header = ["recipe", "chosen_F"] + M7 + ["dMacroF1_vs_AST"]
    rows = []
    for label, _ in RECIPES:
        fg = results[label]["F+G"]
        rows.append([label, results[label]["chosen"]] +
                    [f"{fg[k]:.4f}" for k in M7] +
                    [f"{fg['Macro_F1'] - base:+.4f}"])
    with open(OUT / "probe_ast_cfg.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)

    # decision among the 3 candidate second-subgraphs
    cand = {lab: results[lab]["F+G"]["Macro_F1"] for lab in ("AST+CFG", "AST+DFG", "AST+PDG")}
    best_lab = max(cand, key=cand.get)
    d = cand[best_lab] - base
    verdict = (f"{best_lab} WINS by {d:+.4f}" if d > 0.005
               else f"AST-only WINS (best 2nd subgraph {best_lab} only {d:+.4f})")

    lines = [f"PROBE 2nd-subgraph  [{PROJECT}]  deployed F+G, all metrics:",
             "  " + "recipe".ljust(9) + "  " + "  ".join(m.split('_')[0][:5].rjust(6) for m in M7)]
    for label, _ in RECIPES:
        fg = results[label]["F+G"]
        lines.append("  " + label.ljust(9) + "  " +
                     "  ".join(f"{fg[k]:.4f}"[:6].rjust(6) for k in M7))
    lines += [f"  Macro-F1: AST-only={base:.4f}; "
              f"CFG={cand['AST+CFG']:.4f} DFG={cand['AST+DFG']:.4f} PDG={cand['AST+PDG']:.4f}",
              f"  -> {verdict}"]
    txt = "\n".join(lines)
    print("\n" + txt)

    OUT.mkdir(exist_ok=True)
    pickle.dump(dict(project=PROJECT, results=results, base_macrof1=base,
                     candidates=cand, best_second=best_lab, delta=d, verdict=verdict),
                open(OUT / "probe_ast_cfg.pkl", "wb"))
    (OUT / "probe_ast_cfg.txt").write_text(txt, encoding="utf-8")
    print(f"\nsaved -> {OUT/'probe_ast_cfg.pkl'} (+ .txt, .csv)")


if __name__ == "__main__":
    main()
