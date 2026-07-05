"""
Evaluate the five graph-native online KG-inference methods (kg_methods.py) on
every structural subgraph, on BOTH graph scopes (subgraph-only and full), under
the identical prequential protocol used elsewhere. This is ADDITIVE: it writes a
NEW cache and does not touch subgraph_rq_results.pkl (M/R/T/P/E/Fusion) at all.

Out: outputs/subgraph_kg_methods_results.pkl
     { variant: { scope: { method: final_metrics-dict } , 'meta': {...} } }

Run:  python inference/run_kg_methods_rq.py            # all variants, both scopes
      python inference/run_kg_methods_rq.py --limit 1500 --variants V3_ast
"""
import argparse, pickle
import numpy as np
from pathlib import Path

from advanced_infer import load_kg
from online_infer import WARMUP_FRAC, BLOCK
from online_jit import final_metrics
import kg_methods as km

OUT = Path(__file__).resolve().parent.parent / "outputs"
REFIT_EMB = 6          # refit DeepWalk / KGE embeddings every N blocks
DW_DIM = 64; KGE_DIM = 32

VARIANTS = [
    ("V1_none", "none (core only)", None,      None),
    ("V2a_cfg", "CFG",              "CFGNode", "atype"),
    ("V2b_dfg", "DFG",              "DFGNode", "atype"),
    ("V2c_pdg", "PDG/CPG",          "PDGNode", "atype"),
    ("V2d_seq", "Token-seq",        "SEQNode", "atype"),
    ("V3_ast",  "AST",              "ASTNode", "ast_type"),
]


def run_variant_scope(cids, y, tokens, files, devs, scope, limit=0):
    G = km.build_scoped_graph(cids, tokens, files, devs, scope)
    C, P, Nc, Nh, N, edges = (G["C"], G["P"], G["Nc"], G["Nh"], G["N"], G["edges"])
    Ntot = len(cids) if limit <= 0 else min(len(cids), limit)
    w = int(len(cids) * WARMUP_FRAC)
    past_mask = np.zeros(Nc, bool)

    pred = {m: np.full(Nc, np.nan) for m in km.METHODS}
    dw_clf = kge_clf = None; dwE = kgeE = None
    i = w; blk = 0
    while i < Ntot:
        j = min(Ntot, i + BLOCK); block = np.arange(i, j); past = np.arange(i)
        # --- cheap per-block graph methods ---
        pred["RN"][block] = km.rn_scores(C, y, past, block)
        pred["LP"][block] = km.lp_scores(C, y, past, block)
        bs = past[y[past] == 1]; gs = past[y[past] == 0]
        rb = km._ppr(P, list(bs), N); rg = km._ppr(P, list(gs), N)
        pred["PPR"][block] = rb[block] / (rb[block] + rg[block] + 1e-12)
        # --- embeddings: refit on past every REFIT_EMB blocks, then LR head ---
        if blk % REFIT_EMB == 0:
            past_mask[:] = False; past_mask[past] = True
            dwE = km.dw_embed(C, past, DW_DIM)
            dw_clf = km._lr().fit(dwE[past], y[past]) if len(set(y[past])) > 1 else None
            kgeE = km.kge_embed(edges, Nc, Nh, past_mask, dim=KGE_DIM)
            kge_clf = km._lr().fit(kgeE[past], y[past]) if len(set(y[past])) > 1 else None
        grate = y[past].mean() if len(past) else 0.0
        pred["DW"][block]  = dw_clf.predict_proba(dwE[block])[:, 1] if dw_clf else grate
        pred["KGE"][block] = kge_clf.predict_proba(kgeE[block])[:, 1] if kge_clf else grate
        i = j; blk += 1

    ev = np.arange(w, Ntot); yt = y[ev]
    return {m: final_metrics(yt, np.nan_to_num(pred[m][ev], nan=yt.mean()))
            for m in km.METHODS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--scopes", nargs="*", default=km.SCOPES)
    args = ap.parse_args()
    variants = [v for v in VARIANTS if not args.variants or v[0] in args.variants]

    results = {}
    for vid, name, label, prop in variants:
        commits, tokens, files, devs = load_kg(label, prop)
        cids = sorted(commits, key=lambda c: commits[c]["ts"])
        y = np.array([commits[c]["buggy"] for c in cids])
        results[vid] = {"name": name, "label": label}
        for scope in args.scopes:
            print(f"\n=== {vid} [{scope}] ===")
            results[vid][scope] = run_variant_scope(cids, y, tokens, files, devs,
                                                    scope, args.limit)
            for m in km.METHODS:
                r = results[vid][scope][m]
                print(f"  {km.METHOD_NAME[m]:<30} PR={r['PR_AUC']:.3f} "
                      f"ROC={r['ROC_AUC']:.3f} F1on={r['F1_online']:.3f}")

    # per-scope method x variant PR-AUC tables
    for scope in args.scopes:
        print("\n" + "=" * 74)
        print(f"GRAPH-NATIVE METHOD x SUBGRAPH  --  PR-AUC  [scope={scope}]")
        short = {v[0]: v[1][:6] for v in variants}
        hdr = f"{'method':<30}" + "".join(f"{short[v[0]]:>8}" for v in variants)
        print(hdr); print("-" * len(hdr))
        for m in km.METHODS:
            row = f"{km.METHOD_NAME[m]:<30}"
            vals = [results[v[0]][scope][m]["PR_AUC"] for v in variants]
            best = max(vals)
            for val in vals:
                row += f"{val:>7.3f}" + ("<" if abs(val - best) < 1e-9 else " ")
            print(row)

    OUT.mkdir(exist_ok=True)
    results["_meta"] = dict(warmup_frac=WARMUP_FRAC, block=BLOCK, refit_emb=REFIT_EMB,
                            dw_dim=DW_DIM, kge_dim=KGE_DIM, scopes=list(args.scopes))
    pickle.dump(results, open(OUT / "subgraph_kg_methods_results.pkl", "wb"))
    print(f"\nsaved -> {OUT/'subgraph_kg_methods_results.pkl'}")


if __name__ == "__main__":
    main()
