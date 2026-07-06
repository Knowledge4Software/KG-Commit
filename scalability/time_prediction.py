"""
E4 -- Prediction-time latency & inference complexity for the FINAL methodology
(no rebuild, no live DB; the graph family is rebuilt in-memory from the cached
hub dictionaries).

We measure the deployed inference stack only: the five graph-inference methods
RN, PPR, LP, DW, KGE (inference/kg_methods.py) on the six-graph family, plus the
deployed fusion. Timing separates the two cost regimes of the prequential loop:

  * PREDICT  -- steady per-commit scoring of a block from strictly-past state
               (this is the deployment latency: ms per commit at inference time)
  * TRAIN    -- the bursty per-block refit of the embeddings (DW/KGE) and the
               graph rebuild, amortised across the block

For each (method, graph) we report predict ms/commit (median, p95), train
ms/block, and throughput (commits/s), plus the one-off graph-construction time.
We also record the graph size (Nc, Nh, nnz) so latency can be related to scale
and cross-checked against each method's known complexity:

  RN   O(nnz(C_block . C_past^T))     one sparse hop
  LP   O(iters . nnz(C))              spreading activation
  PPR  O(iters . nnz(P)) x 2 seeds    random walk with restart, per block
  DW   O(randomized_svd(PPMI))        refit every REFIT_EMB blocks
  KGE  O(epochs . |triples| . dim)    DistMult SGD, refit every REFIT_EMB blocks

Output: outputs/scalability/prediction_latency.json

Run:  python scalability/time_prediction.py            # all 6 graphs
      python scalability/time_prediction.py --quick    # core, ast, final only
"""
import argparse
import time
import numpy as np

import _common as C


def time_graph(g, cids, y, files, devs, tok, cstg, quick=False):
    import kg_methods as km
    import run_final_experiments as rfe
    from online_infer import WARMUP_FRAC, BLOCK

    # one-off graph construction cost
    t0 = time.perf_counter()
    Cm, P, Nc, Nh, N, edges = rfe.build_graph(g, cids, files, devs, tok, cstg)
    build_s = time.perf_counter() - t0
    nnz = int(Cm.nnz)

    W = int(Nc * WARMUP_FRAC)
    REFIT_EMB, DW_DIM, KGE_DIM = rfe.REFIT_EMB, rfe.DW_DIM, rfe.KGE_DIM
    # per-method timing accumulators
    predict_ms = {m: [] for m in C.FINAL_METHODS}   # per-commit predict latency
    train_ms = {m: [] for m in C.FINAL_METHODS}     # per-block train/refit latency
    past_mask = np.zeros(Nc, bool)
    dw_clf = kge_clf = dwE = kgeE = None

    i = W; blk = 0
    while i < Nc:
        j = min(Nc, i + BLOCK); idx = np.arange(i, j); past = np.arange(i)
        nblk = j - i; gr = y[past].mean() if len(past) else 0.0

        def per_commit(t_block):
            predict_ms_val = 1000.0 * t_block / max(nblk, 1)
            return predict_ms_val

        # RN
        t = time.perf_counter(); km.rn_scores(Cm, y, past, idx)
        predict_ms["RN"].append(per_commit(time.perf_counter() - t))
        # LP
        t = time.perf_counter(); km.lp_scores(Cm, y, past, idx)
        predict_ms["LP"].append(per_commit(time.perf_counter() - t))
        # PPR (two seeded walks)
        bs = past[y[past] == 1]; gs = past[y[past] == 0]
        t = time.perf_counter()
        rb = km._ppr(P, list(bs), N); rg = km._ppr(P, list(gs), N)
        _ = rb[idx] / (rb[idx] + rg[idx] + 1e-12)
        predict_ms["PPR"].append(per_commit(time.perf_counter() - t))
        # DW / KGE: train (refit) cost is bursty; predict cost is the LR apply
        if blk % REFIT_EMB == 0:
            past_mask[:] = False; past_mask[past] = True
            t = time.perf_counter()
            dwE = km.dw_embed(Cm, past, DW_DIM)
            dw_clf = km._lr().fit(dwE[past], y[past]) if len(set(y[past])) > 1 else None
            train_ms["DW"].append(1000.0 * (time.perf_counter() - t))
            t = time.perf_counter()
            kgeE = km.kge_embed(edges, Nc, Nh, past_mask, dim=KGE_DIM)
            kge_clf = km._lr().fit(kgeE[past], y[past]) if len(set(y[past])) > 1 else None
            train_ms["KGE"].append(1000.0 * (time.perf_counter() - t))
        t = time.perf_counter()
        _ = dw_clf.predict_proba(dwE[idx])[:, 1] if dw_clf else np.full(nblk, gr)
        predict_ms["DW"].append(per_commit(time.perf_counter() - t))
        t = time.perf_counter()
        _ = kge_clf.predict_proba(kgeE[idx])[:, 1] if kge_clf else np.full(nblk, gr)
        predict_ms["KGE"].append(per_commit(time.perf_counter() - t))
        # RN/LP/PPR are recomputed each block -> their "train" cost is folded into predict
        i = j; blk += 1

    def summ(vals):
        a = np.asarray(vals, float)
        if a.size == 0:
            return dict(median=0.0, p95=0.0, mean=0.0, n=0)
        return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                    mean=float(a.mean()), n=int(a.size))

    res = dict(graph=g, name=C.GRAPH_NAME[g], Nc=Nc, Nh=Nh, nnz=nnz,
               n_edges=int(edges.shape[0]), build_graph_s=build_s,
               predict_ms_per_commit={m: summ(predict_ms[m]) for m in C.FINAL_METHODS},
               train_ms_per_block={m: summ(train_ms[m]) for m in ("DW", "KGE")})
    # deployed-throughput proxy: sum of median per-commit predict latencies for F+G's
    # constituent methods (RN + PPR) -- the deployed model is graph-native.
    dep = res["predict_ms_per_commit"]["RN"]["median"] + \
        res["predict_ms_per_commit"]["PPR"]["median"]
    res["deployed_F_predict_ms_per_commit"] = dep
    res["deployed_F_throughput_cps"] = 1000.0 / dep if dep > 0 else float("inf")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    b = C.load_final_graph_cache()
    cids, y, files, devs, tok, cstg = (b["cids"], b["y"], b["files"], b["devs"],
                                       b["tok"], b["cstg"])
    graphs = ["core", "ast", "final"] if args.quick else C.FINAL_GRAPHS
    out = {"_meta": {"scope": "final V4 inference: RN/PPR/LP/DW/KGE on the six-graph "
                             "family; deployed F=RN+PPR (graph-native)",
                     "note": "predict = steady per-commit latency; train = bursty "
                             "embedding refit per block"}}
    for g in graphs:
        print(f"timing graph {C.GRAPH_NAME[g]} ...")
        r = time_graph(g, cids, y, files, devs, tok, cstg, quick=args.quick)
        out[g] = r
        pm = r["predict_ms_per_commit"]
        print(f"  Nh={r['Nh']:>7,} nnz={r['nnz']:>8,}  "
              f"RN={pm['RN']['median']:.3f} LP={pm['LP']['median']:.3f} "
              f"PPR={pm['PPR']['median']:.3f} DW={pm['DW']['median']:.4f} "
              f"KGE={pm['KGE']['median']:.4f} ms/commit; "
              f"F throughput={r['deployed_F_throughput_cps']:.0f} c/s")
    p = C.save_json(out, "prediction_latency.json")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
