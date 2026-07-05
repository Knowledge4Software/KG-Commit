"""
Comprehensive fusion ablation on the FINAL method (Core + AST + CSTG, no X) to
choose a defensible, high-performance fusion formula.

Ten inference methods are combined, each run on the final graph, leakage-free
(prequential). Two fusion styles are evaluated and compared (per the design
decision):
  * SCORE STACKING  -- every method emits one probability; the fusion is a
    prequential logistic regression over the selected methods' probabilities.
    All 2^10-1 = 1023 non-empty subsets are evaluated.
  * FEATURE FUSION  -- raw feature blocks of the top channels {M,R,T,P,G,DW} are
    concatenated into one prequential LR (the current Fusion's style). All
    2^6-1 = 63 subsets are evaluated.

Methods / channels
  M   JIT metrics (LR)                 [non-graph reference]
  R   relational priors (wvRN)         graph/relational
  T   AST structural TF-IDF            graph
  P   Personalized PageRank            graph
  E   KG embedding (SVD/LSA)           graph
  G   CSTG semantic-text               graph
  RN  relational neighbour (wvRN vote) graph
  LP  label propagation                graph
  DW  DeepWalk embedding               graph
  KGE DistMult KG embedding            graph

All seven headline metrics are reported (Precision, Recall, Macro-F1, Buggy-F1,
G-Mean, AUC, Accuracy) at the online-tuned operating point (online_jit).

Out: outputs/fusion_ablation_results.pkl  +  outputs/tables/v4/fusion_ablation_*.csv

Run:  python inference/run_fusion_ablation.py [--limit N]
"""
import argparse, pickle, itertools, csv
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings; from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import kg_methods as km
from online_jit import final_metrics
from online_infer import build_incidence, online_priors

OUT = Path(__file__).resolve().parent.parent / "outputs"
TAB = OUT / "tables" / "v4"; TAB.mkdir(parents=True, exist_ok=True)
BLOCK = 50
REFIT_EMB = 4                    # refit embeddings every N blocks
DW_DIM, KGE_DIM, SVD_DIM = 64, 32, 64

METHODS = ["M", "R", "T", "P", "E", "G", "RN", "LP", "DW", "KGE"]
GRAPHY  = {"R", "T", "P", "E", "G", "RN", "LP", "DW", "KGE"}   # everything but M
FF_CHANNELS = ["M", "R", "T", "P", "G", "DW"]                  # feature-fusion set
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]


def _lr(sparse=False):
    return LogisticRegression(max_iter=1500, class_weight="balanced",
                              solver="liblinear" if sparse else "lbfgs")


# ── Phase A: leakage-free per-commit scores + embedding blocks ───────────────

def compute_streams(limit=0):
    print("loading cached streams + KG cache ...")
    S = pickle.load(open(OUT / "online_jit_streams_v5.pkl", "rb"))
    commits, tokens, files, devs = pickle.load(open(OUT / "kg_stream_cache.pkl", "rb"))
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y = np.asarray(S["y"]); N = int(S["N"]); W = int(S["W"])
    Ntot = N if limit <= 0 else min(N, limit)

    Xms = StandardScaler().fit(S["Xms"][:W]).transform(S["Xms"])      # M
    Xp  = S["Xp"]                                                     # R (priors)
    Xh  = S["Xh"]                                                     # T (tfidf)
    ppr = S["ppr_full"]                                               # P
    # G (CSTG): dense context + sparse text
    Gdense = np.hstack([S["cstg_prior"][:, None], S["cstg_typed"], S["cstg_consist"]])
    Gsparse = S["Xcstg"]

    # final Core+AST graph for the graph-native methods (token+file+dev hubs)
    G = km.build_scoped_graph(cids, tokens, files, devs, "full")
    C, Ptrans, Nc, Nh, Ngraph, edges = (G["C"], G["P"], G["Nc"], G["Nh"], G["N"], G["edges"])

    score = {m: np.full(N, np.nan) for m in METHODS}
    emb = {"E": np.zeros((N, SVD_DIM)), "DW": np.zeros((N, DW_DIM)), "KGE": np.zeros((N, KGE_DIM))}
    score["P"] = ppr.copy()                                          # already prequential

    from sklearn.decomposition import TruncatedSVD
    past_mask = np.zeros(Nc, bool)
    mlr = tlr = glr = None; svdE = dwE = kgeE = None
    esvd_clf = dw_clf = kge_clf = None
    i = W; blk = 0
    while i < Ntot:
        j = min(Ntot, i + BLOCK); idx = np.arange(i, j); past = np.arange(i)
        grate = y[past].mean() if len(past) else 0.0
        # cheap per-block
        score["R"][idx] = 0.5 * Xp[idx, 2] + 0.3 * Xp[idx, 1] + 0.2 * Xp[idx, 0]
        score["RN"][idx] = km.rn_scores(C, y, past, idx)
        score["LP"][idx] = km.lp_scores(C, y, past, idx)
        # refit LR heads
        if blk % 2 == 0:
            mlr = _lr().fit(Xms[past], y[past]) if len(set(y[past])) > 1 else None
            tlr = _lr(sparse=True).fit(Xh[past], y[past]) if len(set(y[past])) > 1 else None
            gX = sp.hstack([sp.csr_matrix(Gdense[past]), Gsparse[past]]).tocsr()
            glr = _lr(sparse=True).fit(gX, y[past]) if len(set(y[past])) > 1 else None
        score["M"][idx] = mlr.predict_proba(Xms[idx])[:, 1] if mlr else grate
        score["T"][idx] = tlr.predict_proba(Xh[idx])[:, 1] if tlr else grate
        gXb = sp.hstack([sp.csr_matrix(Gdense[idx]), Gsparse[idx]]).tocsr()
        score["G"][idx] = glr.predict_proba(gXb)[:, 1] if glr else grate
        # embeddings (refit on past every REFIT_EMB blocks)
        if blk % REFIT_EMB == 0:
            past_mask[:] = False; past_mask[past] = True
            d = min(SVD_DIM, C.shape[1] - 1)
            svdE = TruncatedSVD(n_components=max(d, 2), random_state=0).fit(C[past])
            Efull = svdE.transform(C)
            esvd_clf = _lr().fit(Efull[past], y[past]) if len(set(y[past])) > 1 else None
            dwE = km.dw_embed(C, past, DW_DIM)
            dw_clf = _lr().fit(dwE[past], y[past]) if len(set(y[past])) > 1 else None
            kgeE = km.kge_embed(edges, Nc, Nh, past_mask, dim=KGE_DIM)
            kge_clf = _lr().fit(kgeE[past], y[past]) if len(set(y[past])) > 1 else None
        emb["E"][idx] = Efull[idx]; emb["DW"][idx] = dwE[idx]; emb["KGE"][idx] = kgeE[idx]
        score["E"][idx]   = esvd_clf.predict_proba(Efull[idx])[:, 1] if esvd_clf else grate
        score["DW"][idx]  = dw_clf.predict_proba(dwE[idx])[:, 1] if dw_clf else grate
        score["KGE"][idx] = kge_clf.predict_proba(kgeE[idx])[:, 1] if kge_clf else grate
        i = j; blk += 1
        if blk % 10 == 0: print(f"  scored {j}/{Ntot}")

    feat = {"M": Xms, "R": Xp, "T": Xh, "P": ppr[:, None],
            "G": (Gdense, Gsparse), "E": emb["E"], "DW": emb["DW"], "KGE": emb["KGE"]}
    return dict(score=score, feat=feat, y=y, N=N, W=W, Ntot=Ntot)


# ── Phase B: score stacking over any subset of methods ──────────────────────

def stack_eval(D, subset, init=300):
    y, W, N = D["y"], D["W"], D["Ntot"]
    Z = np.column_stack([np.nan_to_num(D["score"][m], nan=y[:W].mean()) for m in subset])
    ev0 = W + init
    pred = np.full(N, np.nan); i = ev0; blk = 0
    clf = _lr().fit(Z[W:ev0], y[W:ev0]) if len(set(y[W:ev0])) > 1 else None
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        pred[idx] = clf.predict_proba(Z[idx])[:, 1] if clf else y[:i].mean()
        if blk % 3 == 0 and len(set(y[W:j])) > 1:      # periodic refit (prequential)
            clf = _lr().fit(Z[W:j], y[W:j])
        i = j; blk += 1
    ev = np.arange(ev0, N)
    return final_metrics(y[ev], np.nan_to_num(pred[ev], nan=y[ev].mean()))


# ── Phase C: feature fusion over a subset of channels ───────────────────────

def feat_eval(D, subset, init=300, refit=5):
    y, W, N = D["y"], D["W"], D["Ntot"]; feat = D["feat"]
    def build(idx, scaler):
        dens, spar = [], []
        for m in subset:
            f = feat[m]
            if m == "T":
                spar.append(f[idx])
            elif m == "G":
                gd, gs = f; dens.append(gd[idx]); spar.append(gs[idx])
            else:
                dens.append(f[idx] if f.ndim > 1 else f[idx][:, None])
        mats = []
        if dens:
            D_ = np.hstack(dens); mats.append(sp.csr_matrix(scaler.transform(D_)))
        mats += spar
        return sp.hstack(mats).tocsr() if mats else None
    def dcols(idx):
        dens = []
        for m in subset:
            if m == "T": continue
            f = feat[m]
            if m == "G": dens.append(f[0][idx])
            else: dens.append(f[idx] if f.ndim > 1 else f[idx][:, None])
        return np.hstack(dens) if dens else None
    ev0 = W + init
    sc = StandardScaler().fit(dcols(np.arange(W, ev0))) if dcols(np.arange(W, ev0)) is not None else _Id()
    clf = _lr(sparse=True).fit(build(np.arange(W, ev0), sc), y[W:ev0]) if len(set(y[W:ev0])) > 1 else None
    pred = np.full(N, np.nan); i = ev0; blk = 0
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        pred[idx] = clf.predict_proba(build(idx, sc))[:, 1] if clf else y[:i].mean()
        if blk % refit == 0 and len(set(y[W:j])) > 1:
            dc = dcols(np.arange(W, j)); sc = StandardScaler().fit(dc) if dc is not None else sc
            clf = _lr(sparse=True).fit(build(np.arange(W, j), sc), y[W:j])
        i = j; blk += 1
    ev = np.arange(ev0, N)
    return final_metrics(y[ev], np.nan_to_num(pred[ev], nan=y[ev].mean()))


class _Id:
    def transform(self, x): return x


def _subsets(items):
    for r in range(1, len(items) + 1):
        for c in itertools.combinations(items, r):
            yield list(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="only a few subsets (smoke)")
    args = ap.parse_args()

    D = compute_streams(args.limit)
    results = {"stack": {}, "feat": {}, "meta": dict(N=D["N"], W=D["W"], methods=METHODS,
                                                     ff_channels=FF_CHANNELS)}

    # score-stacking over all subsets of the 10 methods
    combos = list(_subsets(METHODS))
    if args.quick: combos = [c for c in combos if len(c) <= 2][:20]
    print(f"\nscore-stacking: {len(combos)} subsets ...")
    for k, sub in enumerate(combos):
        key = "+".join(sub)
        results["stack"][key] = dict(methods=sub, graph_only=(set(sub) <= GRAPHY),
                                     **stack_eval(D, sub))
        if k % 100 == 0: print(f"  stack {k}/{len(combos)}")

    # feature-fusion over subsets of the top channels
    fcombos = list(_subsets(FF_CHANNELS))
    if args.quick: fcombos = fcombos[:8]
    print(f"\nfeature-fusion: {len(fcombos)} subsets ...")
    for k, sub in enumerate(fcombos):
        key = "+".join(sub)
        results["feat"][key] = dict(methods=sub, graph_only=(set(sub) <= GRAPHY),
                                    **feat_eval(D, sub))
        if k % 10 == 0: print(f"  feat {k}/{len(fcombos)}")

    pickle.dump(results, open(OUT / "fusion_ablation_results.pkl", "wb"))
    # CSV dumps
    for fam in ("stack", "feat"):
        with open(TAB / f"fusion_ablation_{fam}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["fusion", "n", "graph_only"] + M7)
            for key, r in sorted(results[fam].items(), key=lambda kv: -kv[1]["Buggy_F1"]):
                w.writerow([key, len(r["methods"]), int(r["graph_only"])] +
                           [f"{r[mk]:.4f}" for mk in M7])

    # quick top-lines
    def top(fam, metric, graph_only=None, n=8):
        items = [(k, r) for k, r in results[fam].items()
                 if graph_only is None or r["graph_only"] == graph_only]
        return sorted(items, key=lambda kv: -kv[1][metric])[:n]
    print("\n=== TOP score-stacking by Buggy-F1 (all) ===")
    for k, r in top("stack", "Buggy_F1"):
        print(f"  {r['Buggy_F1']:.3f} BF1 | {r['Macro_F1']:.3f} MF1 | {r['G_Mean']:.3f} GM "
              f"| {r['AUC']:.3f} AUC | {'GRAPH' if r['graph_only'] else 'has-M'} | {k}")
    print("\n=== TOP GRAPH-ONLY score-stacking by Buggy-F1 ===")
    for k, r in top("stack", "Buggy_F1", graph_only=True):
        print(f"  {r['Buggy_F1']:.3f} BF1 | {r['Macro_F1']:.3f} MF1 | {r['AUC']:.3f} AUC | {k}")
    print("\n=== TOP feature-fusion by Buggy-F1 ===")
    for k, r in top("feat", "Buggy_F1"):
        print(f"  {r['Buggy_F1']:.3f} BF1 | {r['Macro_F1']:.3f} MF1 | {r['AUC']:.3f} AUC "
              f"| {'GRAPH' if r['graph_only'] else 'has-M'} | {k}")
    print(f"\nsaved -> {OUT/'fusion_ablation_results.pkl'} + CSVs")


if __name__ == "__main__":
    main()
