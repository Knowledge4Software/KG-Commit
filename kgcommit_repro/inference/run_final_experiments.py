"""
Final experiments: the five canonical graph-inference methods (RN, PPR, LP, DW,
KGE) evaluated online (prequentially) on a family of six knowledge graphs of
increasing richness, all sharing the Core relational layer (commit-file-developer):

  core   Core only (file + developer hubs)
  ast    Core + AST change-token hubs
  cfg    Core + CFG change-token hubs
  dfg    Core + DFG change-token hubs
  pdg    Core + PDG change-token hubs
  final  Core + AST tokens + CSTG term hubs   (the deployed KG; X excluded)

For every (method, graph) we record the seven headline metrics and the rolling
online-evaluation trajectory of each metric (for the stream plots). No commit
text (X) anywhere. Output: outputs/final_experiments_results.pkl

Run:  python inference/run_final_experiments.py [--limit N]
"""
import argparse, pickle, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score)

import kg_methods as km
from advanced_infer import load_kg
from online_infer import WARMUP_FRAC, BLOCK
from online_jit import final_metrics, online_decisions

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import OUT  # per-project outputs/<project>/
from config.project_config import NEO4J_URI, NEO4J_AUTH
REFIT_EMB = 5; DW_DIM = 64; KGE_DIM = 32; ROLL = 800
# online-trajectory VISUALISATION resolution: two versions are persisted for
# every stream -- the ACCURATE view (window=150, stride=25) and the SMOOTHED view
# (window=800, stride=25). Both are visualisation-only; neither affects the
# protocol or the end-of-stream metrics.
TRAJ_STRIDE = 25; TRAJ_WINDOW = 150; TRAJ_WINDOW_SMOOTH = 800

GRAPHS = ["core", "ast", "cfg", "dfg", "pdg", "final"]   # ast_method dropped (unused)
# final2 is an OPT-IN probe graph (Core+AST+CFG+CSTG) used only to test whether a
# SECOND structural subgraph on top of AST helps; it is not in the default GRAPHS,
# so normal runs are unaffected. Select it explicitly via --graphs ... final2.
GRAPH_NAME = {"core": "Core", "ast": "Core+AST", "ast_method": "Core+AST-m", "cfg": "Core+CFG",
              "dfg": "Core+DFG", "pdg": "Core+PDG", "final": "Core+AST+CSTG",
              "final2": "Core+AST+CFG+CSTG"}
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
REL = {"T": 0, "F": 1, "D": 2, "S": 3}     # token, file, dev, cstg-term


# ── data loading ─────────────────────────────────────────────────────────────

def fetch_cstg_terms(cids_set):
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    terms = {}
    with d.session() as s:
        for r in s.run("""MATCH (c:Commit {in_jit:true})-[:MENTIONS]->(t:Term)
                          RETURN c.id AS c, collect(t.id) AS ts"""):
            if r["c"] in cids_set:
                terms[r["c"]] = [(str(t), 1.0) for t in r["ts"]]
    d.close()
    return terms


def load_all():
    print("loading layers ...")
    commits, tok_ast, files, devs = load_kg("ASTNode", "ast_type")
    tok = {"ast": tok_ast}
    for g, lbl in [("ast_method", "ASTMethodNode"), ("cfg", "CFGNode"), ("dfg", "DFGNode"), ("pdg", "PDGNode")]:
        tok[g] = load_kg(lbl, "atype")[1]
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y = np.array([commits[c]["buggy"] for c in cids])
    cstg = fetch_cstg_terms(set(cids))
    print(f"  {len(cids)} commits; cstg terms for {len(cstg)} commits")
    return commits, cids, y, files, devs, tok, cstg


def build_graph(g, cids, files, devs, tok, cstg):
    """Per-commit hub multiset for graph g -> incidence C, PPR transition P, edges."""
    cidx = {c: i for i, c in enumerate(cids)}; Nc = len(cids)
    raw = []                                   # (i, hubkey, weight, relchar)
    for c in cids:
        i = cidx[c]
        for f in files.get(c, ()):  raw.append((i, "F:" + f, 1.0, "F"))
        if c in devs:               raw.append((i, "D:" + devs[c], 1.0, "D"))
        if g in ("ast", "ast_method", "cfg", "dfg", "pdg"):
            for t, n in tok[g].get(c, ()):     raw.append((i, f"T{g}:" + t, float(n), "T"))
        elif g == "final":
            for t, n in tok["ast"].get(c, ()): raw.append((i, "Tast:" + t, float(n), "T"))
            for t, n in cstg.get(c, ()):       raw.append((i, "S:" + t, float(n), "S"))
        elif g in ("final2", "final_astcfg", "final_astdfg", "final_astpdg"):
            # probe graphs: Core + AST + <second subgraph> + CSTG
            second = {"final2": "cfg", "final_astcfg": "cfg",
                      "final_astdfg": "dfg", "final_astpdg": "pdg"}[g]
            for t, n in tok["ast"].get(c, ()):    raw.append((i, "Tast:" + t, float(n), "T"))
            for t, n in tok[second].get(c, ()):   raw.append((i, f"T{second}:" + t, float(n), "T"))
            for t, n in cstg.get(c, ()):          raw.append((i, "S:" + t, float(n), "S"))
    df = defaultdict(int)
    for i, h, w, r in raw: df[h] += 1
    hub = {}; rows = []; cols = []; data = []; edges = []
    for i, h, w, r in raw:
        j = hub.setdefault(h, len(hub))
        rows.append(i); cols.append(j)
        data.append(w * math.log(1.0 + Nc / df[h]))
        edges.append((i, REL[r], j))
    Nh = len(hub); N = Nc + Nh
    C = sp.csr_matrix((data, (rows, cols)), shape=(Nc, max(Nh, 1)))
    B = sp.csr_matrix((data, (rows, [c + Nc for c in cols])), shape=(N, N))
    A = B + B.T; deg = np.asarray(A.sum(0)).ravel(); deg[deg == 0] = 1.0
    P = A.multiply(sp.csr_matrix(1.0 / deg)).tocsr()
    return C, P, Nc, Nh, N, (np.array(edges) if edges else np.zeros((0, 3), int))


# ── prequential run of the five methods on one graph ────────────────────────

def run_graph(g, cids, y, files, devs, tok, cstg, limit=0):
    from sklearn.decomposition import TruncatedSVD
    C, P, Nc, Nh, N, edges = build_graph(g, cids, files, devs, tok, cstg)
    Ntot = Nc if limit <= 0 else min(Nc, limit)
    W = int(Nc * WARMUP_FRAC)
    pred = {m: np.full(Nc, np.nan) for m in METHODS}
    past_mask = np.zeros(Nc, bool)
    dw_clf = kge_clf = None; dwE = kgeE = None
    i = W; blk = 0
    while i < Ntot:
        j = min(Ntot, i + BLOCK); idx = np.arange(i, j); past = np.arange(i)
        gr = y[past].mean() if len(past) else 0.0
        pred["RN"][idx] = km.rn_scores(C, y, past, idx)
        pred["LP"][idx] = km.lp_scores(C, y, past, idx)
        bs = past[y[past] == 1]; gs = past[y[past] == 0]
        rb = km._ppr(P, list(bs), N); rg = km._ppr(P, list(gs), N)
        pred["PPR"][idx] = rb[idx] / (rb[idx] + rg[idx] + 1e-12)
        if blk % REFIT_EMB == 0:
            past_mask[:] = False; past_mask[past] = True
            dwE = km.dw_embed(C, past, DW_DIM)
            dw_clf = km._lr().fit(dwE[past], y[past]) if len(set(y[past])) > 1 else None
            kgeE = km.kge_embed(edges, Nc, Nh, past_mask, dim=KGE_DIM)
            kge_clf = km._lr().fit(kgeE[past], y[past]) if len(set(y[past])) > 1 else None
        pred["DW"][idx]  = dw_clf.predict_proba(dwE[idx])[:, 1] if dw_clf else gr
        pred["KGE"][idx] = kge_clf.predict_proba(kgeE[idx])[:, 1] if kge_clf else gr
        i = j; blk += 1
    ev = np.arange(W, Ntot); yt = y[ev]
    out = {}
    for m in METHODS:
        p = np.clip(np.nan_to_num(pred[m][ev], nan=yt.mean()), 0, 1)
        out[m] = dict(metrics=final_metrics(yt, p),
                      traj=metric_traj(yt, p, W),
                      traj_smooth=metric_traj(yt, p, W, roll=TRAJ_WINDOW_SMOOTH))
    return out, pred      # pred: full-length per-commit scores per method


def metric_traj(y_ev, p_ev, warmup, roll=TRAJ_WINDOW, step=TRAJ_STRIDE, gap=0):
    """Rolling 7-metric online-evaluation trajectory (x = commit index).
    `gap` (verification-latency G) lags the online threshold's revealed labels."""
    yhat = online_decisions(p_ev, y_ev, gap=gap)
    tr = {"idx": []}; tr.update({m: [] for m in M7})
    n = len(y_ev)
    for j in range(step, n + 1, step):
        lo = max(0, j - roll); ys, ps, yh = y_ev[lo:j], p_ev[lo:j], yhat[lo:j]
        rec = recall_score(ys, yh, pos_label=1, zero_division=0)
        spec = recall_score(ys, yh, pos_label=0, zero_division=0)
        tr["idx"].append(warmup + j)
        tr["Precision"].append(precision_score(ys, yh, pos_label=1, zero_division=0))
        tr["Recall"].append(rec)
        tr["Macro_F1"].append(f1_score(ys, yh, average="macro", zero_division=0))
        tr["Buggy_F1"].append(f1_score(ys, yh, pos_label=1, zero_division=0))
        tr["G_Mean"].append(float(np.sqrt(max(rec, 0) * max(spec, 0))))
        tr["AUC"].append(roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else np.nan)
        tr["ACC"].append(float((yh == ys).mean()))
    return tr


def _lr_stack():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")


def fuse_methods(pred, y, W, N, refit=3):
    """Prequential-LR stacking of the 5 graph methods on ONE graph, so each graph
    (core, ast, ..., final) has a consistent 5-method Fusion comparable to the
    final-graph fusion in run_final_fusion. Same stacking recipe as eval_subset:
    fit LR on the expanding past window over the methods' per-commit scores."""
    Z = np.column_stack([np.nan_to_num(pred[m], nan=float(y[:W].mean())) for m in METHODS])
    p_all = np.full(N, np.nan); i = W; blk = 0
    clf = _lr_stack().fit(Z[:W], y[:W]) if len(set(y[:W])) > 1 else None
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        p_all[idx] = clf.predict_proba(Z[idx])[:, 1] if clf else float(y[:i].mean())
        if blk % refit == 0 and len(set(y[:j])) > 1:
            clf = _lr_stack().fit(Z[:j], y[:j])
        i = j; blk += 1
    ev = np.arange(W, N)
    p = np.clip(np.nan_to_num(p_all[ev], nan=float(y[ev].mean())), 0, 1)
    return dict(metrics=final_metrics(y[ev], p), traj=metric_traj(y[ev], p, W),
                traj_smooth=metric_traj(y[ev], p, W, roll=TRAJ_WINDOW_SMOOTH),
                p_ev=p.astype(np.float32),   # raw per-commit fused score (no Neo4j needed to re-smooth)
                methods=list(METHODS))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--graphs", nargs="*", default=GRAPHS)
    ap.add_argument("--outdir-suffix", type=str, default="")
    args = ap.parse_args()
    commits, cids, y, files, devs, tok, cstg = load_all()
    y = np.asarray(y); N = len(cids); W = int(N * WARMUP_FRAC)

    results = {"meta": dict(graphs=GRAPHS, methods=METHODS, metrics=M7,
                            warmup=W, N=N)}
    raw_scores = {}     # graph -> {method -> full-length per-commit score array}
    for g in args.graphs:
        print(f"\n=== graph: {GRAPH_NAME[g]} ===")
        res, pred = run_graph(g, cids, y, files, devs, tok, cstg, args.limit)
        # per-graph 5-method fusion (consistent model family across all graphs)
        if not args.limit:
            res["Fusion"] = fuse_methods(pred, y, W, N)
        results[g] = res
        raw_scores[g] = {m: np.asarray(pred[m], dtype=np.float32) for m in METHODS}
        for m in METHODS:
            r = res[m]["metrics"]
            print(f"  {m:<4} BF1={r['Buggy_F1']:.3f} MacroF1={r['Macro_F1']:.3f} "
                  f"GM={r['G_Mean']:.3f} AUC={r['AUC']:.3f}")
        if "Fusion" in res:
            r = res["Fusion"]["metrics"]
            print(f"  {'FUSE':<4} BF1={r['Buggy_F1']:.3f} MacroF1={r['Macro_F1']:.3f} "
                  f"GM={r['G_Mean']:.3f} AUC={r['AUC']:.3f}")
    out_dir = OUT if not args.outdir_suffix else OUT / args.outdir_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(results, open(out_dir / "final_experiments_results.pkl", "wb"))
    print(f"\nsaved -> {out_dir/'final_experiments_results.pkl'}")

    # ---- persist RAW per-commit method scores so any future re-smoothing /
    #      re-selection / new-window rendering needs NO Neo4j rebuild ----
    if not args.limit:
        _persist_raw_scores(raw_scores, cids, y, W, N, out_dir)


def _persist_raw_scores(raw_scores, cids, y, W, N, out_dir):
    """Dump the full-length per-method per-graph score arrays (+ y, cids, warmup)
    to a .pkl and a flat long .csv. Downstream re-smoothing at any window reads
    these instead of recomputing from Neo4j."""
    import csv
    bundle = dict(graphs=list(raw_scores.keys()), methods=METHODS,
                  cids=list(cids), y=np.asarray(y, dtype=np.int8),
                  warmup=W, N=N, scores=raw_scores)
    pickle.dump(bundle, open(out_dir / "raw_method_scores.pkl", "wb"))
    csv_path = out_dir / "raw_method_scores.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["commit_index", "commit_id", "y", "graph", "method", "score"])
        for g, mscores in raw_scores.items():
            for m in METHODS:
                arr = mscores[m]
                for i in range(N):
                    v = arr[i]
                    w.writerow([i, cids[i], int(y[i]), g, m,
                                "" if v != v else f"{float(v):.6f}"])  # v!=v -> NaN
    print(f"saved raw per-method scores -> {out_dir/'raw_method_scores.pkl'} "
          f"(+ {csv_path.name}; warmup W={W}, N={N})")


if __name__ == "__main__":
    main()
