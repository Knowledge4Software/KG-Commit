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

OUT = Path(__file__).resolve().parent.parent / "outputs"
NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")
REFIT_EMB = 5; DW_DIM = 64; KGE_DIM = 32; ROLL = 800

GRAPHS = ["core", "ast", "cfg", "dfg", "pdg", "final"]
GRAPH_NAME = {"core": "Core", "ast": "Core+AST", "cfg": "Core+CFG",
              "dfg": "Core+DFG", "pdg": "Core+PDG", "final": "Core+AST+CSTG"}
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
    for g, lbl in [("cfg", "CFGNode"), ("dfg", "DFGNode"), ("pdg", "PDGNode")]:
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
        if g in ("ast", "cfg", "dfg", "pdg"):
            for t, n in tok[g].get(c, ()):     raw.append((i, f"T{g}:" + t, float(n), "T"))
        elif g == "final":
            for t, n in tok["ast"].get(c, ()): raw.append((i, "Tast:" + t, float(n), "T"))
            for t, n in cstg.get(c, ()):       raw.append((i, "S:" + t, float(n), "S"))
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
                      traj=metric_traj(yt, p, W))
    return out, pred      # pred: full-length per-commit scores per method


def metric_traj(y_ev, p_ev, warmup, roll=ROLL, step=BLOCK):
    """Rolling 7-metric online-evaluation trajectory (x = commit index)."""
    yhat = online_decisions(p_ev, y_ev)
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--graphs", nargs="*", default=GRAPHS)
    args = ap.parse_args()
    commits, cids, y, files, devs, tok, cstg = load_all()

    results = {"meta": dict(graphs=GRAPHS, methods=METHODS, metrics=M7,
                            warmup=int(len(cids) * WARMUP_FRAC), N=len(cids))}
    for g in args.graphs:
        print(f"\n=== graph: {GRAPH_NAME[g]} ===")
        res, _ = run_graph(g, cids, y, files, devs, tok, cstg, args.limit)
        results[g] = res
        for m in METHODS:
            r = res[m]["metrics"]
            print(f"  {m:<4} BF1={r['Buggy_F1']:.3f} MacroF1={r['Macro_F1']:.3f} "
                  f"GM={r['G_Mean']:.3f} AUC={r['AUC']:.3f}")
    OUT.mkdir(exist_ok=True)
    pickle.dump(results, open(OUT / "final_experiments_results.pkl", "wb"))
    print(f"\nsaved -> {OUT/'final_experiments_results.pkl'}")


if __name__ == "__main__":
    main()
