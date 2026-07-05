"""
v4 RQ "Which subgraph is best?" -- online prequential comparison.

Runs the IDENTICAL online / prequential protocol for every structural-subgraph
variant and reports the KG-native results, so the ONLY thing that differs across
rows is the subgraph feeding the change-token stream:

  V1  none (core only)     tokens = {}          -> M + R + P
  V2a cfg   :CFGNode        \
  V2b dfg   :DFGNode         > tokens = {edge}:{type} for that layer  -> M+T+R+P
  V2c pdg   :PDGNode        /
  V2d seq   :SEQNode        /
  V3  ast   :ASTNode  (ast_type)   incumbent -- expected best

Text features (X) and the Semantic-Text subgraph (G/CSTG) are NEVER enabled here
-- this stage isolates the structural subgraph. Core (M metrics, R relational
priors, P PPR), the classifier, warmup, and block schedule are held constant
(reused from online_infer), so the comparison is fair.

Headline metric = the Fusion (M+T+R+P) prequential score; we also print the
isolated structural contributions (T-only TF-IDF, P-only PPR) per variant.

Run:  python inference/run_subgraph_rq.py
Out:  outputs/subgraph_rq_results.pkl  + printed table
"""
import numpy as np, scipy.sparse as sp, pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings; from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score)
from advanced_infer import load_kg, ppr, METRICS
from online_infer import (build_incidence, online_priors, WARMUP_FRAC, BLOCK,
                          HASH_DIM)
from online_jit import final_metrics, cum_metrics, online_decisions

OUT = Path(__file__).resolve().parent.parent / "outputs"
REFIT_EVERY = 3
SVD_DIM = 64
SVD_REFRESH = 5     # refit the KG (SVD/LSA) embedding every N blocks
ROLL = 800          # rolling-window size for the trajectory plot

# the inference methods compared per subgraph. M (JIT metrics) and R (relational
# priors) do NOT use the structural tokens, so they are subgraph-INDEPENDENT
# references; T (structural TF-IDF), P (PPR) and E (KG embedding) all consume the
# subgraph's change-tokens and therefore vary; Fusion = M+T+R+P.
METHOD_ORDER = ["M", "R", "T", "P", "E", "Fusion"]
METHOD_NAME = {
    "M": "JIT metrics (LR)", "R": "Relational priors (wvRN)",
    "T": "Structural TF-IDF", "P": "Personalized PageRank",
    "E": "KG embedding (SVD/LSA)", "Fusion": "Fusion (M+T+R+P)"}
STRUCTURAL = {"T", "P", "E", "Fusion"}   # methods that depend on the subgraph

# variant -> (neo4j node label, type property).  None label = core-only.
VARIANTS = [
    ("V1_none", "none (core only)",       None,      None),
    ("V2a_cfg", "CFG delta",              "CFGNode", "atype"),
    ("V2b_dfg", "DFG (def-use) delta",    "DFGNode", "atype"),
    ("V2c_pdg", "PDG/CPG delta",          "PDGNode", "atype"),
    ("V2d_seq", "token/stmt-seq delta",   "SEQNode", "atype"),
    ("V3_ast",  "AST delta (incumbent)",  "ASTNode", "ast_type"),
]


def _lr(sparse=False):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              solver="liblinear" if sparse else "lbfgs")


STREAM_METRICS = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]

def metric_trajectory(y_ev, p_ev, warmup, roll=ROLL, step=BLOCK):
    """Rolling-window trend of the seven headline metrics over the online stream.
    x = commit index in the chronological stream (warmup + window end); threshold-
    based metrics use the leakage-free online-tuned decisions restricted to each
    window; AUC is threshold-free."""
    p_ev = np.clip(np.asarray(p_ev, float), 0, 1); y_ev = np.asarray(y_ev)
    yhat = online_decisions(p_ev, y_ev)
    traj = {"idx": []}; traj.update({m: [] for m in STREAM_METRICS})
    n = len(y_ev)
    for j in range(step, n + 1, step):
        lo = max(0, j - roll)
        ys, ps, yh = y_ev[lo:j], p_ev[lo:j], yhat[lo:j]
        rec = recall_score(ys, yh, pos_label=1, zero_division=0)
        spec = recall_score(ys, yh, pos_label=0, zero_division=0)
        traj["idx"].append(warmup + j)
        traj["Precision"].append(precision_score(ys, yh, pos_label=1, zero_division=0))
        traj["Recall"].append(rec)
        traj["Macro_F1"].append(f1_score(ys, yh, average="macro", zero_division=0))
        traj["Buggy_F1"].append(f1_score(ys, yh, pos_label=1, zero_division=0))
        traj["G_Mean"].append(float(np.sqrt(max(rec, 0) * max(spec, 0))))
        traj["AUC"].append(roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else np.nan)
        traj["ACC"].append(float((yh == ys).mean()))
    return traj


def run_variant(node_label, type_prop):
    commits, tokens, files, devs = load_kg(node_label, type_prop)
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y = np.array([commits[c]["buggy"] for c in cids]); Nc = len(cids)
    Xm = np.array([[commits[c][m] for m in METRICS] for c in cids], float)
    docs = [" ".join((t + " ") * int(min(n, 20)) for t, n in tokens.get(c, ())) for c in cids]
    Xh = HashingVectorizer(n_features=HASH_DIM, alternate_sign=False,
                           token_pattern=r"[^\s]+").transform(docs)
    Xp, _ = online_priors(cids, y, files, devs)
    C, P, _, N = build_incidence(cids, tokens, files, devs)

    w = int(Nc * WARMUP_FRAC)
    Xms = StandardScaler().fit(Xm[:w]).transform(Xm)
    ppr_full = np.zeros(Nc)

    def fuse_dense(idx, sc):
        return sc.transform(np.hstack([Xms[idx], Xp[idx], ppr_full[idx][:, None]]))
    def fuse(idx, sc):
        return sp.hstack([sp.csr_matrix(fuse_dense(idx, sc)), Xh[idx]]).tocsr()

    pred = {m: np.full(Nc, np.nan) for m in METHOD_ORDER}
    traj = {"idx": [], "ROC_AUC": [], "PR_AUC": []}   # rolling-window Fusion trajectory
    svd_dim = min(SVD_DIM, C.shape[1] - 1) if C.shape[1] > 2 else 1
    fus = tf = mlr = svd = svd_clf = None; sc_f = None; E = None; blk = 0; i = w
    while i < Nc:
        j = min(Nc, i + BLOCK); idx = np.arange(i, j); past = np.arange(i)
        # PPR from strictly-past seeds (past-only -> leakage-free)
        bs = past[y[past] == 1]; gs = past[y[past] == 0]
        rb = ppr(P, list(bs), N); rg = ppr(P, list(gs), N)
        ppr_full[idx] = rb[idx] / (rb[idx] + rg[idx] + 1e-12)
        if blk % REFIT_EVERY == 0:
            sc_f = StandardScaler().fit(np.hstack([Xms[past], Xp[past], ppr_full[past][:, None]]))
            fus = _lr(sparse=True).fit(fuse(past, sc_f), y[past])
            tf  = _lr(sparse=True).fit(Xh[past], y[past])
            mlr = _lr().fit(Xms[past], y[past])
        if blk % SVD_REFRESH == 0:                       # KG embedding (leakage-free basis)
            svd = TruncatedSVD(n_components=svd_dim, random_state=0).fit(C[past])
            E = svd.transform(C); svd_clf = _lr().fit(E[past], y[past])
        # --- per-method block predictions (past-only models) ---
        pred["M"][idx] = mlr.predict_proba(Xms[idx])[:, 1]
        pred["R"][idx] = 0.5 * Xp[idx, 2] + 0.3 * Xp[idx, 1] + 0.2 * Xp[idx, 0]  # wvRN blend
        pred["T"][idx] = tf.predict_proba(Xh[idx])[:, 1]
        pred["P"][idx] = ppr_full[idx]
        pred["E"][idx] = svd_clf.predict_proba(E[idx])[:, 1]
        pred["Fusion"][idx] = fus.predict_proba(fuse(idx, sc_f))[:, 1]
        # rolling-window Fusion metrics over the last ROLL scored commits
        lo = max(w, j - ROLL); sl = np.arange(lo, j)
        cm = cum_metrics(y[sl], np.nan_to_num(pred["Fusion"][sl], nan=y[:i].mean()))
        traj["idx"].append(j); traj["ROC_AUC"].append(cm["ROC_AUC"]); traj["PR_AUC"].append(cm["PR_AUC"])
        i = j; blk += 1

    ev = np.arange(w, Nc); yt = y[ev]
    vocab = {t for c in cids for t, _ in tokens.get(c, ())}
    n_tok_commits = sum(1 for c in cids if tokens.get(c))
    methods = {m: final_metrics(yt, np.nan_to_num(pred[m][ev], nan=yt.mean()))
               for m in METHOD_ORDER}
    traj7 = metric_trajectory(yt, np.nan_to_num(pred["Fusion"][ev], nan=yt.mean()), w)
    return {
        "methods": methods,
        # backward-compatible aliases used by the tables/figures scripts
        "Fusion": methods["Fusion"], "T_only": methods["T"], "P_only": methods["P"],
        "traj": traj, "traj7": traj7,
        "n_token_types": len(vocab), "n_commits_with_tokens": n_tok_commits,
        "warmup": w, "n_eval": len(ev), "stream_bug": float(yt.mean()),
    }


def main():
    results = {}
    for vid, name, label, prop in VARIANTS:
        print(f"\n=== {vid}: {name} (label={label}) ===")
        try:
            results[vid] = dict(name=name, label=label, **run_variant(label, prop))
        except Exception as e:
            print(f"  !! {vid} failed: {type(e).__name__}: {e}")
            results[vid] = dict(name=name, label=label, error=str(e))

    ok = [v for v in VARIANTS if "error" not in results[v[0]]]
    short = {"V1_none": "Core", "V2a_cfg": "CFG", "V2b_dfg": "DFG",
             "V2c_pdg": "PDG", "V2d_seq": "Seq", "V3_ast": "AST"}
    # per-method x variant tables, one per metric
    for mk, mlabel in [("PR_AUC", "PR-AUC"), ("ROC_AUC", "ROC-AUC"),
                       ("F1_online", "F1-online")]:
        print("\n" + "=" * 78)
        print(f"INFERENCE METHOD x SUBGRAPH  --  {mlabel}   (X and G off; "
              f"* = subgraph-dependent)")
        hdr = f"{'method':<26}" + "".join(f"{short[v[0]]:>8}" for v in ok)
        print(hdr); print("-" * len(hdr))
        for m in METHOD_ORDER:
            tag = "*" if m in STRUCTURAL else " "
            row = f"{tag}{METHOD_NAME[m]:<25}"
            vals = [results[v[0]]["methods"][m][mk] for v in ok]
            best = max(vals) if m in STRUCTURAL else None
            for v, val in zip(ok, vals):
                mark = "" if best is None else ("<" if abs(val - best) < 1e-9 else " ")
                row += f"{val:>7.3f}{mark}"
            print(row)
    OUT.mkdir(exist_ok=True)
    pickle.dump(results, open(OUT / "subgraph_rq_results.pkl", "wb"))
    print(f"\nsaved -> {OUT/'subgraph_rq_results.pkl'}")


if __name__ == "__main__":
    main()
