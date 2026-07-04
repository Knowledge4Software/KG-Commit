"""
ONLINE / prequential Commit Semantic-Text Graph.

KG-Commit is a JIT, commit-level framework: growth, modelling and prediction all
happen ONLINE at each commit arrival. This runs the CSTG under that protocol --
predict-then-grow, strictly past-only -- so the text layer's value is measured the
way it would be deployed, not in a batch split.

At each arriving commit C (in author_date order):
  1. PREDICT  C from CSTG state built ONLY from earlier commits
     - text  : hashing-vectorised graph-of-words tokens (stateless -> leakage-free)
     - prior : centrality-weighted historical bug-rate of C's terms (past counters)
  2. EVALUATE (reveal label, update running metrics)
  3. GROW     the CSTG state (term bug-rate + co-occurrence counters advance)
  4. LEARN    (SGD partial_fit for the linear models; expanding-window otherwise)

Parsing + graph-of-words centrality are COMMIT-LOCAL (no leakage) and cached.
Reports cumulative ROC/PR-AUC + online operating-point F1 + effort-aware metrics,
for JIT metrics, CSTG-text, CSTG-prior, and their online Fusion.

Run:  python inference/online_cstg.py
"""
import pickle
from pathlib import Path
import numpy as np, scipy.sparse as sp, pandas as pd
from collections import defaultdict
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import warnings; from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import effort_metrics as em
import cstg as C

ROOT = Path(__file__).resolve().parent.parent
DIFFS = ROOT / "data/apachejit/apachejit_with_diffs_rebuilt.csv"
METRICS = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
WARMUP = 0.30
BLOCK = 50
HASH_DIM = 2 ** 18


def precompute_docs(texts):
    """Per-commit (doc string of typed GoW tokens, {term: centrality}). Commit-local
    -> leakage-free; cached because graph-of-words TextRank is the slow part."""
    cache = ROOT / "outputs" / "cstg_online_docs.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    docs, cens = [], []
    for k, t in enumerate(texts):
        toks = C.commit_terms(C.parse_commit_text(t))
        seq = [term for _, term in toks]
        doc = " ".join(f"{typ}:{term}" for typ, term in toks)     # natural TF + typing
        cen = C.graph_of_words_weights(seq) if seq else {}
        docs.append(doc); cens.append(cen)
        if k % 1000 == 0: print(f"  parsed {k}/{len(texts)}")
    pickle.dump((docs, cens), open(cache, "wb"))
    return docs, cens


def online_f1(p, y, init=300, step=150):
    y = np.asarray(y); yhat = np.zeros(len(y), int); thr = 0.5
    def best_thr(yy, pp):
        P = int(yy.sum())
        if P == 0 or P == len(yy): return 0.5
        o = np.argsort(-pp); ys = yy[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
        f1 = 2 * tp / (2 * tp + fp + (P - tp) + 1e-12)
        return float(pp[o][int(np.argmax(f1))])
    for i in range(len(y)):
        yhat[i] = int(p[i] >= thr)
        if i + 1 >= init and (i + 1) % step == 0: thr = best_thr(y[:i+1], p[:i+1])
    return float(f1_score(y, yhat, zero_division=0))


def final_metrics(y, p, eff):
    out = dict(ROC=float("nan"), PR=float("nan"))
    if len(np.unique(y)) > 1:
        out["ROC"] = roc_auc_score(y, p); out["PR"] = average_precision_score(y, p)
    out["F1@.5"] = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
    out["F1_online"] = online_f1(np.asarray(p), np.asarray(y))
    out["Popt"] = em.popt(y, p, eff); out["ACC20"] = em.recall_at_effort(y, p, eff, 0.20)
    return out


def main():
    df = pd.read_csv(DIFFS)
    df = df[df["project"] == "apache/groovy"].sort_values("author_date").reset_index(drop=True)
    y = df["buggy"].astype(int).to_numpy(); n = len(df)
    eff = (df["la"] + df["ld"]).to_numpy(float)
    Xm = df[METRICS].fillna(0).to_numpy(float)
    print(f"parsing {n} commit texts (cached)..."); docs, cens = precompute_docs(df["diff_text"].astype(str).tolist())

    W = int(n * WARMUP)
    Xh = HashingVectorizer(n_features=HASH_DIM, alternate_sign=False).transform(docs)   # stateless
    scaler = StandardScaler().fit(Xm[:W]); Xms = scaler.transform(Xm)

    # incremental term-risk counters (THE online graph-native signal)
    tb = defaultdict(float); tt = defaultdict(float); gb = gt = 0.0
    prior = np.zeros(n)
    def prior_of(i):
        cen = cens[i]; g = (gb / gt) if gt else 0.0
        if not cen: return g
        num = den = 0.0
        for term, c in cen.items():
            r = (tb[term] + g * 5) / (tt[term] + 5)
            num += c * r; den += c
        return num / den if den else g
    def grow(i):
        nonlocal gb, gt
        gt += 1; gb += y[i]
        for term in cens[i]:
            tt[term] += 1; tb[term] += y[i]
    for i in range(W):
        prior[i] = prior_of(i); grow(i)

    cw = {0: 1.0, 1: float((y[:W] == 0).sum() / max(1, (y[:W] == 1).sum()))}
    sgd_jit = SGDClassifier(loss="log_loss", class_weight=cw, random_state=0).partial_fit(Xms[:W], y[:W], classes=[0, 1])
    sgd_txt = SGDClassifier(loss="log_loss", class_weight=cw, random_state=0).partial_fit(Xh[:W], y[:W], classes=[0, 1])
    def fuse(idx): return sp.hstack([sp.csr_matrix(Xms[idx]), Xh[idx], sp.csr_matrix(prior[idx][:, None])]).tocsr()
    lrf = LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")
    fus = lrf.fit(fuse(np.arange(W)), y[:W])

    preds = {k: np.full(n, np.nan) for k in ["JIT", "CSTG_text", "CSTG_prior", "Fusion"]}
    i = W; blk = 0
    while i < n:
        j = min(n, i + BLOCK); idx = np.arange(i, j)
        for k in idx: prior[k] = prior_of(k)                    # past-only prior
        preds["JIT"][idx] = sgd_jit.predict_proba(Xms[idx])[:, 1]
        preds["CSTG_text"][idx] = sgd_txt.predict_proba(Xh[idx])[:, 1]
        preds["CSTG_prior"][idx] = prior[idx]
        preds["Fusion"][idx] = fus.predict_proba(fuse(idx))[:, 1]
        for k in idx: grow(k)                                   # GROW state
        sgd_jit.partial_fit(Xms[idx], y[idx]); sgd_txt.partial_fit(Xh[idx], y[idx])
        if blk % 4 == 0:
            fus = lrf.fit(fuse(np.arange(j)), y[:j])            # expanding-window refit
        i = j; blk += 1
        if blk % 20 == 0: print(f"  streamed {j}/{n}")

    ev = np.arange(W, n); ye = y[ev]; ee = eff[ev]
    print(f"\nprequential over commits [{W}:{n}] ({n-W} scored, bug-rate {ye.mean():.3f})\n")
    print(f"{'method':<14}{'ROC':>7}{'PR':>7}{'F1_on':>8}{'F1@.5':>7}{'Popt':>7}{'ACC20':>7}")
    print("-" * 57)
    res = {}
    for k in preds:
        p = np.nan_to_num(preds[k][ev], nan=ye.mean()); m = final_metrics(ye, p, ee); res[k] = m
        print(f"{k:<14}{m['ROC']:7.3f}{m['PR']:7.3f}{m['F1_online']:8.3f}{m['F1@.5']:7.3f}{m['Popt']:7.3f}{m['ACC20']:7.3f}")
    pickle.dump(res, open(ROOT / "outputs" / "online_cstg_results.pkl", "wb"))
    print("\nsaved -> outputs/online_cstg_results.pkl")


if __name__ == "__main__":
    main()
