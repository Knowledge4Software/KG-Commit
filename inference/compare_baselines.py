"""
Controlled comparison vs the within-project baselines
(baselines/WithinProj_baselines.ipynb): SAME chronological split as the
baselines -- sort groovy commits by author_date, train on the first 70%, test on
the LAST 15% (the middle 15% validation block is unused), evaluate ROC-AUC and
buggy-class F1 (0.5 threshold). This isolates the FEATURE contribution: every
model here is a balanced Logistic Regression, exactly like the baseline LR, so
differences come from the KG features, not the classifier.

Run:  python inference/compare_baselines.py
"""
import numpy as np, scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, average_precision_score
import advanced_infer as ai, online_infer as oi

OUT = Path(__file__).resolve().parent.parent / "outputs"

def lr(): return LogisticRegression(class_weight="balanced", max_iter=2000)

def evalp(y, p):
    yh = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
    return roc_auc_score(y, p), average_precision_score(y, p), f1, pr, rc

def main():
    commits, tokens, files, devs = ai.load_kg()
    cids = sorted(commits, key=lambda c: commits[c]["ts"])   # = author_date order
    y = np.array([commits[c]["buggy"] for c in cids]); n = len(cids)
    tr = np.arange(int(0.70*n)); te = np.arange(int(0.85*n), n)   # baselines' exact split
    print(f"groovy commits={n}  train={len(tr)} (first 70%)  test={len(te)} (last 15%)  "
          f"test bug-rate={y[te].mean():.3f}\n")

    Xm = np.array([[commits[c][m] for m in ai.METRICS] for c in cids], float)
    docs = [" ".join((t+" ")*int(min(k,20)) for t,k in tokens.get(c,())) for c in cids]
    Xp,_ = oi.online_priors(cids, y, files, devs)
    C, P, _, N = oi.build_incidence(cids, tokens, files, devs)
    En = np.load(OUT/f"emb_node2vec_{n}.npy"); Et = np.load(OUT/f"emb_transe_{n}.npy")
    Ed = np.load(OUT/f"emb_distmult_{n}.npy")
    Esvd = TruncatedSVD(n_components=64, random_state=0).fit_transform(C)

    rows = []
    def dense(name, X):
        sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
        m = lr().fit(Xs[tr], y[tr]); p = m.predict_proba(Xs[te])[:,1]
        rows.append((name, *evalp(y[te], p)))

    dense("Ours: JIT-metrics (LR)", Xm)
    # structural TF-IDF (fit vocab on train docs only)
    vec = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=3).fit([docs[i] for i in tr])
    Xt = vec.transform(docs)
    m = lr().fit(Xt[tr], y[tr]); p = m.predict_proba(Xt[te])[:,1]
    rows.append(("Ours: Structural TF-IDF", *evalp(y[te], p)))
    dense("Ours: node2vec", En); dense("Ours: TransE", Et); dense("Ours: DistMult", Ed)
    dense("Ours: KGE-SVD", Esvd)
    # wvRN priors (score, no fit)
    score = 0.5*Xp[:,2]+0.3*Xp[:,1]+0.2*Xp[:,0]
    rows.append(("Ours: wvRN priors", *evalp(y[te], score[te])))
    # PPR (seeds from train)
    bs = tr[y[tr]==1]; gs = tr[y[tr]==0]
    rb = oi.ppr(P, list(bs), N); rg = oi.ppr(P, list(gs), N)
    pp = rb[:n]/(rb[:n]+rg[:n]+1e-12)
    rows.append(("Ours: PPR", *evalp(y[te], pp[te])))
    # Fusion (metrics+priors+tfidf+ppr)
    sc = StandardScaler().fit(np.hstack([Xm,Xp])[tr]); dns = sc.transform(np.hstack([Xm,Xp]))
    Xf = sp.hstack([sp.csr_matrix(dns), Xt, sp.csr_matrix(pp[:,None])]).tocsr()
    m = lr().fit(Xf[tr], y[tr]); p = m.predict_proba(Xf[te])[:,1]
    rows.append(("Ours: FUSION (no text)", *evalp(y[te], p)))
    # V2: commit-message TEXT (vocab fit on train only)
    msgs = [str(commits[c].get("message","")) for c in cids]
    tvec = TfidfVectorizer(ngram_range=(1,2), min_df=3, lowercase=True).fit([msgs[i] for i in tr])
    Xtx = tvec.transform(msgs)
    m = lr().fit(Xtx[tr], y[tr]); p = m.predict_proba(Xtx[te])[:,1]
    rows.append(("Ours: Commit-text (V2)", *evalp(y[te], p)))
    # V2: Fusion + text
    Xft = sp.hstack([Xf, Xtx]).tocsr()
    m = lr().fit(Xft[tr], y[tr]); p = m.predict_proba(Xft[te])[:,1]
    rows.append(("Ours: FUSION+text (V2)", *evalp(y[te], p)))

    print(f"{'method':<28}{'ROC':>7}{'PR-AUC':>8}{'F1':>7}{'Prec':>7}{'Rec':>7}")
    print("-"*64)
    for nm,roc,prc,f1,p,rc in rows:
        print(f"{nm:<28}{roc:7.3f}{prc:8.3f}{f1:7.3f}{p:7.3f}{rc:7.3f}")
    print("\n-- baselines (same split, from baselines/results, ROC | F1) --")
    print("  JIT LR 0.698|0.232   RF 0.708|0.164   XGBoost 0.752|0.281")
    print("  JIT RF-tuned 0.729|0.336   XGBoost-tuned 0.755|0.267")
    print("  Text TF-IDF-768D: LR 0.805|0.182   RF 0.813|0.321   XGB 0.798|0.224")
    print("  Naive: Random 0.500|0.094   AllBuggy 0.500|0.133")

if __name__ == "__main__":
    main()
