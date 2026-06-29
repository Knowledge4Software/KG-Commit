"""
Consolidated runner: evaluate EVERY inference method on the full KG, in both
OFFLINE (time-ordered split) and ONLINE (prequential, expanding-window) modes,
returning predictions + metrics for each so a notebook can build a perfect
comparison (tables, ROC/PR curves, bar charts).

Methods (CPU, no GNN):
  JIT-metrics      classic ApacheJIT change metrics (baseline)
  wvRN-priors      online relational file/dev/global historical bug-rate
  TFIDF            structural TF-IDF over AST-change tokens
  PPR              Personalized PageRank label propagation over the KG
  KGE-SVD          Truncated-SVD/LSA embedding of the commit-hub matrix
  node2vec         random-walk embedding (gensim)
  TransE           translational KG embedding (pykeen)
  DistMult         bilinear KG embedding (pykeen)
  Fusion           early fusion of JIT + TFIDF + priors + PPR

Embeddings and the full results dict are CACHED under outputs/, so the notebook
loads instantly after the first (slow, pykeen-training) run.

API:  results = evaluate_all()   # dict: results['offline'|'online'][method] =
      {'y':array, 'p':array, 'metrics':{ROC_AUC,PR_AUC,F1,MCC}}
      results['meta'] = small per-commit frame for KG-signal plots.
"""
import os, pickle, numpy as np, scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_curve, precision_recall_curve)
import warnings; from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# NOTE: neo4j / gensim / pykeen are imported lazily inside evaluate_all() only
# when (re)computing, so a notebook can LOAD the cached results with just
# numpy / pandas / sklearn (no Neo4j connection or heavy deps required).

OUT = Path(__file__).resolve().parent.parent / "outputs"; OUT.mkdir(exist_ok=True)
RESULTS_PKL = OUT/"inference_results.pkl"
TEST_FRAC, WARMUP_FRAC, BLOCK, REFIT = 0.30, 0.40, 200, 3

def _m(y, p):
    yh=(p>=0.5).astype(int)
    return dict(ROC_AUC=float(roc_auc_score(y,p)), PR_AUC=float(average_precision_score(y,p)),
                F1=float(f1_score(y,yh,zero_division=0)), MCC=float(matthews_corrcoef(y,yh)))

def _lr(sparse=False):
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              solver="liblinear" if sparse else "lbfgs")

def _cache_emb(name, fn, Nc):
    f=OUT/f"emb_{name}_{Nc}.npy"
    if f.exists(): return np.load(f)
    E=fn(); np.save(f, E); return E

def evaluate_all(force=False):
    if RESULTS_PKL.exists() and not force:
        return pickle.load(open(RESULTS_PKL,"rb"))

    # heavy deps only needed when (re)computing from the live KG
    import advanced_infer as ai, online_infer as oi, kge_infer as ke

    # ---- load KG ----
    commits, tokens, files, devs = ai.load_kg()
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y  = np.array([commits[c]["buggy"] for c in cids]); Nc=len(cids)
    Xm = np.array([[commits[c][m] for m in ai.METRICS] for c in cids], float)
    docs=[" ".join((t+" ")*int(min(n,20)) for t,n in tokens.get(c,())) for c in cids]
    Xh = HashingVectorizer(n_features=2**18, alternate_sign=False,
                           token_pattern=r"[^\s]+").transform(docs)
    Xp,_ = oi.online_priors(cids, y, files, devs)
    C, P, _, N = oi.build_incidence(cids, tokens, files, devs)
    # embeddings (cached)
    kc, kedges = ke.load()                      # edges format for embeddings
    kcids = sorted(kc, key=lambda c: kc[c]["ts"])
    assert kcids == cids, "commit ordering mismatch"
    En = _cache_emb("node2vec", lambda: ke.node2vec_embed(cids, kedges), Nc)
    Et = _cache_emb("transe",   lambda: ke.kge_embed(cids, kedges, "TransE"), Nc)
    Ed = _cache_emb("distmult", lambda: ke.kge_embed(cids, kedges, "DistMult"), Nc)

    cut=int(Nc*(1-TEST_FRAC)); w=int(Nc*WARMUP_FRAC)
    tr=np.arange(cut); te=np.arange(cut,Nc); ev=np.arange(w,Nc)

    off={}; on={}

    # ---------- helpers ----------
    def off_dense(name, X):
        sc=StandardScaler().fit(X[tr]); Xs=sc.transform(X)
        clf=_lr().fit(Xs[tr], y[tr]); p=clf.predict_proba(Xs[te])[:,1]
        off[name]={"y":y[te],"p":p,"metrics":_m(y[te],p)}
    def on_dense(name, X):
        preds=np.full(Nc,np.nan); clf=None; blk=0; i=w
        while i<Nc:
            j=min(Nc,i+BLOCK); idx=np.arange(i,j); past=np.arange(i)
            if blk%REFIT==0:
                sc=StandardScaler().fit(X[past]); Xs=sc.transform(X)
                clf=_lr().fit(Xs[past], y[past])
            preds[idx]=clf.predict_proba(Xs[idx])[:,1]; i=j; blk+=1
        p=preds[ev]; on[name]={"y":y[ev],"p":p,"metrics":_m(y[ev],p)}

    # ---------- JIT metrics ----------
    off_dense("JIT-metrics", Xm); on_dense("JIT-metrics", Xm)
    # ---------- embeddings ----------
    for nm,E in [("node2vec",En),("TransE",Et),("DistMult",Ed),("KGE-SVD",
                 TruncatedSVD(n_components=64,random_state=0).fit_transform(C))]:
        off_dense(nm, E); on_dense(nm, E)
    # ---------- wvRN priors (blended rate; no training) ----------
    score_pr=0.5*Xp[:,2]+0.3*Xp[:,1]+0.2*Xp[:,0]
    off["wvRN-priors"]={"y":y[te],"p":score_pr[te],"metrics":_m(y[te],score_pr[te])}
    on ["wvRN-priors"]={"y":y[ev],"p":score_pr[ev],"metrics":_m(y[ev],score_pr[ev])}
    # ---------- structural TF-IDF ----------
    vec=TfidfVectorizer(token_pattern=r"[^\s]+",min_df=3).fit([docs[i] for i in tr])
    Xt=vec.transform(docs)
    clf=_lr(sparse=True).fit(Xt[tr],y[tr]); p=clf.predict_proba(Xt[te])[:,1]
    off["TFIDF"]={"y":y[te],"p":p,"metrics":_m(y[te],p)}
    on_dense_sparse=None
    # online tfidf via hashing (stateless) expanding-window
    def on_sparse(name, X):
        preds=np.full(Nc,np.nan); clf=None; blk=0; i=w
        while i<Nc:
            j=min(Nc,i+BLOCK); idx=np.arange(i,j); past=np.arange(i)
            if blk%REFIT==0: clf=_lr(sparse=True).fit(X[past],y[past])
            preds[idx]=clf.predict_proba(X[idx])[:,1]; i=j; blk+=1
        p=preds[ev]; on[name]={"y":y[ev],"p":p,"metrics":_m(y[ev],p)}
    on_sparse("TFIDF", Xh)
    # ---------- PPR (leakage-safe) ----------
    def ppr_scores(seed_split):
        sc=np.full(Nc,np.nan)
        if seed_split=="offline":
            bs=tr[y[tr]==1]; gs=tr[y[tr]==0]
            rb=oi.ppr(P,list(bs),N); rg=oi.ppr(P,list(gs),N)
            sc=rb[:Nc]/(rb[:Nc]+rg[:Nc]+1e-12); return sc[te], y[te]
        # online: chronological folds, seeds from earlier folds only
        folds=np.array_split(np.arange(Nc),5); seen=[]
        for fold in folds:
            if seen:
                past=np.concatenate(seen)
                bs=[i for i in past if y[i]==1]; gs=[i for i in past if y[i]==0]
                rb=oi.ppr(P,bs,N); rg=oi.ppr(P,gs,N)
                sc[fold]=rb[fold]/(rb[fold]+rg[fold]+1e-12)
            else: sc[fold]=y[np.array(seen,int)].mean() if seen else 0.0
            seen.append(fold)
        sc=np.nan_to_num(sc,nan=float(y[tr].mean())); return sc[ev], y[ev]
    ppr_off_p,ppr_off_y=ppr_scores("offline"); off["PPR"]={"y":ppr_off_y,"p":ppr_off_p,"metrics":_m(ppr_off_y,ppr_off_p)}
    ppr_on_p,ppr_on_y=ppr_scores("online");   on ["PPR"]={"y":ppr_on_y,"p":ppr_on_p,"metrics":_m(ppr_on_y,ppr_on_p)}
    # ---------- Fusion ----------
    sc=StandardScaler().fit(np.hstack([Xm,Xp])[tr]); dense=sc.transform(np.hstack([Xm,Xp]))
    ppr_full=np.full(Nc,np.nan); ppr_full[te]=ppr_off_p   # offline ppr feature
    Xfo=sp.hstack([sp.csr_matrix(dense),Xt,sp.csr_matrix(np.nan_to_num(ppr_full,nan=float(y[tr].mean()))[:,None])]).tocsr()
    clf=_lr(sparse=True).fit(Xfo[tr],y[tr]); p=clf.predict_proba(Xfo[te])[:,1]
    off["Fusion"]={"y":y[te],"p":p,"metrics":_m(y[te],p)}
    # online fusion: dense+hashing+priors via expanding window (+online ppr score column)
    ppr_on_full=np.full(Nc,np.nan)
    folds=np.array_split(np.arange(Nc),5); seen=[]
    for fold in folds:
        if seen:
            past=np.concatenate(seen); bs=[i for i in past if y[i]==1]; gs=[i for i in past if y[i]==0]
            rb=oi.ppr(P,bs,N); rg=oi.ppr(P,gs,N); ppr_on_full[fold]=rb[fold]/(rb[fold]+rg[fold]+1e-12)
        else: ppr_on_full[fold]=0.0
        seen.append(fold)
    ppr_on_full=np.nan_to_num(ppr_on_full,nan=float(y[tr].mean()))
    Xfdense=np.hstack([dense, ppr_on_full[:,None]])
    preds=np.full(Nc,np.nan); clf=None; blk=0; i=w
    while i<Nc:
        j=min(Nc,i+BLOCK); idx=np.arange(i,j); past=np.arange(i)
        if blk%REFIT==0:
            scf=StandardScaler().fit(Xfdense[past])
            Xall=sp.hstack([sp.csr_matrix(scf.transform(Xfdense)),Xh]).tocsr()
            clf=_lr(sparse=True).fit(Xall[past],y[past])
        preds[idx]=clf.predict_proba(Xall[idx])[:,1]; i=j; blk+=1
    p=preds[ev]; on["Fusion"]={"y":y[ev],"p":p,"metrics":_m(y[ev],p)}

    # ---------- meta for KG-signal plots ----------
    delta_total=np.array([sum(n for _,n in tokens.get(c,())) for c in cids])
    meta=dict(buggy=y, author_ts=np.array([commits[c]["ts"] for c in cids]),
              delta_total=delta_total,
              la=np.array([commits[c]["la"] for c in cids]),
              ent=np.array([commits[c]["ent"] for c in cids]))
    results=dict(offline=off, online=on, meta=meta,
                 split=dict(Nc=Nc,cut=int(cut),warmup=int(w),
                            test_bug=float(y[te].mean()),online_bug=float(y[ev].mean())))
    pickle.dump(results, open(RESULTS_PKL,"wb"))
    return results

if __name__=="__main__":
    r=evaluate_all(force=True)
    print("\n== OFFLINE ==");  [print(f"  {k:13} ROC={v['metrics']['ROC_AUC']:.3f} PR={v['metrics']['PR_AUC']:.3f} F1={v['metrics']['F1']:.3f} MCC={v['metrics']['MCC']:.3f}") for k,v in r['offline'].items()]
    print("== ONLINE ==");    [print(f"  {k:13} ROC={v['metrics']['ROC_AUC']:.3f} PR={v['metrics']['PR_AUC']:.3f} F1={v['metrics']['F1']:.3f} MCC={v['metrics']['MCC']:.3f}") for k,v in r['online'].items()]
    print(f"\nsaved -> {RESULTS_PKL}")
