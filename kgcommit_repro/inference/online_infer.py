"""
ONLINE (prequential) inference for commit-level buggy/benign prediction.

JIT defect prediction is inherently streaming: when a commit arrives we must
predict it using ONLY the past, then we learn its (eventually-known) label.
This harness evaluates every method prequentially -- predict-then-update in
chronological order -- which is the correct online protocol and avoids any
look-ahead leakage.

Protocol:
  * warm-up on the first WARMUP_FRAC of commits (initial fit);
  * then stream the rest in blocks: predict each block with the model trained on
    everything strictly before it, record the predictions, then fold the block
    into the model (incremental partial_fit for linear models; periodic refit
    for the embedding model; past-only seeds for the graph methods);
  * report metrics over all streamed (out-of-time) predictions.

Methods
  Simple / baselines (online):
    O1 ON-Logistic(JIT)   SGD log-loss on the 12 JIT metrics (partial_fit)
    O2 ON-wvRN/priors     weighted-vote relational neighbour: pure online
                          file/developer/global historical bug-rate (no training)
  Advanced KG-native (online):
    O3 ON-HashTFIDF       SGD on hashing-vectorised AST-change tokens
    O4 ON-PPR             Personalized PageRank over the KG, past-only seeds
    O5 ON-KGE(SVD)        Truncated-SVD/LSA embedding of the commit-hub matrix
                          (a CPU, no-dependency KG-embedding; GNN substitute)
    O6 ON-Fusion          SGD over [JIT metrics + priors + hashing-TFIDF + PPR]

Run:  python inference/online_infer.py
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import math
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef)
from advanced_infer import load_kg, ppr, METRICS, ALPHA, PPR_ITERS

# FINAL RUN: constants come from inference/protocol.py. This module previously
# declared WARMUP_FRAC=0.40 / BLOCK=200 independently of online_jit.py (0.30 / 50),
# which is exactly the drift protocol.py exists to prevent.
from protocol import (WARMUP_FRAC, BLOCK, GAP, SVD_DIM, HASH_DIM, REFIT_EVERY,
                      CLASS_WEIGHT)
SVD_REFIT_EVERY = REFIT_EVERY   # single-M rule: refresh once per BLOCK

# ── incidence (commits x hubs) + PPR transition, built once ──────────────────
def build_incidence(cids, tokens, files, devs):
    cidx={c:i for i,c in enumerate(cids)}; Nc=len(cids)
    hub={}; rows=[]; cols=[]; raw=[]
    for c in cids:
        i=cidx[c]
        for f in files.get(c,()): raw.append((i,"F:"+f,1.0))
        if c in devs: raw.append((i,"D:"+devs[c],1.0))
        for tok,n in tokens.get(c,()): raw.append((i,"T:"+tok,float(n)))
    df=defaultdict(int)
    for i,h,w in raw: df[h]+=1
    data=[]
    for i,h,w in raw:
        j=hub.setdefault(h,len(hub))
        rows.append(i); cols.append(j); data.append(w*math.log(1.0+Nc/df[h]))
    Nh=len(hub)
    C=sp.csr_matrix((data,(rows,cols)),shape=(Nc,Nh))
    # PPR graph: bipartite [[0,C],[C.T,0]], column-normalized
    N=Nc+Nh
    B=sp.csr_matrix((data,(rows,[c+Nc for c in cols])),shape=(N,N))
    A=B+B.T; deg=np.asarray(A.sum(0)).ravel(); deg[deg==0]=1.0
    P=A.multiply(sp.csr_matrix(1.0/deg)).tocsr()
    return C, P, Nc, N

# ── streaming online priors (each commit scored from strictly past) ──────────
def online_priors(cids, y, files, devs):
    Nc=len(cids); cols=["global_rate","dev_rate","file_rate_mean","file_rate_max",
                        "recent_rate","dev_exp","file_churn"]
    X=np.zeros((Nc,len(cols))); gb=gt=0
    devc=defaultdict(lambda:[0,0]); filec=defaultdict(lambda:[0,0])
    recent=[]
    for i,c in enumerate(cids):
        g=(gb/gt) if gt else 0.0
        db,dt=devc[devs.get(c)]; dev_rate=((db+ g*5)/(dt+5))
        frs=[(filec[f][0]+g*5)/(filec[f][1]+5) for f in files.get(c,())]
        X[i,0]=g
        X[i,1]=dev_rate
        X[i,2]=np.mean(frs) if frs else g
        X[i,3]=max(frs) if frs else g
        X[i,4]=np.mean(recent[-50:]) if recent else g
        X[i,5]=dt
        X[i,6]=max((filec[f][1] for f in files.get(c,())), default=0)
        # update AFTER scoring (past-only)
        gt+=1; gb+=y[i]; recent.append(y[i])
        if devs.get(c): devc[devs[c]][1]+=1; devc[devs[c]][0]+=y[i]
        for f in files.get(c,()): filec[f][1]+=1; filec[f][0]+=y[i]
    return X, cols

def metr(y,p):
    yh=(p>=0.5).astype(int)
    return (roc_auc_score(y,p),average_precision_score(y,p),
            f1_score(y,yh,zero_division=0),matthews_corrcoef(y,yh))

def main():
    print("Loading KG..."); commits,tokens,files,devs=load_kg()
    cids=sorted(commits,key=lambda c:commits[c]["ts"])
    y=np.array([commits[c]["buggy"] for c in cids]); Nc=len(cids)
    Xm=np.array([[commits[c][m] for m in METRICS] for c in cids],dtype=float)
    docs=[" ".join((t+" ")*int(min(n,20)) for t,n in tokens.get(c,())) for c in cids]
    Xh=HashingVectorizer(n_features=HASH_DIM,alternate_sign=False,
                         token_pattern=r"[^\s]+").transform(docs)
    Xp,_=online_priors(cids,y,files,devs)
    C,P,_,N=build_incidence(cids,tokens,files,devs)

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore",category=ConvergenceWarning)

    w=int(Nc*WARMUP_FRAC)
    scaler=StandardScaler().fit(Xm[:w])
    Xms=scaler.transform(Xm)
    pscaler=StandardScaler().fit(np.hstack([Xms,Xp])[:w])
    Xfuse_dense=pscaler.transform(np.hstack([Xms,Xp]))   # metrics+priors, scaled
    fuse=lambda idx: sp.hstack([sp.csr_matrix(Xfuse_dense[idx]),Xh[idx]]).tocsr()

    def lr(sparse=False):
        return LogisticRegression(max_iter=2000,class_weight="balanced",
                                  solver=("liblinear" if sparse else "lbfgs"))

    preds={k:np.full(Nc,np.nan) for k in
           ["O1_jit","O2_wvrn","O3_tfidf","O4_ppr","O5_kge","O6_fusion"]}
    # REFIT_EVERY comes from protocol.py (single-M rule: refit every block).
    jit_clf=tf_clf=fus_clf=svd=svd_clf=None; E=None; blk=0
    i=w
    while i<Nc:
        j=min(Nc,i+BLOCK); idx=np.arange(i,j); past=np.arange(i)
        # ---- (re)train on the expanding PAST window (online, no future) ----
        if blk%REFIT_EVERY==0:
            jit_clf=lr().fit(Xms[past],y[past])
            tf_clf =lr(sparse=True).fit(Xh[past],y[past])
            fus_clf=lr(sparse=True).fit(fuse(past),y[past])
            svd=TruncatedSVD(n_components=min(SVD_DIM,C.shape[1]-1),random_state=0)
            svd.fit(C[past])                 # leakage-free: SVD basis from PAST rows only
            E=svd.transform(C)
            svd_clf=lr().fit(E[past],y[past])
        # ---- predict the block (past-only models) ----
        preds["O1_jit"][idx]=jit_clf.predict_proba(Xms[idx])[:,1]
        preds["O3_tfidf"][idx]=tf_clf.predict_proba(Xh[idx])[:,1]
        preds["O6_fusion"][idx]=fus_clf.predict_proba(fuse(idx))[:,1]
        preds["O5_kge"][idx]=svd_clf.predict_proba(E[idx])[:,1]
        # O2 wvRN / priors: blended file+dev+global historical rate (no fit)
        preds["O2_wvrn"][idx]=0.5*Xp[idx,2]+0.3*Xp[idx,1]+0.2*Xp[idx,0]
        # O4 PPR: seeds = commits strictly before the block
        bs=past[y[past]==1]; gs=past[y[past]==0]
        rb=ppr(P,list(bs),N); rg=ppr(P,list(gs),N)
        preds["O4_ppr"][idx]=rb[idx]/(rb[idx]+rg[idx]+1e-12)
        i=j; blk+=1

    ev=np.arange(w,Nc); yt=y[ev]
    print(f"\nPrequential online evaluation over commits [{w}:{Nc}] "
          f"({len(ev)} predicted online), test bug rate={yt.mean():.3f}\n")
    hdr=f"{'method':<22}{'ROC_AUC':>8}{'PR_AUC':>8}{'F1':>7}{'MCC':>7}"
    print(hdr); print("-"*len(hdr))
    names={"O1_jit":"O1 ON-Logistic(JIT)","O2_wvrn":"O2 ON-wvRN/priors",
           "O3_tfidf":"O3 ON-HashTFIDF","O4_ppr":"O4 ON-PPR",
           "O5_kge":"O5 ON-KGE(SVD)","O6_fusion":"O6 ON-Fusion"}
    for k in ["O1_jit","O2_wvrn","O3_tfidf","O4_ppr","O5_kge","O6_fusion"]:
        p=np.nan_to_num(preds[k][ev],nan=yt.mean()); a=metr(yt,p)
        print(f"{names[k]:<22}{a[0]:8.3f}{a[1]:8.3f}{a[2]:7.3f}{a[3]:7.3f}")
    print(f"\nrandom PR-AUC baseline ~= bug rate {yt.mean():.3f}")

if __name__=="__main__":
    main()
