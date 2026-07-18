"""
Advanced, CPU-only, GNN-free KG inference for buggy/benign commit classification.

Three graph-native methods (all light, no GPU):

  M1. Structural TF-IDF
      Each commit = a "document" of AST-change tokens "{edge}:{ast_type}"
      (e.g. ADDS:MethodInvocation, REMOVES:IfStatement). TF-IDF -> Logistic
      Regression. Learns which kinds of structural edits predict bugs.

  M2. Personalized PageRank (PPR) label propagation over the heterogeneous KG
      Build the bipartite graph  Commit <-> {File, Developer, changed-AST-type}
      (meta-paths C-File-C, C-Dev-C, C-Type-C, IDF-weighted hubs). Run PPR
      seeded from TRAIN buggy commits and (separately) TRAIN benign commits;
      score a test commit by its relative proximity to buggy vs benign history.
      Transductive, leakage-free (only train labels seed the walk).

  M3. Stacking
      Combine the PPR score + TF-IDF probability + the 12 JIT metrics in a
      HistGradientBoosting classifier.

Time-ordered split (train=older, test=newer). Reports ROC-AUC / PR-AUC / F1 / MCC
and compares against the metrics-only baseline.

Run:  python inference/advanced_infer.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
import scipy.sparse as sp
# neo4j is imported lazily inside load_kg() so modules that only reuse the pure
# helpers (or cached data) can be imported without a Neo4j install.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef)

import _kgc_paths  # noqa: F401
from config.project_config import NEO4J_URI, NEO4J_AUTH
METRICS=["la","ld","nf","nd","ns","ent","ndev","age","nuc","aexp","arexp","asexp"]
TEST_FRAC=0.30
ALPHA=0.85          # PPR damping
PPR_ITERS=60

# ── pull everything we need from the KG in a few passes ──────────────────────
def load_kg(node_label="ASTNode", type_prop="ast_type"):
    """Load core commit/file/dev graph + per-commit structural change-tokens.

    The token stream is the ONLY channel through which the structural subgraph
    reaches inference, so it is parametrized: the AST layer (default) reads
    :ASTNode.ast_type; the v4 alternative subgraphs read :CFGNode/:DFGNode/
    :PDGNode/:SEQNode.atype. Pass node_label=None for the 'no subgraph' (core-
    only) variant -> empty token stream."""
    from neo4j import GraphDatabase
    d=GraphDatabase.driver(NEO4J_URI,auth=NEO4J_AUTH)
    def q(cypher):                       # execute_read auto-retries transient errors
        with d.session(default_access_mode="READ") as s:
            return s.execute_read(lambda tx: [r.data() for r in tx.run(cypher)])
    commits={}
    for r in q(f"""MATCH (c:Commit {{in_jit:true}})
        RETURN c.id AS id, c.author_ts AS ts, c.buggy AS b, c.message AS msg,
               {", ".join(f"c.{m} AS {m}" for m in METRICS)}"""):
        commits[r["id"]]={"ts":r["ts"] or 0,"buggy":int(bool(r["b"])),
                          "message":(r["msg"] or ""),
                          **{m:(r[m] if r[m] is not None else 0.0) for m in METRICS}}
    print(f"  commits: {len(commits)}")
    tokens=defaultdict(list)
    if node_label:
        for r in q(f"""MATCH (c:Commit {{in_jit:true}})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:{node_label})
            RETURN c.id AS id, type(r)+':'+coalesce(a.{type_prop},'?') AS tok, count(*) AS n"""):
            tokens[r["id"]].append((r["tok"], r["n"]))
    files=defaultdict(list)
    for r in q("""MATCH (c:Commit {in_jit:true})-[:MODIFIED|ADDED|DELETED]->(f:File)
        RETURN c.id AS id, f.id AS f"""):
        files[r["id"]].append(r["f"])
    devs={}
    for r in q("""MATCH (c:Commit {in_jit:true})-[:AUTHORED_BY]->(dv:Developer)
        RETURN c.id AS id, dv.id AS d"""):
        devs[r["id"]]=r["d"]
    d.close()
    return commits, tokens, files, devs

# ── PPR over the heterogeneous bipartite KG ──────────────────────────────────
def build_ppr_graph(cids, tokens, files, devs):
    """Bipartite incidence: commits x hubs (files, devs, changed AST types), IDF-weighted."""
    cidx={c:i for i,c in enumerate(cids)}; Nc=len(cids)
    hub_id={}; rows=[]; cols=[]; vals=[]
    def hub(name):
        return hub_id.setdefault(name, len(hub_id))
    # collect document-frequency for IDF on each hub
    raw=[]   # (commit_idx, hubname, weight)
    for c in cids:
        i=cidx[c]
        for f in files.get(c,()): raw.append((i,"F:"+f,1.0))
        if c in devs: raw.append((i,"D:"+devs[c],1.0))
        for tok,n in tokens.get(c,()): raw.append((i,"T:"+tok, float(n)))
    df=defaultdict(int)
    for i,h,w in raw: df[h]+=1
    import math
    for i,h,w in raw:
        idf=math.log(1.0+Nc/df[h])
        j=hub(h); rows.append(i); cols.append(Nc+j); vals.append(w*idf)
    Nh=len(hub_id); N=Nc+Nh
    B=sp.csr_matrix((vals,(rows,cols)),shape=(N,N))
    A=B+B.T                                   # symmetric bipartite
    deg=np.asarray(A.sum(axis=0)).ravel(); deg[deg==0]=1.0
    P=A.multiply(sp.csr_matrix(1.0/deg))      # column-normalized transition
    return P.tocsr(), Nc

def ppr(P, seed_idx, N, alpha=ALPHA, iters=PPR_ITERS):
    s=np.zeros(N,dtype=np.float64)
    if len(seed_idx)==0: return s
    s[seed_idx]=1.0/len(seed_idx)
    r=s.copy()
    for _ in range(iters):
        r=(1-alpha)*s+alpha*(P@r)
    return r

def evalp(y,p):
    yh=(p>=0.5).astype(int)
    return (roc_auc_score(y,p),average_precision_score(y,p),
            f1_score(y,yh,zero_division=0),matthews_corrcoef(y,yh))

def main():
    print("Loading KG..."); commits,tokens,files,devs=load_kg()
    cids=sorted(commits, key=lambda c:commits[c]["ts"])
    y=np.array([commits[c]["buggy"] for c in cids])
    ts=np.array([commits[c]["ts"] for c in cids])
    Nc=len(cids); cut=int(Nc*(1-TEST_FRAC))
    train=np.arange(cut); test=np.arange(cut,Nc)
    print(f"  train={len(train)} test={len(test)}  "
          f"train bug={y[train].mean():.3f} test bug={y[test].mean():.3f}\n")
    hdr=f"{'method':<26}{'ROC_AUC':>8}{'PR_AUC':>8}{'F1':>7}{'MCC':>7}"
    print(hdr); print("-"*len(hdr))

    # baseline: metrics only
    Xm=np.array([[commits[c][m] for m in METRICS] for c in cids])
    base=make_pipeline(StandardScaler(),
         HistGradientBoostingClassifier(max_iter=300,class_weight="balanced",random_state=0))
    base.fit(Xm[train],y[train]); pm=base.predict_proba(Xm[test])[:,1]
    a=evalp(y[test],pm); print(f"{'baseline: JIT metrics':<26}{a[0]:8.3f}{a[1]:8.3f}{a[2]:7.3f}{a[3]:7.3f}")

    # M1: structural TF-IDF
    docs=[" ".join((tok+" ")*int(min(n,20)) for tok,n in tokens.get(c,())) for c in cids]
    vec=TfidfVectorizer(token_pattern=r"[^\s]+", min_df=3)
    vec.fit([docs[i] for i in train])      # leakage-free: vocab+IDF from TRAIN only
    Xt=vec.transform(docs)
    lr=LogisticRegression(max_iter=2000,class_weight="balanced")
    lr.fit(Xt[train],y[train]); pt=lr.predict_proba(Xt[test])[:,1]
    a=evalp(y[test],pt); print(f"{'M1 structural TF-IDF':<26}{a[0]:8.3f}{a[1]:8.3f}{a[2]:7.3f}{a[3]:7.3f}")

    # M2: PPR label propagation. Score = relative proximity to buggy vs benign
    # TRAIN history. Leakage-safe per-commit score via 4 chronological folds:
    # each fold is scored using seeds from strictly EARLIER folds only (so a
    # commit is never a seed for its own score) -> usable for train rows too.
    P,_=build_ppr_graph(cids,tokens,files,devs); N=P.shape[0]
    score=np.full(Nc, np.nan)
    folds=np.array_split(np.arange(Nc), 5)            # chronological folds
    seen=[]
    for k,fold in enumerate(folds):
        if seen:                                       # seeds from earlier folds
            past=np.concatenate(seen)
            bs=[i for i in past if y[i]==1]; gs=[i for i in past if y[i]==0]
            rb=ppr(P,bs,N); rg=ppr(P,gs,N)
            score[fold]=rb[fold]/(rb[fold]+rg[fold]+1e-12)
        else:
            score[fold]=y[np.concatenate(seen)].mean() if seen else 0.0
        seen.append(fold)
    score=np.nan_to_num(score, nan=float(y[train].mean()))
    a=evalp(y[test],score[test])
    print(f"{'M2 PPR propagation':<26}{a[0]:8.3f}{a[1]:8.3f}{a[2]:7.3f}{a[3]:7.3f}")

    # M3: early FUSION (leakage-free) -- one Logistic Regression over
    # [scaled JIT metrics  +  structural TF-IDF  +  PPR score].
    from scipy.sparse import hstack, csr_matrix
    sc=StandardScaler().fit(Xm[train])
    Xfuse=hstack([csr_matrix(sc.transform(Xm)), Xt, csr_matrix(score[:,None])]).tocsr()
    fl=LogisticRegression(max_iter=3000,class_weight="balanced")
    fl.fit(Xfuse[train],y[train]); pf=fl.predict_proba(Xfuse[test])[:,1]
    a=evalp(y[test],pf)
    print(f"{'M3 fusion (JIT+TFIDF+PPR)':<26}{a[0]:8.3f}{a[1]:8.3f}{a[2]:7.3f}{a[3]:7.3f}")

    print(f"\nGraph: {Nc} commits + {N-Nc} hub nodes (files/devs/AST-types)")
    print(f"random PR-AUC baseline ~= test bug rate {y[test].mean():.3f}")

if __name__=="__main__":
    main()
