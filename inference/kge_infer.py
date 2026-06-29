"""
Famous KG-embedding inference methods for buggy/benign commit classification
(CPU-only, GNN-free). Two families:

  node2vec / DeepWalk  (gensim)  - biased random walks over the heterogeneous
        KG (commits + file/developer/AST-type hubs) -> skip-gram embeddings ->
        a linear classifier on the commit vectors. This is the classic shallow
        graph-embedding "GNN substitute" (RDF2Vec is the same idea on KG walks).

  TransE / DistMult    (pykeen)  - translational / bilinear knowledge-graph
        embeddings trained on the KG triples; commit entity vectors -> linear
        classifier.

Both are evaluated with a time-ordered split (and node2vec also prequentially
online, refit on the expanding past window). KG embeddings are transductive: the
embedding uses graph STRUCTURE only (no labels), and the buggy label is split by
time -- so there is no label leakage.

Run:  python inference/kge_infer.py
"""
import numpy as np, random
import scipy.sparse as sp
from collections import defaultdict
from neo4j import GraphDatabase
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef)

NEO4J_URI="bolt://localhost:7687"; NEO4J_AUTH=("neo4j","password1234")
DIM=64; TEST_FRAC=0.30
random.seed(0); np.random.seed(0)

def q(driver,cypher):
    with driver.session(default_access_mode="READ") as s:
        return s.execute_read(lambda tx:[r.data() for r in tx.run(cypher)])

def load():
    d=GraphDatabase.driver(NEO4J_URI,auth=NEO4J_AUTH)
    commits={}
    for r in q(d,"""MATCH (c:Commit {in_jit:true})
        RETURN c.id AS id, c.author_ts AS ts, c.buggy AS b"""):
        commits[r["id"]]={"ts":r["ts"] or 0,"buggy":int(bool(r["b"]))}
    edges=defaultdict(list)   # commit -> list of (relation, hub-entity)
    for r in q(d,"""MATCH (c:Commit {in_jit:true})-[:MODIFIED|ADDED|DELETED]->(f:File)
        RETURN c.id AS c, f.id AS h"""): edges[r["c"]].append(("TOUCHES","F:"+r["h"]))
    for r in q(d,"""MATCH (c:Commit {in_jit:true})-[:AUTHORED_BY]->(dv:Developer)
        RETURN c.id AS c, dv.id AS h"""): edges[r["c"]].append(("BY","D:"+r["h"]))
    for r in q(d,"""MATCH (c:Commit {in_jit:true})-[:FIXES_ISSUE]->(i:Issue)
        RETURN c.id AS c, i.id AS h"""): edges[r["c"]].append(("FIXES","I:"+r["h"]))
    for r in q(d,"""MATCH (c:Commit {in_jit:true})-[rel:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
        RETURN c.id AS c, type(rel)+'_'+coalesce(a.ast_type,'?') AS rel, count(*) AS n"""):
        edges[r["c"]].append((r["rel"].split("_")[0]+"_TYPE","T:"+r["rel"]))
    d.close()
    return commits, edges

def metr(y,p):
    yh=(p>=0.5).astype(int)
    return (roc_auc_score(y,p),average_precision_score(y,p),
            f1_score(y,yh,zero_division=0),matthews_corrcoef(y,yh))

# ── node2vec / DeepWalk ──────────────────────────────────────────────────────
def build_adj(cids, edges):
    """Undirected adjacency list over commits + hubs (weighted by IDF)."""
    import math; Nc=len(cids)
    df=defaultdict(int)
    for c in cids:
        for rel,h in edges.get(c,()): df[h]+=1
    adj=defaultdict(list)
    for c in cids:
        for rel,h in edges.get(c,()):
            w=math.log(1.0+Nc/df[h])
            adj[c].append((h,w)); adj[h].append((c,w))
    return adj

def walks(adj, nodes, num=10, length=20):
    W=[]
    for _ in range(num):
        random.shuffle(nodes)
        for n in nodes:
            walk=[n]; cur=n
            for _ in range(length-1):
                nb=adj.get(cur)
                if not nb: break
                ns,ws=zip(*nb); ws=np.array(ws); ws=ws/ws.sum()
                cur=ns[np.random.choice(len(ns),p=ws)]; walk.append(cur)
            W.append(walk)
    return W

def node2vec_embed(cids, edges, dim=DIM):
    from gensim.models import Word2Vec
    adj=build_adj(cids,edges)
    nodes=list(adj.keys())
    W=walks(adj,nodes)
    m=Word2Vec(W,vector_size=dim,window=5,min_count=0,sg=1,workers=4,epochs=5,seed=0)
    E=np.zeros((len(cids),dim))
    for i,c in enumerate(cids):
        if c in m.wv: E[i]=m.wv[c]
    return E

# ── pykeen TransE / DistMult ─────────────────────────────────────────────────
def kge_embed(cids, edges, model="TransE", dim=DIM, epochs=40, batch_size=4096):
    # Train embeddings directly (no expensive link-prediction evaluation) -- we
    # only need the entity vectors, so we skip pykeen's pipeline() evaluation.
    import torch
    from pykeen.triples import TriplesFactory
    from pykeen.models import TransE, DistMult
    from pykeen.training import SLCWATrainingLoop
    triples=[]
    for c in cids:
        for rel,h in edges.get(c,()):
            triples.append((c,rel,h))
    tf=TriplesFactory.from_labeled_triples(np.array(triples,dtype=str))
    cls={"TransE":TransE,"DistMult":DistMult}[model]
    m=cls(triples_factory=tf, embedding_dim=dim, random_seed=0).to("cpu")
    opt=torch.optim.Adam(m.parameters(), lr=0.01)
    loop=SLCWATrainingLoop(model=m, triples_factory=tf, optimizer=opt)
    loop.train(triples_factory=tf, num_epochs=epochs, batch_size=batch_size, use_tqdm=False)
    ent=m.entity_representations[0](indices=None).detach().cpu().numpy()
    if ent.ndim>2: ent=ent.reshape(ent.shape[0],-1)
    id2idx=tf.entity_to_id
    E=np.zeros((len(cids),ent.shape[1]))
    for i,c in enumerate(cids):
        if c in id2idx: E[i]=ent[id2idx[c]]
    return E

def offline(name,E,y,cut):
    sc=StandardScaler().fit(E[:cut])
    Es=sc.transform(E)
    clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Es[:cut],y[:cut])
    p=clf.predict_proba(Es[cut:])[:,1]; a=metr(y[cut:],p)
    print(f"{name:<26}{a[0]:8.3f}{a[1]:8.3f}{a[2]:7.3f}{a[3]:7.3f}")

def online(name,E,y,w,block=200,refit=3):
    Nc=len(y); preds=np.full(Nc,np.nan); clf=None; blk=0; i=w
    while i<Nc:
        j=min(Nc,i+block); idx=np.arange(i,j); past=np.arange(i)
        if blk%refit==0:
            sc=StandardScaler().fit(E[past]); Es=sc.transform(E)
            clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Es[past],y[past])
        preds[idx]=clf.predict_proba(Es[idx])[:,1]; i=j; blk+=1
    ev=np.arange(w,Nc); a=metr(y[ev],np.nan_to_num(preds[ev],nan=y[:w].mean()))
    print(f"{name:<26}{a[0]:8.3f}{a[1]:8.3f}{a[2]:7.3f}{a[3]:7.3f}")

def main():
    print("Loading KG..."); commits,edges=load()
    cids=sorted(commits,key=lambda c:commits[c]["ts"])
    y=np.array([commits[c]["buggy"] for c in cids]); Nc=len(cids)
    cut=int(Nc*(1-TEST_FRAC)); w=int(Nc*0.40)
    print(f"  {Nc} commits; offline split at {cut}; online warmup {w}\n")

    print("Embedding: node2vec (gensim random walks)...")
    En=node2vec_embed(cids,edges)
    print("Embedding: TransE (pykeen)...")
    Et=kge_embed(cids,edges,"TransE")
    print("Embedding: DistMult (pykeen)...")
    Ed=kge_embed(cids,edges,"DistMult")

    hdr=f"{'method':<26}{'ROC_AUC':>8}{'PR_AUC':>8}{'F1':>7}{'MCC':>7}"
    print("\n== OFFLINE (time-ordered 70/30) ==\n"+hdr); print("-"*len(hdr))
    offline("node2vec + LogReg",En,y,cut)
    offline("TransE + LogReg",Et,y,cut)
    offline("DistMult + LogReg",Ed,y,cut)
    print("\n== ONLINE (prequential, expanding window) ==\n"+hdr); print("-"*len(hdr))
    online("ON node2vec",En,y,w)
    online("ON TransE",Et,y,w)
    online("ON DistMult",Ed,y,w)
    print(f"\nrandom PR-AUC ~= test bug rate {y[cut:].mean():.3f} (offline), "
          f"{y[w:].mean():.3f} (online)")

if __name__=="__main__":
    main()
