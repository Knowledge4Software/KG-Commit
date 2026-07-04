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
import effort_metrics as em

OUT = Path(__file__).resolve().parent.parent / "outputs"

def lr(): return LogisticRegression(class_weight="balanced", max_iter=2000)

def evalp(y, p, eff=None):
    """ROC, PR-AUC, F1 + effort-aware Popt and ACC@20%LOC (if effort given)."""
    yh = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
    popt = acc20 = float("nan")
    if eff is not None:
        popt = em.popt(y, p, eff); acc20 = em.recall_at_effort(y, p, eff, 0.20)
    return roc_auc_score(y, p), average_precision_score(y, p), f1, popt, acc20

def _load_mentions():
    """Load the V3 text-graph layer edges (Commit)-[:MENTIONS]->(Term)."""
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(ai.NEO4J_URI, auth=ai.NEO4J_AUTH)
    with d.session(default_access_mode="READ") as s:
        rows = s.execute_read(lambda tx: [r.data() for r in tx.run(
            "MATCH (c:Commit {in_jit:true})-[m:MENTIONS]->(t:Term) "
            "RETURN c.id AS c, t.id AS t, coalesce(m.tfidf,1.0) AS w, t.kind AS k")])
    d.close()
    return rows


def text_layer_rows(commits, cids, tr, te, y, eff, Xf):
    """V3 rows using the graph-native text layer. Beyond the flat message TF-IDF,
    the graph layer supports (a) a Commit x Term incidence usable by the same
    relational machinery, and (b) a leakage-free Commit-Term-Commit META-PATH
    prior: a commit's risk = tfidf-weighted historical bug-rate of the Terms it
    mentions, using TRAIN commits only. That propagation is what a flat per-commit
    vector cannot represent."""
    ment = _load_mentions()
    if not ment:
        raise RuntimeError("no MENTIONS layer in graph (run build_text_layer.py --build)")
    cidx = {c: i for i, c in enumerate(cids)}
    tset = sorted({r["t"] for r in ment}); tidx = {t: j for j, t in enumerate(tset)}
    R = []; Col = []; V = []
    for r in ment:
        if r["c"] in cidx:
            R.append(cidx[r["c"]]); Col.append(tidx[r["t"]]); V.append(float(r["w"]))
    Xtl = sp.csr_matrix((V, (R, Col)), shape=(len(cids), len(tset)))

    # leakage-free Commit-Term-Commit meta-path prior (train-only term bug-rates)
    g = y[tr].mean()
    tb = np.zeros(len(tset)); tt = np.zeros(len(tset))
    Xtr = Xtl[tr]
    for i, ci in enumerate(tr):
        cols = Xtl[ci].indices
        tt[cols] += 1; tb[cols] += y[ci]
    term_rate = (tb + g*5) / (tt + 5)                 # smoothed term bug-rate
    dens = np.asarray(Xtl.sum(1)).ravel(); dens[dens == 0] = 1.0
    prior = np.asarray(Xtl.multiply(term_rate[None, :]).sum(1)).ravel() / dens

    out = []
    m = lr().fit(Xtl[tr], y[tr]); p = m.predict_proba(Xtl[te])[:, 1]
    out.append(("Ours: Text-GRAPH MENTIONS", *evalp(y[te], p, eff[te])))
    out.append(("Ours: Text-GRAPH prior (C-T-C)", *evalp(y[te], prior[te], eff[te])))
    Xfg = sp.hstack([Xf, Xtl, sp.csr_matrix(prior[:, None])]).tocsr()
    m = lr().fit(Xfg[tr], y[tr]); p = m.predict_proba(Xfg[te])[:, 1]
    out.append(("Ours: FUSION+text-GRAPH (V3)", *evalp(y[te], p, eff[te])))

    # MATCHED-CLASSIFIER comparison to the RF/XGB text baselines: run our best
    # feature set (V3 fusion+text-graph) through the SAME learner families as the
    # 768-d baseline, so the comparison isolates the FEATURES fairly.
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    Xd = Xfg.toarray()
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=0).fit(Xd[tr], y[tr])
    p = rf.predict_proba(Xd[te])[:, 1]
    out.append(("Ours: V3 fusion+textgraph [RF]", *evalp(y[te], p, eff[te])))
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         random_state=0).fit(Xd[tr], y[tr])
    p = hgb.predict_proba(Xd[te])[:, 1]
    out.append(("Ours: V3 fusion+textgraph [HGB]", *evalp(y[te], p, eff[te])))

    # FAIR TEST of contribution #3: "KG + text vs text alone". Fold the baseline's
    # own 768-d message representation (TF-IDF + Truncated-SVD, train-fit) into the
    # fusion and compare to that same text representation on its own -- both RF.
    msgs = [str(commits[c].get("message", "")) for c in cids]
    mv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, lowercase=True).fit([msgs[i] for i in tr])
    Xmsg = mv.transform(msgs)
    k = min(768, Xmsg.shape[1] - 1)
    svd768 = TruncatedSVD(n_components=k, random_state=0).fit(Xmsg[tr])
    T = svd768.transform(Xmsg)
    def rf(): return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=0)
    m = rf().fit(T[tr], y[tr]); p = m.predict_proba(T[te])[:, 1]
    out.append((f"Ours: text-{k} alone [RF]", *evalp(y[te], p, eff[te])))
    Xall = sp.hstack([Xfg, sp.csr_matrix(T)]).toarray()
    m = rf().fit(Xall[tr], y[tr]); p = m.predict_proba(Xall[te])[:, 1]
    out.append((f"Ours: V3 fusion+textgraph+text{k} [RF]", *evalp(y[te], p, eff[te])))
    return out


def main():
    commits, tokens, files, devs = ai.load_kg()
    cids = sorted(commits, key=lambda c: commits[c]["ts"])   # = author_date order
    y = np.array([commits[c]["buggy"] for c in cids]); n = len(cids)
    tr = np.arange(int(0.70*n)); te = np.arange(int(0.85*n), n)   # baselines' exact split
    eff = np.array([commits[c]["la"]+commits[c]["ld"] for c in cids], float)  # churn = LOC
    print(f"groovy commits={n}  train={len(tr)} (first 70%)  test={len(te)} (last 15%)  "
          f"test bug-rate={y[te].mean():.3f}\n")

    Xm = np.array([[commits[c][m] for m in ai.METRICS] for c in cids], float)
    docs = [" ".join((t+" ")*int(min(k,20)) for t,k in tokens.get(c,())) for c in cids]
    Xp,_ = oi.online_priors(cids, y, files, devs)
    C, P, _, N = oi.build_incidence(cids, tokens, files, devs)
    En = np.load(OUT/f"emb_node2vec_{n}.npy"); Et = np.load(OUT/f"emb_transe_{n}.npy")
    Ed = np.load(OUT/f"emb_distmult_{n}.npy")
    Esvd = TruncatedSVD(n_components=64, random_state=0).fit(C[tr]).transform(C)  # leak-free: fit on train

    rows = []
    def dense(name, X):
        sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
        m = lr().fit(Xs[tr], y[tr]); p = m.predict_proba(Xs[te])[:,1]
        rows.append((name, *evalp(y[te], p, eff[te])))

    dense("Ours: JIT-metrics (LR)", Xm)
    # structural TF-IDF (fit vocab on train docs only)
    vec = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=3).fit([docs[i] for i in tr])
    Xt = vec.transform(docs)
    m = lr().fit(Xt[tr], y[tr]); p = m.predict_proba(Xt[te])[:,1]
    rows.append(("Ours: Structural TF-IDF", *evalp(y[te], p, eff[te])))
    dense("Ours: node2vec", En); dense("Ours: TransE", Et); dense("Ours: DistMult", Ed)
    dense("Ours: KGE-SVD", Esvd)
    # wvRN priors (score, no fit)
    score = 0.5*Xp[:,2]+0.3*Xp[:,1]+0.2*Xp[:,0]
    rows.append(("Ours: wvRN priors", *evalp(y[te], score[te], eff[te])))
    # PPR (seeds from train)
    bs = tr[y[tr]==1]; gs = tr[y[tr]==0]
    rb = oi.ppr(P, list(bs), N); rg = oi.ppr(P, list(gs), N)
    pp = rb[:n]/(rb[:n]+rg[:n]+1e-12)
    rows.append(("Ours: PPR", *evalp(y[te], pp[te], eff[te])))
    # Fusion (metrics+priors+tfidf+ppr)
    sc = StandardScaler().fit(np.hstack([Xm,Xp])[tr]); dns = sc.transform(np.hstack([Xm,Xp]))
    Xf = sp.hstack([sp.csr_matrix(dns), Xt, sp.csr_matrix(pp[:,None])]).tocsr()
    m = lr().fit(Xf[tr], y[tr]); p = m.predict_proba(Xf[te])[:,1]
    rows.append(("Ours: FUSION (no text)", *evalp(y[te], p, eff[te])))
    # V2: commit-message TEXT (vocab fit on train only)
    msgs = [str(commits[c].get("message","")) for c in cids]
    tvec = TfidfVectorizer(ngram_range=(1,2), min_df=3, lowercase=True).fit([msgs[i] for i in tr])
    Xtx = tvec.transform(msgs)
    m = lr().fit(Xtx[tr], y[tr]); p = m.predict_proba(Xtx[te])[:,1]
    rows.append(("Ours: Commit-text (V2)", *evalp(y[te], p, eff[te])))
    # V2: Fusion + text
    Xft = sp.hstack([Xf, Xtx]).tocsr()
    m = lr().fit(Xft[tr], y[tr]); p = m.predict_proba(Xft[te])[:,1]
    rows.append(("Ours: FUSION+text (V2)", *evalp(y[te], p, eff[te])))

    # V3: TEXT-GRAPH-LAYER rows (Term/MENTIONS layer) -- added if the layer exists
    try:
        rows += text_layer_rows(commits, cids, tr, te, y, eff, Xf)
    except Exception as e:
        print(f"[text-layer rows skipped: {e}]")

    print(f"{'method':<28}{'ROC':>7}{'PR-AUC':>8}{'F1':>7}{'Popt':>7}{'ACC20':>7}")
    print("-"*64)
    for nm,roc,prc,f1,po,a20 in rows:
        print(f"{nm:<28}{roc:7.3f}{prc:8.3f}{f1:7.3f}{po:7.3f}{a20:7.3f}")
    print("\n-- baselines (same split, from baselines/results, ROC | F1) --")
    print("  JIT LR 0.698|0.232   RF 0.708|0.164   XGBoost 0.752|0.281")
    print("  JIT RF-tuned 0.729|0.336   XGBoost-tuned 0.755|0.267")
    print("  Text TF-IDF-768D: LR 0.805|0.182   RF 0.813|0.321   XGB 0.798|0.224")
    print("  Naive: Random 0.500|0.094   AllBuggy 0.500|0.133")

if __name__ == "__main__":
    main()
