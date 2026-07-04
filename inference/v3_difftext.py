"""
V3 decisive contribution-#3 test on the controlled split.

The strong "Text TF-IDF-768D" baseline (ROC 0.813) is, per the baselines notebook,
TfidfVectorizer(max_features=768, sublinear_tf=True, ngram_range=(1,2),
stop_words='english') over the commit DIFF TEXT + RandomForest -- i.e. bag-of-words
on the actual code changes, NOT a transformer and NOT our AST-token abstraction.

This script:
  1. reproduces that baseline exactly (validates ~0.813),
  2. builds our KG fusion (metrics + structural-TFIDF + priors + PPR + text-graph
     MENTIONS layer),
  3. tests the FAIR question: does KG + diff-text beat diff-text ALONE?
on the baselines' chronological split (train first 70%, test last 15%), with
effort-aware metrics. Matched RF learner throughout.

Run:  python inference/v3_difftext.py
"""
import numpy as np, scipy.sparse as sp, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import advanced_infer as ai, online_infer as oi, effort_metrics as em
from compare_baselines import evalp

DIFFS = Path(__file__).resolve().parent.parent / "data/apachejit/apachejit_with_diffs_rebuilt.csv"


def rf():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=0)


def load_mentions():
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(ai.NEO4J_URI, auth=ai.NEO4J_AUTH)
    with d.session(default_access_mode="READ") as s:
        rows = s.execute_read(lambda tx: [r.data() for r in tx.run(
            "MATCH (c:Commit {in_jit:true})-[m:MENTIONS]->(t:Term) "
            "RETURN c.id AS c, t.id AS t, coalesce(m.tfidf,1.0) AS w")])
    d.close()
    return rows


def main():
    commits, tokens, files, devs = ai.load_kg()
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y = np.array([commits[c]["buggy"] for c in cids]); n = len(cids)
    tr = np.arange(int(0.70 * n)); te = np.arange(int(0.85 * n), n)
    eff = np.array([commits[c]["la"] + commits[c]["ld"] for c in cids], float)
    print(f"groovy commits={n}  train={len(tr)}  test={len(te)}  test bug-rate={y[te].mean():.3f}")

    # ---- diff text, aligned to cids by commit_id ----
    df = pd.read_csv(DIFFS, usecols=["commit_id", "project", "diff_text"])
    df = df[df["project"] == "apache/groovy"]
    dmap = dict(zip(df["commit_id"].astype(str), df["diff_text"].astype(str)))
    diffs = [dmap.get(c, "") for c in cids]
    cov = sum(1 for c in cids if c in dmap)
    print(f"diff-text coverage: {cov}/{n} commits matched by id\n")

    # ---- KG fusion features (metrics + structural TFIDF + priors + PPR + text-graph) ----
    Xm = np.array([[commits[c][m] for m in ai.METRICS] for c in cids], float)
    docs = [" ".join((t + " ") * int(min(k, 20)) for t, k in tokens.get(c, ())) for c in cids]
    vt = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=3).fit([docs[i] for i in tr])
    Xt = vt.transform(docs)
    Xp, _ = oi.online_priors(cids, y, files, devs)
    C, P, _, N = oi.build_incidence(cids, tokens, files, devs)
    bs = tr[y[tr] == 1]; gs = tr[y[tr] == 0]
    rb = oi.ppr(P, list(bs), N); rg = oi.ppr(P, list(gs), N)
    pp = rb[:n] / (rb[:n] + rg[:n] + 1e-12)
    # text-graph MENTIONS matrix
    ment = load_mentions(); cidx = {c: i for i, c in enumerate(cids)}
    tset = sorted({r["t"] for r in ment}); tix = {t: j for j, t in enumerate(tset)}
    R = []; Co = []; V = []
    for r in ment:
        if r["c"] in cidx:
            R.append(cidx[r["c"]]); Co.append(tix[r["t"]]); V.append(float(r["w"]))
    Xtl = sp.csr_matrix((V, (R, Co)), shape=(n, len(tset)))
    sc = StandardScaler().fit(np.hstack([Xm, Xp])[tr])
    Xdns = sc.transform(np.hstack([Xm, Xp]))
    Xkg = sp.hstack([sp.csr_matrix(Xdns), Xt, sp.csr_matrix(pp[:, None]), Xtl]).tocsr()

    # ---- diff-text TF-IDF 768 (EXACT baseline config) ----
    dv = TfidfVectorizer(max_features=768, sublinear_tf=True, ngram_range=(1, 2),
                         stop_words="english").fit([diffs[i] for i in tr])
    Xdt = dv.transform(diffs)

    rows = []
    def run(name, X):
        Xd = X.toarray() if sp.issparse(X) else X
        m = rf().fit(Xd[tr], y[tr]); p = m.predict_proba(Xd[te])[:, 1]
        rows.append((name, *evalp(y[te], p, eff[te])))

    run("Baseline: diff-text TFIDF768 [RF]", Xdt)              # expect ~0.813
    run("KG fusion (no diff-text) [RF]", Xkg)
    run("KG fusion + diff-text [RF]", sp.hstack([Xkg, Xdt]).tocsr())

    print(f"{'method':<38}{'ROC':>7}{'PR-AUC':>8}{'F1':>7}{'Popt':>7}{'ACC20':>7}")
    print("-" * 74)
    for nm, roc, pr, f1, po, a in rows:
        print(f"{nm:<38}{roc:7.3f}{pr:8.3f}{f1:7.3f}{po:7.3f}{a:7.3f}")
    print("\npublished baseline: Text TF-IDF-768D RF = 0.813 ROC | 0.321 F1")


if __name__ == "__main__":
    main()
