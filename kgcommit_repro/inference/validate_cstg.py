"""
Validate the CSTG layer's PERFORMANCE (DB-free) on the controlled split.

Uses data/apachejit/apachejit_with_diffs_rebuilt.csv (labels + JIT metrics +
diff_text) so we can test the make-or-break question -- does the Commit
Semantic-Text Graph beat the strong diff-text TF-IDF baseline, and does each CSTG
component add? -- before investing in Neo4j ingestion and figures.

Matched Random-Forest learner; balanced LR shown too; effort-aware metrics
(Popt, ACC@20%LOC). Chronological split: train first 70%, test last 15%.

Run:  python inference/validate_cstg.py
"""
import numpy as np, scipy.sparse as sp, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import effort_metrics as em
import cstg as C

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import DIFF_CSV as DIFFS, PROJECT_KEY as _PKEY, OUT as _OUT
METRICS = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]


def lr():  return LogisticRegression(class_weight="balanced", max_iter=2000)
def rf():  return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=0)

def evalp(y, p, eff):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    yh = (p >= 0.5).astype(int)
    return (roc_auc_score(y, p), average_precision_score(y, p),
            f1_score(y, yh, zero_division=0),
            em.popt(y, p, eff), em.recall_at_effort(y, p, eff, 0.20))


def main():
    df = pd.read_csv(DIFFS)
    df = df[df["project"] == _PKEY].sort_values("author_date").reset_index(drop=True)
    y = df["buggy"].astype(int).to_numpy(); n = len(df)
    tr = np.arange(int(0.70 * n)); te = np.arange(int(0.85 * n), n)
    eff = (df["la"] + df["ld"]).to_numpy(float)
    Xm = df[METRICS].fillna(0).to_numpy(float)
    texts = df["diff_text"].astype(str).tolist()
    print(f"{_PKEY}: n={n}  train={len(tr)}  test={len(te)}  test bug-rate={y[te].mean():.3f}")

    import pickle
    cache = _OUT / "cstg_bundle.pkl"      # outputs/<project>/cstg_bundle.pkl
    if cache.exists():
        print("loading cached CSTG bundle...")
        cs, B = pickle.load(open(cache, "rb"))
    else:
        print("fitting CSTG (graph-of-words + NPMI + propagation, train-only)...")
        cs = C.CSTG().fit(texts, y, tr)
        B = cs.transform()
        pickle.dump((cs, B), open(cache, "wb"))
    Xtw, prior, typed = B["X_twidf"], B["prior"], B["typed"]
    print(f"  vocab={len(cs.vocab)}  npmi-edges={sum(len(v) for v in cs.npmi.values())//2}")

    # baseline diff-text TF-IDF-768 (exact config)
    dv = TfidfVectorizer(max_features=768, sublinear_tf=True, ngram_range=(1, 2),
                         stop_words="english").fit([texts[i] for i in tr])
    Xdt = dv.transform(texts)

    rows = []
    def run(name, X, learner, dense=False):
        Xx = X.toarray() if (dense and sp.issparse(X)) else X
        m = learner().fit(Xx[tr], y[tr]); p = m.predict_proba(Xx[te])[:, 1]
        rows.append((name, *evalp(y[te], p, eff[te])))
    def score(name, p):
        rows.append((name, *evalp(y[te], p[te], eff[te])))

    # --- headline comparison (matched RF) ---
    run("JIT metrics [RF]", Xm, rf)
    run("Baseline diff-text TFIDF768 [RF]", Xdt, rf, dense=True)
    run("CSTG TW-IDF [RF]", Xtw, rf, dense=True)
    score("CSTG term-risk prior (NPMI-prop)", prior)
    Xcstg = sp.hstack([Xtw, sp.csr_matrix(prior[:, None]), sp.csr_matrix(typed)]).tocsr()
    run("CSTG full [RF]", Xcstg, rf, dense=True)
    Xall = sp.hstack([sp.csr_matrix(StandardScaler().fit(Xm[tr]).transform(Xm)), Xcstg]).tocsr()
    run("JIT + CSTG full [RF]", Xall, rf, dense=True)

    print(f"\n{'method':<36}{'ROC':>7}{'PR':>7}{'F1':>7}{'Popt':>7}{'ACC20':>7}")
    print("-" * 71)
    for nm, roc, pr, f1, po, a in rows:
        print(f"{nm:<36}{roc:7.3f}{pr:7.3f}{f1:7.3f}{po:7.3f}{a:7.3f}")
    print("\npublished baseline: diff-text TF-IDF-768D RF = 0.813 ROC | 0.321 F1")

    # --- component ablation (balanced LR, isolates each piece) ---
    print("\n--- CSTG component ablation (balanced LR) ---")
    abl = []
    def ablrun(name, X, dense=False):
        Xx = X.toarray() if (dense and sp.issparse(X)) else X
        m = lr().fit(Xx[tr], y[tr]); p = m.predict_proba(Xx[te])[:, 1]
        abl.append((name, *evalp(y[te], p, eff[te])))
    tfidf_plain = TfidfVectorizer(max_features=len(cs.vocab), ngram_range=(1, 1),
                                  stop_words="english").fit([texts[i] for i in tr])
    ablrun("(a) plain TF-IDF (bag-of-words)", tfidf_plain.transform(texts))
    ablrun("(b) TW-IDF (graph-of-words)", Xtw)
    abl.append(("(c) NPMI-propagated prior", *evalp(y[te], prior[te], eff[te])))
    ablrun("(d) TW-IDF + prior", sp.hstack([Xtw, sp.csr_matrix(prior[:, None])]).tocsr())
    ablrun("(e) full CSTG (+typed)", Xcstg)
    print(f"{'component':<34}{'ROC':>7}{'PR':>7}{'F1':>7}{'Popt':>7}{'ACC20':>7}")
    print("-" * 69)
    for nm, roc, pr, f1, po, a in abl:
        print(f"{nm:<34}{roc:7.3f}{pr:7.3f}{f1:7.3f}{po:7.3f}{a:7.3f}")


if __name__ == "__main__":
    main()
