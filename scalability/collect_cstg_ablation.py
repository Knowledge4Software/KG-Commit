"""
CSTG internal-layer ablation: a component-by-component comparison of the CSTG
semantic-text layer's own building blocks, on the controlled split (train first
70%, test last 15%; DB-free, reuses the cached fitted bundle outputs/cstg_bundle.pkl
-- no re-parse, no Neo4j).

This extends inference/validate_cstg.py's honest ablation
(docs/KG-Commit evolution.tex, sec:cstg, "What drives it") with the one row that
section's Q4 discussion flags as missing: a MATCHED-LEARNER (RF) plain-TF-IDF row,
so the "does the full CSTG representation beat a bag-of-words baseline" comparison
is apples-to-apples under the SAME non-linear learner, not mixed linear-vs-RF.

Components compared (both learners, so within- and across-learner reads are both
honest):
  (a) plain TF-IDF (bag-of-words), matched vocabulary size
  (b) TW-IDF (graph-of-words: TextRank centrality x IDF)
  (c) NPMI-propagated term-risk prior alone
  (d) TW-IDF + prior
  (e) full CSTG (TW-IDF + prior + typed-mass)
  (f) JIT + full CSTG (the deployed-style combination on this split)
plus the published external baseline for context.

Output: outputs/scalability/cstg_ablation.json
        outputs/scalability/tab_cstg_ablation.tex
        docs/figures/v4/scalability/fig_cstg_ablation.{png,pdf}

Run:  python scalability/collect_cstg_ablation.py
"""
import pickle
import numpy as np
import scipy.sparse as sp
import pandas as pd

import _common as C

DIFFS = C.ROOT / "data" / "apachejit" / "apachejit_with_diffs_rebuilt.csv"
METRICS = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
PUBLISHED_BASELINE = dict(ROC=0.813, PR=float("nan"), F1=0.321, Popt=float("nan"), ACC20=float("nan"))


def _lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def _rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=0)


def _evalp(y, p, eff):
    import effort_metrics as em
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    yh = (p >= 0.5).astype(int)
    return dict(ROC=float(roc_auc_score(y, p)), PR=float(average_precision_score(y, p)),
               F1=float(f1_score(y, yh, zero_division=0)),
               Popt=float(em.popt(y, p, eff)),
               ACC20=float(em.recall_at_effort(y, p, eff, 0.20)))


def main():
    import cstg as Cstg
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(DIFFS)
    df = df[df["project"] == "apache/groovy"].sort_values("author_date").reset_index(drop=True)
    y = df["buggy"].astype(int).to_numpy(); n = len(df)
    tr = np.arange(int(0.70 * n)); te = np.arange(int(0.85 * n), n)
    eff = (df["la"] + df["ld"]).to_numpy(float)
    Xm = df[METRICS].fillna(0).to_numpy(float)
    texts = df["diff_text"].astype(str).tolist()
    print(f"groovy={n} train={len(tr)} test={len(te)} test bug-rate={y[te].mean():.3f}")

    cache = C.ROOT / "outputs" / "cstg_bundle.pkl"
    print("loading cached CSTG bundle (fitted train-only, no re-parse) ...")
    cs, B = pickle.load(open(cache, "rb"))
    Xtw, prior, typed = B["X_twidf"], B["prior"], B["typed"]
    print(f"  vocab={len(cs.vocab)} npmi-edges={sum(len(v) for v in cs.npmi.values())//2}")

    Xcstg = sp.hstack([Xtw, sp.csr_matrix(prior[:, None]), sp.csr_matrix(typed)]).tocsr()
    tfidf_plain = TfidfVectorizer(max_features=len(cs.vocab), ngram_range=(1, 1),
                                  stop_words="english").fit([texts[i] for i in tr])
    Xtfidf = tfidf_plain.transform(texts)
    Xms = StandardScaler().fit(Xm[tr]).transform(Xm)
    Xall = sp.hstack([sp.csr_matrix(Xms), Xcstg]).tocsr()

    def run(X, learner_fn, dense=False):
        Xx = X.toarray() if (dense and sp.issparse(X)) else X
        m = learner_fn().fit(Xx[tr], y[tr]); p = m.predict_proba(Xx[te])[:, 1]
        return _evalp(y[te], p, eff[te])

    def score(p):
        return _evalp(y[te], p[te], eff[te])

    components = {}
    print("running matched-learner comparison (RF, dense) ...")
    components["plain_tfidf_RF"] = dict(
        pretty="Plain TF-IDF (bag-of-words) [RF]",
        **run(Xtfidf, _rf, dense=True))
    components["tw_idf_RF"] = dict(
        pretty="TW-IDF (graph-of-words) [RF]",
        **run(Xtw, _rf, dense=True))
    components["cstg_full_RF"] = dict(
        pretty="Full CSTG (+prior+typed) [RF]",
        **run(Xcstg, _rf, dense=True))
    components["jit_cstg_full_RF"] = dict(
        pretty="JIT + full CSTG [RF]",
        **run(Xall, _rf, dense=True))

    print("running component-isolation comparison (balanced LR) ...")
    components["plain_tfidf_LR"] = dict(
        pretty="(a) Plain TF-IDF (bag-of-words) [LR]",
        **run(Xtfidf, _lr, dense=False))
    components["tw_idf_LR"] = dict(
        pretty="(b) TW-IDF (graph-of-words) [LR]",
        **run(Xtw, _lr, dense=False))
    components["prior_LR"] = dict(
        pretty="(c) NPMI-propagated prior alone",
        **score(prior))
    components["tw_idf_prior_LR"] = dict(
        pretty="(d) TW-IDF + prior [LR]",
        **run(sp.hstack([Xtw, sp.csr_matrix(prior[:, None])]).tocsr(), _lr, dense=False))
    components["cstg_full_LR"] = dict(
        pretty="(e) Full CSTG (+typed) [LR]",
        **run(Xcstg, _lr, dense=False))

    out = {
        "_meta": dict(
            scope="CSTG internal-layer ablation, controlled split (train 70% / test "
                  "last 15%), DB-free replay of the cached fitted bundle",
            n_train=len(tr), n_test=len(te), test_bug_rate=float(y[te].mean()),
            published_baseline_RF=PUBLISHED_BASELINE,
            note="RF rows are the matched-learner comparison (fixes the mixed "
                 "linear-vs-RF gap flagged in sec:v4-cstg-qa Q4); LR rows are the "
                 "original component-isolation ablation."),
        "components": components,
    }
    p = C.save_json(out, "cstg_ablation.json")
    print(f"\nsaved -> {p}")

    print(f"\n{'component':<38}{'ROC':>7}{'PR':>7}{'F1':>7}{'Popt':>7}{'ACC20':>7}")
    print("-" * 73)
    for k, v in components.items():
        print(f"{v['pretty']:<38}{v['ROC']:7.3f}{v['PR']:7.3f}{v['F1']:7.3f}{v['Popt']:7.3f}{v['ACC20']:7.3f}")
    print(f"\npublished diff-text TF-IDF-768D RF: ROC={PUBLISHED_BASELINE['ROC']} F1={PUBLISHED_BASELINE['F1']}")

    # sanity anchor vs the numbers already in the paper (sec:cstg)
    print("\nanchor vs paper (sec:cstg 'What drives it'): TW-IDF-alone ROC should be "
          f"~0.746 (got {components['tw_idf_LR']['ROC']:.3f}); plain TF-IDF ROC ~0.769 "
          f"(got {components['plain_tfidf_LR']['ROC']:.3f}); full CSTG-RF ROC ~0.821 "
          f"(got {components['cstg_full_RF']['ROC']:.3f})")


if __name__ == "__main__":
    main()
