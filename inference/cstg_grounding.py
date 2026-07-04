"""
Grounded cross-modal prior (online) -- the non-overlapping CSTG signal.

A commit's message/diff terms are GROUNDED in code entities (Term-[:GROUNDS_IN]->
ASTNode, ingested by ingest_cstg.py). This lets a commit's *textual* intent point
at code the commit may not even touch. The grounded prior is:

  for each code-term a commit mentions -> the files that term grounds in ->
  their historical bug-rate (from commits that TOUCHED them, strictly past).

This is orthogonal to the token streams (T, X, G) and to the file-priors (R, which
use the files a commit *changes*, not the files its *words* name). Everything is
past-only (predict-then-grow).

Quick prequential check here; if it helps it is added to the online ablation.

Run:  python inference/cstg_grounding.py
"""
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import advanced_infer as ai, effort_metrics as em
import cstg as C

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DIFFS = ROOT / "data/apachejit/apachejit_with_diffs_rebuilt.csv"
NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")


def term_to_files():
    """{term_text: set(files)} from the ingested GROUNDS_IN edges (cached)."""
    cache = OUT / "cstg_term_files.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with d.session(default_access_mode="READ") as s:
        rows = s.execute_read(lambda tx: [r.data() for r in tx.run(
            "MATCH (t:Term)-[:GROUNDS_IN]->(a:ASTNode) "
            "RETURN t.text AS term, collect(DISTINCT a.file) AS files")])
    d.close()
    m = {r["term"]: set(r["files"]) for r in rows}
    pickle.dump(m, open(cache, "wb")); return m


def commit_codeterms():
    """{commit_id: set(code-terms)} from the cached polarity parse."""
    docs = pickle.load(open(OUT / "cstg_polarity_docs.pkl", "rb"))
    return {cid: {term for _, typ, term in toks if typ == "code"} for cid, (toks, cen) in docs.items()}


def main():
    # commits in chronological order + labels + effort + files they TOUCH
    commits, tokens, files, devs = ai.load_kg()
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y = np.array([commits[c]["buggy"] for c in cids]); n = len(cids)
    eff = np.array([commits[c]["la"] + commits[c]["ld"] for c in cids], float)
    W = int(n * 0.30)

    t2f = term_to_files(); c2t = commit_codeterms()
    print(f"grounding map: {len(t2f)} grounded terms; {sum(len(v) for v in t2f.values())} term-file links")

    # incremental file bug-rate (grown from files each commit TOUCHES), past-only
    filec = defaultdict(lambda: [0, 0]); gb = gt = 0.0
    grounded = np.zeros(n)
    for i, c in enumerate(cids):
        g = (gb / gt) if gt else 0.0
        # files named by this commit's code-terms (cross-modal)
        gfiles = set()
        for term in c2t.get(c, ()):
            gfiles |= t2f.get(term, set())
        if gfiles:
            rs = [(filec[f][0] + g * 5) / (filec[f][1] + 5) for f in gfiles]
            grounded[i] = float(np.mean(rs))
        else:
            grounded[i] = g
        gt += 1; gb += int(y[i])                       # grow AFTER scoring
        for f in files.get(c, ()): filec[f][1] += 1; filec[f][0] += int(y[i])

    # quick prequential check (grounded alone), plus correlation with labels on stream
    ev = np.arange(W, n); ye = y[ev]; ge = grounded[ev]
    roc = roc_auc_score(ye, ge); pr = average_precision_score(ye, ge)
    popt = em.popt(ye, ge, eff[ev]); acc = em.recall_at_effort(ye, ge, eff[ev], 0.20)
    print(f"\ngrounded prior (online, streamed [{W}:{n}], bug-rate {ye.mean():.3f}):")
    print(f"  ROC {roc:.3f}  PR {pr:.3f}  Popt {popt:.3f}  ACC20 {acc:.3f}")
    np.save(OUT / "cstg_grounded_prior.npy", grounded)
    print(f"saved -> outputs/cstg_grounded_prior.npy  (aligned to KG ts-order cids)")


if __name__ == "__main__":
    main()
