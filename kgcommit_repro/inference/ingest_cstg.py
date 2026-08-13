"""
Ingest the Commit Semantic-Text Graph (CSTG) into Neo4j as a first-class layer.

The CSTG grows ONLINE at commit arrival (see online_cstg.py); this writes the
accumulated state as a queryable/visualisable graph, alongside the AST and delta
layers:

  (Term {kind})                                     -- typed vocabulary
  (Commit)-[:MENTIONS {tw}]->(Term)                 -- top TW-IDF terms per commit
  (Term)-[:COOCCURS {npmi}]->(Term)                 -- NPMI co-occurrence (pruned)
  (Term)-[:GROUNDS_IN]->(ASTNode)                   -- code term == alive identifier leaf
  (Term)-[:REFERS_TO]->(File)                       -- term appears in a changed path

Reuses the cached fit (outputs/cstg_bundle.pkl). Idempotent (MERGE). Aggressive
pruning keeps the graph meaningful and writable on the containerised DB.

  python inference/ingest_cstg.py                    # Term + MENTIONS + COOCCURS
  python inference/ingest_cstg.py --ground           # also GROUNDS_IN / REFERS_TO
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
from neo4j import GraphDatabase
import cstg as C  # noqa: F401  (resolves the pickled CSTG class)

import _kgc_paths  # noqa: F401
from config.project_config import NEO4J_URI, NEO4J_AUTH, DIFF_CSV as DIFFS, \
    PROJECT_KEY as _PKEY, OUT as ROOT
MENTIONS_TOPK = 20      # top TW-IDF terms per commit
COOCCURS_TOPK = 8       # top NPMI neighbours per term
COOCCURS_MIN = 0.35
GROUND = "--ground" in sys.argv


def main():
    cs, B = pickle.load(open(ROOT / "cstg_bundle.pkl", "rb"))  # outputs/<project>/
    df = pd.read_csv(DIFFS, usecols=["commit_id", "project", "author_date"])
    df = df[df["project"] == _PKEY].sort_values("author_date").reset_index(drop=True)
    cids = df["commit_id"].astype(str).tolist()
    Xtw = B["X_twidf"].tocsr()
    inv = {j: t for t, j in cs.vocab.items()}
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with d.session() as s:
        for stmt in ("CREATE INDEX term_id IF NOT EXISTS FOR (t:Term) ON (t.id)",
                     "CREATE INDEX term_text IF NOT EXISTS FOR (t:Term) ON (t.text)",
                     "CREATE INDEX commit_id IF NOT EXISTS FOR (c:Commit) ON (c.id)",
                     "CREATE INDEX astnode_value IF NOT EXISTS FOR (a:ASTNode) ON (a.value)"):
            s.run(stmt)
        s.run("CALL db.awaitIndexes(300)")

        # ---- Term nodes (typed, with propagated risk) ----
        terms = [{"id": f"{cs.ttype[t]}:{t}", "text": t, "kind": cs.ttype[t],
                  "risk": float(cs.term_risk[t])} for t in cs.vocab]
        for i in range(0, len(terms), 5000):
            s.execute_write(lambda tx: tx.run(
                "UNWIND $r AS t MERGE (n:Term {id:t.id}) "
                "SET n.text=t.text, n.kind=t.kind, n.risk=t.risk", r=terms[i:i+5000]))
        print(f"Term nodes: {len(terms)}")

        # ---- MENTIONS: top-k TW-IDF terms per commit ----
        rows = []
        for i, cid in enumerate(cids):
            r = Xtw.getrow(i)
            if r.nnz == 0: continue
            order = np.argsort(-r.data)[:MENTIONS_TOPK]
            for k in order:
                j = r.indices[k]
                rows.append({"c": cid, "t": f"{cs.ttype[inv[j]]}:{inv[j]}", "w": float(r.data[k])})
        for i in range(0, len(rows), 5000):
            s.execute_write(lambda tx: tx.run(
                "UNWIND $r AS m MATCH (c:Commit {id:m.c}), (t:Term {id:m.t}) "
                "MERGE (c)-[e:MENTIONS]->(t) SET e.tw=m.w", r=rows[i:i+5000]))
        print(f"MENTIONS edges: {len(rows)}")

        # ---- COOCCURS: top-k NPMI neighbours per term ----
        seen = set(); crows = []
        for t, nbrs in cs.npmi.items():
            for u, w in sorted(nbrs, key=lambda x: -x[1])[:COOCCURS_TOPK]:
                if w < COOCCURS_MIN: continue
                key = tuple(sorted((t, u)))
                if key in seen: continue
                seen.add(key)
                crows.append({"a": f"{cs.ttype[t]}:{t}", "b": f"{cs.ttype[u]}:{u}", "w": float(w)})
        for i in range(0, len(crows), 5000):
            s.execute_write(lambda tx: tx.run(
                "UNWIND $r AS e MATCH (a:Term {id:e.a}), (b:Term {id:e.b}) "
                "MERGE (a)-[r:COOCCURS]->(b) SET r.npmi=e.w", r=crows[i:i+5000]))
        print(f"COOCCURS edges: {len(crows)}")

        if GROUND:
            # GROUNDS_IN: distinctive code terms -> alive identifier leaves (capped)
            # Batched: one transaction per 200 terms. The single-transaction form
            # accumulated millions of MERGEs and exceeded
            # dbms.memory.transaction.total.max, killing the server.
            s.run("""
                MATCH (t:Term {kind:'code'})
                CALL (t) {
                  MATCH (a:ASTNode {is_leaf:true}) WHERE a.alive=true AND a.value=t.text
                  WITH a LIMIT 200 RETURN collect(a) AS as
                }
                CALL (t, as) {
                  UNWIND as AS a MERGE (t)-[:GROUNDS_IN]->(a)
                } IN TRANSACTIONS OF 200 ROWS
            """).consume()
            n = s.run("MATCH ()-[r:GROUNDS_IN]->() RETURN count(r) AS n").single()["n"]
            print(f"GROUNDS_IN edges: {n}")
            # Batched + de-duplicated. The original matched every (commit,file)
            # edge, so a file touched by N commits was scanned N times.
            s.run("""
                MATCH (t:Term) WHERE size(t.text) >= 5
                CALL (t) {
                  MATCH (f:File) WHERE toLower(f.id) CONTAINS t.text
                  MERGE (t)-[:REFERS_TO]->(f)
                } IN TRANSACTIONS OF 100 ROWS
            """).consume()
            m = s.run("MATCH ()-[r:REFERS_TO]->() RETURN count(r) AS n").single()["n"]
            print(f"REFERS_TO edges: {m}")
    d.close()
    print("done -> CSTG layer in Neo4j (growth is online; this is the accumulated state).")


if __name__ == "__main__":
    main()
