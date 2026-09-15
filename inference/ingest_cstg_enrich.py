"""
Grow the CSTG in Neo4j with the semantic ENRICHMENT: change-intent typing.

Adds a small set of Intent nodes and one edge per labelled commit:
    (Commit)-[:HAS_INTENT]->(Intent {type})
plus the intent as a Commit property (c.intent) for easy querying. This is the
graph-side of the intent-realization modelling; the recency/bug-cache signal is a
runtime online feature (per-commit, past-only) and is not a static edge.

Idempotent (MERGE). Fast now that Commit.id is indexed.

Run:  python inference/ingest_cstg_enrich.py
"""
from neo4j import GraphDatabase
import advanced_infer as ai
import cstg_consistency as cc

import _kgc_paths  # noqa: F401
from config.project_config import NEO4J_URI, NEO4J_AUTH


def main():
    commits, *_ = ai.load_kg()
    cids = list(commits)
    intents = cc.intent_of(commits, cids)
    rows = [{"c": c, "it": it} for c, it in intents.items()]
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with d.session() as s:
        s.run("CREATE INDEX intent_id IF NOT EXISTS FOR (i:Intent) ON (i.id)")
        s.run("CALL db.awaitIndexes(120)")
        for i in range(0, len(rows), 5000):
            s.execute_write(lambda tx: tx.run("""
                UNWIND $rows AS r
                MERGE (it:Intent {id:r.it})
                WITH it, r MATCH (c:Commit {id:r.c})
                SET c.intent = r.it
                MERGE (c)-[:HAS_INTENT]->(it)
            """, rows=rows[i:i+5000]))
        dist = {r["it"]: r["n"] for r in s.run(
            "MATCH (:Commit)-[:HAS_INTENT]->(i:Intent) RETURN i.id AS it, count(*) AS n ORDER BY n DESC")}
        n = s.run("MATCH ()-[r:HAS_INTENT]->() RETURN count(r) AS n").single()["n"]
    d.close()
    print(f"HAS_INTENT edges: {n}")
    print("intent distribution:", dist)


if __name__ == "__main__":
    main()
