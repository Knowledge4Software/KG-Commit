"""
Collect per-layer structural statistics for the v4 subgraph-ablation write-up.

Gathers, for AST + the four alternative layers (CFG/DFG/PDG/SEQ), the graph-size
and delta tallies used in the comprehensive tables/plots and the LaTeX v4 section:
node counts (total/alive/delta-inserted), delta-edge counts by type
(ADDS/REMOVES/UPDATES/MOVES), token-vocabulary size, commits carrying tokens,
files covered, and the top change-token types.

Cached to outputs/subgraph_layer_stats.json so the notebook and paper can render
without a live DB.

Run:  python inference/collect_subgraph_stats.py
"""
import json
from pathlib import Path
from neo4j import GraphDatabase

OUT = Path(__file__).resolve().parent.parent / "outputs"
NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")

LAYERS = [   # (variant id, pretty, label, type property)
    ("ast", "AST",        "ASTNode", "ast_type"),
    ("cfg", "CFG",        "CFGNode", "atype"),
    ("dfg", "DFG",        "DFGNode", "atype"),
    ("pdg", "PDG/CPG",    "PDGNode", "atype"),
    ("seq", "Token-seq",  "SEQNode", "atype"),
]
DELTA = ["ADDS", "REMOVES", "UPDATES", "MOVES"]


def main():
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    stats = {}
    with d.session() as s:
        one = lambda cy, **kw: s.run(cy, **kw).single()[0]
        for vid, pretty, L, tp in LAYERS:
            row = {"pretty": pretty, "label": L}
            row["nodes"]  = one(f"MATCH (a:{L}) RETURN count(a)")
            row["alive"]  = one(f"MATCH (a:{L}) WHERE coalesce(a.alive,true) RETURN count(a)")
            row["delta_inserted"] = one(f"MATCH (a:{L}) WHERE a.is_delta RETURN count(a)")
            row["files"]  = one(f"MATCH (a:{L}) RETURN count(DISTINCT a.file)")
            for rel in DELTA:
                row[rel] = one(f"MATCH (:Commit)-[r:{rel}]->(:{L}) RETURN count(r)")
            row["delta_total"] = sum(row[r] for r in DELTA)
            row["commits_with_tokens"] = one(
                f"MATCH (c:Commit {{in_jit:true}})-[:ADDS|REMOVES|UPDATES|MOVES]->(:{L}) "
                f"RETURN count(DISTINCT c)")
            # token vocabulary = distinct {edge}:{type}
            vocab = s.run(f"""MATCH (:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:{L})
                RETURN DISTINCT type(r)+':'+coalesce(a.{tp},'?') AS tok""").data()
            row["n_token_types"] = len(vocab)
            row["n_node_types"] = one(
                f"MATCH (a:{L}) RETURN count(DISTINCT coalesce(a.{tp},'?'))")
            # top tokens by frequency
            top = s.run(f"""MATCH (:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:{L})
                RETURN type(r)+':'+coalesce(a.{tp},'?') AS tok, count(*) AS n
                ORDER BY n DESC LIMIT 12""").data()
            row["top_tokens"] = [(t["tok"], t["n"]) for t in top]
            stats[vid] = row
            print(f"{pretty:<10} nodes={row['nodes']:>9,} delta={row['delta_total']:>9,} "
                  f"tok_types={row['n_token_types']:>4} commits_tok={row['commits_with_tokens']:>6,}")
    d.close()
    # core (commit/file/dev) context
    stats["_meta"] = {"note": "v4 subgraph ablation layer stats"}
    OUT.mkdir(exist_ok=True)
    json.dump(stats, open(OUT / "subgraph_layer_stats.json", "w"), indent=1)
    print(f"\nsaved -> {OUT/'subgraph_layer_stats.json'}")


if __name__ == "__main__":
    main()
