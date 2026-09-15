"""
AST / base-structure characterisation for the active project  (NEEDS Neo4j).
============================================================================

Reproduces the first-phase (groovy) AST-stats analysis per project: distributions
that describe the *shape* of the knowledge graph beyond the plain node/edge tallies
already in subgraph_layer_stats.json. Specifically, over the resident graph:

  * node-type distribution          count per :ASTNode node_type (top 30)
  * change-group composition        ADDS/REMOVES/UPDATES/MOVES split, buggy vs benign
  * per-commit delta-size histogram  distribution of change magnitude per commit
  * AST depth / fan-out summary      structural complexity of the base ASTs

Unlike the effort/baseline/param scripts, this one QUERIES Neo4j (the graph must
be the resident project's), so run it only when that project's graph is loaded
(right after build, or after snapshot_neo4j.py restore). It is READ-ONLY.

Out: outputs/<project>/ast_structure_stats.json  + printed summary
Run: KGC_PROJECT=zookeeper python inference/collect_ast_structure_stats.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT, NEO4J_URI, NEO4J_AUTH  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def _rows(s, q, **p):
    return [r.data() for r in s.run(q, **p)]


def main():
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    out = {"project": PROJECT}
    with drv.session() as s:
        n = s.run("MATCH (a:ASTNode) RETURN count(a) AS n").single()["n"]
        if not n:
            raise SystemExit(f"No :ASTNode in the resident graph. Is '{PROJECT}' "
                             f"the loaded project? (build or snapshot_neo4j.py restore)")

        # 1. node-type distribution (top 30)
        out["node_type_dist"] = _rows(s, """
            MATCH (a:ASTNode) WHERE a.node_type IS NOT NULL
            RETURN a.node_type AS type, count(*) AS n
            ORDER BY n DESC LIMIT 30""")

        # 2. change-group composition, buggy vs benign
        out["change_group_by_label"] = _rows(s, """
            MATCH (c:Commit {in_jit:true})-[r:ADDS|REMOVES|UPDATES|MOVES]->()
            RETURN type(r) AS op, c.buggy AS buggy, count(*) AS n
            ORDER BY op, buggy""")

        # 3. per-commit delta-size distribution (adds+removes+updates+moves)
        out["delta_size_per_commit"] = _rows(s, """
            MATCH (c:Commit {in_jit:true})
            OPTIONAL MATCH (c)-[r:ADDS|REMOVES|UPDATES|MOVES]->()
            WITH c, count(r) AS delta
            RETURN delta ORDER BY delta""")
        deltas = [r["delta"] for r in out["delta_size_per_commit"]]
        if deltas:
            import numpy as np
            arr = np.array(deltas)
            out["delta_summary"] = dict(
                n=len(arr), mean=float(arr.mean()), median=float(np.median(arr)),
                p95=float(np.percentile(arr, 95)), max=int(arr.max()))
            out.pop("delta_size_per_commit")   # keep the file small; summary suffices

        # 4. AST depth / fan-out summary (base ASTs; child fan-out proxy)
        fan = s.run("""
            MATCH (a:ASTNode)-[:AST_CHILD]->(ch)
            WITH a, count(ch) AS k
            RETURN avg(k) AS mean_fanout, max(k) AS max_fanout,
                   percentileCont(k, 0.95) AS p95_fanout""").single()
        out["fanout"] = {k: (float(v) if v is not None else None) for k, v in fan.data().items()}

    drv.close()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "ast_structure_stats.json", "w"), indent=2)
    print(f"[{PROJECT}] AST structure stats:")
    print(f"  node types (top 3): {[(r['type'], r['n']) for r in out['node_type_dist'][:3]]}")
    if "delta_summary" in out:
        print(f"  delta/commit: median={out['delta_summary']['median']:.0f} "
              f"p95={out['delta_summary']['p95']:.0f}")
    print(f"  mean fan-out: {out['fanout'].get('mean_fanout')}")
    print(f"saved -> {OUT / 'ast_structure_stats.json'}")


if __name__ == "__main__":
    main()
