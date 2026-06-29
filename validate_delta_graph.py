"""
Step 4: Validate + visualize the delta graph in Neo4j.

Produces:
  - console summary: edge counts by type, commits with deltas, top files
  - outputs/delta_graph_summary.json
  - outputs/delta_graph/<commit>_<file>_delta.html  (pyvis) for a few examples

A delta visualization shows, for one (commit, file):
  - the Commit node (center)
  - its ADDS/REMOVES/UPDATES/MOVES edges (color-coded) to ASTNodes
  - each touched ASTNode plus its AST_CHILD neighbours for code context
"""

import json
from pathlib import Path
from neo4j import GraphDatabase
from pyvis.network import Network

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password1234")
OUT_DIR    = Path("outputs/delta_graph")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_COLOR = {
    "ADDS":    "#2ca02c",   # green
    "REMOVES": "#d62728",   # red
    "UPDATES": "#ff7f0e",   # orange
    "MOVES":   "#9467bd",   # purple
}


def summary(session):
    print("="*68)
    print("DELTA GRAPH SUMMARY")
    print("="*68)

    counts = {}
    for et in EDGE_COLOR:
        n = session.run(
            f"MATCH ()-[r:{et}]->() RETURN count(r) AS n"
        ).single()["n"]
        counts[et] = n
        print(f"  {et:10s}: {n:6d} edges")

    n_delta_nodes = session.run(
        "MATCH (a:ASTNode {is_delta:true}) RETURN count(a) AS n"
    ).single()["n"]
    print(f"  is_delta ASTNodes (inserted): {n_delta_nodes}")

    n_commits = session.run("""
        MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->()
        RETURN count(DISTINCT c) AS n
    """).single()["n"]
    print(f"  Commits carrying delta edges: {n_commits}")

    print("\n  Top commits by total delta edges:")
    rows = session.run("""
        MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->()
        RETURN c.id AS commit, count(r) AS n
        ORDER BY n DESC LIMIT 10
    """).data()
    for r in rows:
        print(f"    {r['commit'][:10]}  {r['n']:5d} edges")

    return {"edge_counts": counts,
            "delta_nodes": n_delta_nodes,
            "commits_with_deltas": n_commits,
            "top_commits": rows}


def pick_examples(session, k=5):
    """Pick (commit, file) pairs with a good mix of edge types for viz."""
    rows = session.run("""
        MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
        WITH c.id AS commit, a.file AS file,
             count(r) AS n,
             count(DISTINCT type(r)) AS kinds
        WHERE file IS NOT NULL
        RETURN commit, file, n, kinds
        ORDER BY kinds DESC, n DESC
        LIMIT $k
    """, k=k).data()
    return rows


def visualize(session, commit, file, max_context=120):
    net = Network(height="800px", width="100%", bgcolor="#ffffff",
                  font_color="#222", directed=True)
    net.barnes_hut(gravity=-8000, spring_length=120)

    cshort = commit[:8]
    fname  = file.split("/")[-1]

    # Center commit node
    cid = f"commit::{commit}"
    net.add_node(cid, label=f"Commit\n{cshort}", shape="star",
                 color="#1f77b4", size=40,
                 title=f"Commit {commit}\nFile: {file}")

    # All delta edges for this (commit, file)
    rows = session.run("""
        MATCH (c:Commit {id:$cid})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
        WHERE a.file = $file
        RETURN type(r) AS etype, a.id AS aid, a.ast_type AS atype,
               a.value AS val, a.pos_line AS line,
               r.old_value AS old_value, r.new_value AS new_value,
               r.new_parent_type AS new_parent_type
        LIMIT $lim
    """, cid=commit, file=file, lim=max_context).data()

    touched = set()
    for r in rows:
        aid   = r["aid"]
        etype = r["etype"]
        touched.add(aid)
        label = f"{r['atype']}"
        if r["val"]:
            label += f"\n{str(r['val'])[:18]}"
        tip = f"{r['atype']}  (line {r['line']})"
        if etype == "UPDATES":
            tip += f"\nold: {r['old_value']}\nnew: {r['new_value']}"
        elif etype == "MOVES":
            tip += f"\nnew parent: {r['new_parent_type']}"
        net.add_node(aid, label=label, shape="dot",
                     color=EDGE_COLOR[etype], size=18, title=tip)
        net.add_edge(cid, aid, color=EDGE_COLOR[etype], width=2,
                     label=etype, title=etype)

    # Add AST_CHILD context (one hop) around touched nodes, for code context
    if touched:
        ctx = session.run("""
            MATCH (a:ASTNode)-[:AST_CHILD]-(nb:ASTNode)
            WHERE a.id IN $ids
            RETURN DISTINCT a.id AS aid, nb.id AS nid,
                   nb.ast_type AS ntype, nb.value AS nval
            LIMIT $lim
        """, ids=list(touched), lim=max_context).data()
        for c in ctx:
            nid = c["nid"]
            if nid not in touched:
                lbl = c["ntype"] or "?"
                if c["nval"]:
                    lbl += f"\n{str(c['nval'])[:16]}"
                net.add_node(nid, label=lbl, shape="dot",
                             color="#cccccc", size=10, title=lbl)
            net.add_edge(c["aid"], nid, color="#dddddd", width=1)

    out = OUT_DIR / f"{cshort}_{fname.replace('.java','')}_delta.html"
    net.write_html(str(out), notebook=False)
    return out, len(rows)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as session:
        s = summary(session)

        print("\n  Generating example visualizations...")
        examples = pick_examples(session, k=5)
        viz_files = []
        for ex in examples:
            out, n = visualize(session, ex["commit"], ex["file"])
            print(f"    {out.name}  ({n} delta edges, {ex['kinds']} kinds)")
            viz_files.append(str(out))

        s["visualizations"] = viz_files
        Path("outputs/delta_graph_summary.json").write_text(
            json.dumps(s, indent=2, default=str))
        print("\n  Summary -> outputs/delta_graph_summary.json")
        print(f"  HTML    -> {OUT_DIR}")
    driver.close()


if __name__ == "__main__":
    main()
