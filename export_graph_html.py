#!/usr/bin/env python3
"""
Export every graph-producing query from neo4j_queries.txt as a self-contained
interactive HTML file using pyvis. Open any output in a browser — no Neo4j needed.

Outputs (under outputs/html/):
  s1_3_whole_graph_sample.html
  s4_4_ast_bitwise_full.html
  s4_5_ast_groovyc_depth3.html
  s5_2_commit_file_ast_chain.html
  s7_1_schema_map.html
  s7_2_branch_commits.html
  s7_3_developer_universe.html
  s7_4_issue_network.html
  s7_5_file_ast_roots.html
  s7_7_file_lifecycle_invoker.html
  s7_8_developer_impact.html
  s7_9_commit_file_ast_history.html
  s7_10_issue_fix_chain.html
  s7_11_churned_file_ast.html
  s7_12_commit_neighbourhood.html
  s7_13_project_two_hop.html
  s8_1_base_master.html
  s8_2_base_commit_files.html
  s8_4_ast_roots_depth1.html
  s8_5_ast_small_files.html
  s8_6_ast_methodnode_full.html
  s8_7_ast_parser_depth3.html
  s8_10_ast_comparison.html

Run:
  python export_graph_html.py
"""

from pathlib import Path
from neo4j import GraphDatabase
from pyvis.network import Network

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password1234"
BASE_COMMIT    = "408b29851d7bbe4d343340832297e4be7e0c5578"
OUT_DIR        = Path(__file__).resolve().parent / "outputs" / "html"

# ── Colour / size palette by label ───────────────────────────────────────────
COLORS = {
    "Project":   "#e74c3c",
    "Branch":    "#9b59b6",
    "Commit":    "#3498db",
    "Developer": "#e67e22",
    "File":      "#27ae60",
    "ASTNode":   "#1abc9c",
    "Issue":     "#f1c40f",
}
NODE_SIZES = {
    "Project": 50, "Branch": 38, "Developer": 32,
    "File": 24, "Commit": 18, "Issue": 15, "ASTNode": 11,
}

# ── Driver-version-safe ID helpers ───────────────────────────────────────────
def _nid(n):
    try:    return n.element_id
    except AttributeError: return str(n.id)

def _rid(r):
    try:    return r.element_id
    except AttributeError: return str(r.id)

def _rel_start(r):
    try:    return r.start_node_element_id
    except AttributeError: return _nid(r.start_node)

def _rel_end(r):
    try:    return r.end_node_element_id
    except AttributeError: return _nid(r.end_node)

# ── Node display ──────────────────────────────────────────────────────────────
def _label(node):
    props  = dict(node)
    labels = list(node.labels)
    nid    = str(props.get("id", ""))
    if not labels:
        return nid[:28]
    lbl = labels[0]
    if lbl == "Commit":    return nid[:8]
    if lbl == "File":      return Path(nid).name or "File"
    if lbl == "ASTNode":   return (props.get("ast_type") or "AST")[:22]
    if lbl == "Developer": return nid.split("@")[0][:22]
    if lbl == "Issue":     return nid[:20]
    return nid[:25]

def _title(node):
    props  = dict(node)
    labels = list(node.labels)
    lines  = ["[" + ", ".join(labels) + "]"]
    for k, v in props.items():
        if v is not None:
            lines.append(f"{k}: {str(v)[:100]}")
    return "\n".join(lines)

# ── pyvis factory ─────────────────────────────────────────────────────────────
PHYSICS = """{
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -90,
      "centralGravity": 0.01,
      "springLength": 130,
      "springConstant": 0.08,
      "damping": 0.4,
      "avoidOverlap": 0.5
    },
    "stabilization": {"iterations": 250, "updateInterval": 50}
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": {"enabled": true}
  },
  "edges": {
    "smooth": {"type": "continuous"},
    "font": {"size": 9, "color": "#cccccc"}
  },
  "nodes": {"font": {"size": 11, "color": "#ffffff"}}
}"""

def make_net(height="900px"):
    net = Network(height=height, width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="white", notebook=False)
    net.set_options(PHYSICS)
    return net

# ── Add Neo4j records (Path / Node / Relationship) to a pyvis net ────────────
def _add_node(net, node, seen_n):
    uid = _nid(node)
    if uid in seen_n: return
    seen_n.add(uid)
    labels = list(node.labels)
    color  = next((COLORS[l] for l in labels if l in COLORS), "#95a5a6")
    size   = next((NODE_SIZES[l] for l in labels if l in NODE_SIZES), 15)
    net.add_node(uid, label=_label(node), title=_title(node), color=color, size=size)

def _add_rel(net, rel, seen_e):
    uid = _rid(rel)
    if uid in seen_e: return
    seen_e.add(uid)
    net.add_edge(_rel_start(rel), _rel_end(rel),
                 label=rel.type, title=rel.type, color="#666688", arrows="to")

def ingest(net, records, seen_n=None, seen_e=None):
    if seen_n is None: seen_n = set()
    if seen_e is None: seen_e = set()
    for record in records:
        for val in record.values():
            if val is None: continue
            # Path: has both .nodes and .relationships
            if hasattr(val, "nodes") and hasattr(val, "relationships"):
                for n in val.nodes:         _add_node(net, n, seen_n)
                for r in val.relationships: _add_rel(net, r, seen_e)
            elif hasattr(val, "labels"):    # Node
                _add_node(net, val, seen_n)
            elif hasattr(val, "type"):      # Relationship
                _add_rel(net, val, seen_e)

def save(net, fname, desc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    net.write_html(str(OUT_DIR / fname))
    print(f"  OK  {fname}   {desc}")

# =============================================================================
#  SECTION 1 — GRAPH OVERVIEW
# =============================================================================

def s1_3_whole_graph_sample(s):
    """[1.3] Quick whole-graph sample."""
    net = make_net()
    ingest(net, s.run("MATCH p = (n)-[r]->(m) RETURN p LIMIT 80"))
    save(net, "s1_3_whole_graph_sample.html", "[1.3] Mixed subgraph sample (80 edges)")

# =============================================================================
#  SECTION 4 — AST STRUCTURE
# =============================================================================

def s4_4_ast_bitwise_full(s):
    """[4.4] Full AST for Bitwise.java (7 nodes)."""
    q = """
    MATCH p = (f:File)-[:HAS_AST]->(r:ASTNode)-[:AST_CHILD*]->(c:ASTNode)
    WHERE f.id = 'src/main/groovy/lang/Bitwise.java'
    RETURN p
    UNION
    MATCH p = (f:File)-[:HAS_AST]->(r:ASTNode)
    WHERE f.id = 'src/main/groovy/lang/Bitwise.java'
    RETURN p
    """
    net = make_net("700px")
    ingest(net, s.run(q))
    save(net, "s4_4_ast_bitwise_full.html", "[4.4] Complete AST of Bitwise.java (7 nodes)")

def s4_5_ast_groovyc_depth3(s):
    """[4.5] AST for Groovyc.java 3 levels deep (923 nodes total)."""
    q = """
    MATCH p = (f:File)-[:HAS_AST]->(r:ASTNode)-[:AST_CHILD*..3]->(c:ASTNode)
    WHERE f.id = 'src/main/org/codehaus/groovy/ant/Groovyc.java'
    RETURN p
    UNION
    MATCH p = (f:File)-[:HAS_AST]->(r:ASTNode)
    WHERE f.id = 'src/main/org/codehaus/groovy/ant/Groovyc.java'
    RETURN p
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s4_5_ast_groovyc_depth3.html", "[4.5] AST of Groovyc.java (depth 3)")

# =============================================================================
#  SECTION 5 — CROSS-ENTITY
# =============================================================================

def s5_2_commit_file_ast_chain(s):
    """[5.2] Commit -> File -> AST root path."""
    q = """
    MATCH p = (c:Commit)-[:ADDED]->(f:File)-[:HAS_AST]->(r:ASTNode)
    RETURN p LIMIT 20
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s5_2_commit_file_ast_chain.html",
         "[5.2] Commit->File->AST root (20 paths)")

# =============================================================================
#  SECTION 7 — WHOLE-GRAPH VISUALIZATIONS
# =============================================================================

def s7_1_schema_map(s):
    """[7.1] One example path per relationship type — schema overview."""
    q = """
    MATCH p = (n)-[r]->(m)
    WITH type(r) AS rel_type, collect(p)[0] AS sample_path
    RETURN sample_path AS p
    """
    net = make_net("700px")
    ingest(net, s.run(q))
    save(net, "s7_1_schema_map.html", "[7.1] Graph schema map (one example per rel type)")

def s7_2_branch_commits(s):
    """[7.2] Branch -> Commits timeline spine."""
    q = "MATCH p = (b:Branch)-[:INSIDE_BRANCH]->(c:Commit) RETURN p LIMIT 60"
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_2_branch_commits.html", "[7.2] Branch -> Commits spine (60 commits)")

def s7_3_developer_universe(s):
    """[7.3] All developers + their commits."""
    q = "MATCH p = (d:Developer)<-[:AUTHORED_BY]-(c:Commit) RETURN p LIMIT 300"
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_3_developer_universe.html",
         "[7.3] Developer universe (all devs, 300 commits)")

def s7_4_issue_network(s):
    """[7.4] Commits linked to the issues they fixed."""
    q = "MATCH p = (c:Commit)-[:FIXES_ISSUE]->(i:Issue) RETURN p LIMIT 200"
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_4_issue_network.html", "[7.4] Issue network (200 commit->issue links)")

def s7_5_file_ast_roots(s):
    """[7.5] All 120 base-snapshot File nodes + their AST roots."""
    q = "MATCH p = (f:File)-[:HAS_AST]->(r:ASTNode) RETURN p"
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_5_file_ast_roots.html", "[7.5] 120 Files each connected to their AST root")

def s7_7_file_lifecycle_invoker(s):
    """[7.7] Every commit that ever touched Invoker.java."""
    q = """
    MATCH p = (d:Developer)<-[:AUTHORED_BY]-(c:Commit)-[rel]->(f:File)
    WHERE f.id = 'src/main/org/codehaus/groovy/runtime/Invoker.java'
      AND type(rel) IN ['ADDED','MODIFIED','DELETED','RENAMED_FROM','RENAMED_TO']
    RETURN p LIMIT 80
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_7_file_lifecycle_invoker.html",
         "[7.7] Full lifecycle of Invoker.java (all commits)")

def s7_8_developer_impact(s):
    """[7.8] One developer's full impact across all files."""
    q = """
    MATCH p = (d:Developer)<-[:AUTHORED_BY]-(c:Commit)-[rel]->(f:File)
    WHERE toLower(d.id) CONTAINS 'guillaume'
      AND type(rel) IN ['ADDED','MODIFIED','DELETED']
    RETURN p LIMIT 100
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_8_developer_impact.html",
         "[7.8] Developer impact - Guillaume's commits and files")

def s7_9_commit_file_ast_history(s):
    """[7.9] Base files: ADDED commit + MODIFIED commits + AST root."""
    q = """
    MATCH p1 = (c1:Commit)-[:ADDED]->(f:File)-[:HAS_AST]->(r:ASTNode)
    MATCH p2 = (c2:Commit)-[:MODIFIED]->(f)
    RETURN p1, p2 LIMIT 60
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_9_commit_file_ast_history.html",
         "[7.9] Base files with commit history + AST root")

def s7_10_issue_fix_chain(s):
    """[7.10] Issue -> Commit -> Developer -> File (end-to-end bug-fix chain)."""
    q = """
    MATCH p1 = (i:Issue)<-[:FIXES_ISSUE]-(c:Commit)-[:AUTHORED_BY]->(d:Developer)
    MATCH p2 = (c)-[rel]->(f:File)
    WHERE type(rel) IN ['MODIFIED','ADDED']
    RETURN p1, p2 LIMIT 50
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_10_issue_fix_chain.html",
         "[7.10] Issue->Commit->Developer->File bug-fix chain")

def s7_11_churned_file_ast(s):
    """[7.11] Most-churned base file + its AST (2 levels) + all its commits."""
    q = """
    MATCH (c:Commit)-[:MODIFIED]->(f:File)-[:HAS_AST]->(:ASTNode)
    WITH f, count(c) AS churn ORDER BY churn DESC LIMIT 1
    MATCH p1 = (c2:Commit)-[:MODIFIED]->(f)
    MATCH p2 = (f)-[:HAS_AST]->(r:ASTNode)-[:AST_CHILD*..2]->(child:ASTNode)
    RETURN p1, p2
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_11_churned_file_ast.html",
         "[7.11] Most-churned base file + AST (depth 2) + all commits")

def s7_12_commit_neighbourhood(s):
    """[7.12] Everything connected to BASE_COMMIT (its full neighbourhood)."""
    q = """
    MATCH (c:Commit {id: $base})
    MATCH p = (c)-[r]-(neighbor)
    RETURN p
    """
    net = make_net()
    ingest(net, s.run(q, base=BASE_COMMIT))
    save(net, "s7_12_commit_neighbourhood.html",
         "[7.12] Complete neighbourhood of BASE_COMMIT")

def s7_13_project_two_hop(s):
    """[7.13] Two hops from the Project node."""
    q = "MATCH p = (proj:Project)-[*1..2]-(n) RETURN p LIMIT 120"
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s7_13_project_two_hop.html",
         "[7.13] Two hops from Project node (graph anchor)")

# =============================================================================
#  SECTION 8 — BASE VERSION GRAPH
# =============================================================================

def s8_1_base_master(s):
    """[8.1] MASTER: Project + Branch + Commit + Developer + 120 Files + AST roots."""
    q = """
    MATCH (c:Commit {id: $base})
    MATCH p1 = (proj:Project)<-[:BELONGS_TO]-(c)
    MATCH p2 = (branch:Branch)<-[:INSIDE_BRANCH]-(c)
    MATCH p3 = (c)-[:AUTHORED_BY]->(d:Developer)
    MATCH p4 = (c)-[:ADDED]->(f:File)
    MATCH p5 = (f)-[:HAS_AST]->(r:ASTNode)
    RETURN p1, p2, p3, p4, p5
    """
    net = make_net()
    ingest(net, s.run(q, base=BASE_COMMIT))
    save(net, "s8_1_base_master.html",
         "[8.1] MASTER: all structural nodes of the base version")

def s8_2_base_commit_files(s):
    """[8.2] BASE_COMMIT + all 120 files it added (no AST)."""
    q = """
    MATCH p = (c:Commit {id: $base})-[:ADDED]->(f:File)
    RETURN p
    """
    net = make_net()
    ingest(net, s.run(q, base=BASE_COMMIT))
    save(net, "s8_2_base_commit_files.html",
         "[8.2] BASE_COMMIT connected to all 120 files it added")

def s8_4_ast_roots_depth1(s):
    """[8.4] AST roots + 1 level of children for every base file."""
    q = """
    MATCH p = (f:File)-[:HAS_AST]->(root:ASTNode)-[:AST_CHILD]->(child:ASTNode)
    RETURN p LIMIT 200
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s8_4_ast_roots_depth1.html",
         "[8.4] All AST roots + first level of children")

def s8_5_ast_small_files(s):
    """[8.5] Complete AST for all files with <= 50 AST nodes."""
    q = """
    MATCH (a:ASTNode)
    WITH a.file AS file, count(a) AS n
    WHERE n <= 50
    WITH collect(file) AS small_files
    MATCH p = (f:File)-[:HAS_AST]->(root:ASTNode)-[:AST_CHILD*]->(child:ASTNode)
    WHERE f.id IN small_files
    RETURN p
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s8_5_ast_small_files.html",
         "[8.5] Full AST for all small files (<=50 nodes each)")

def s8_6_ast_methodnode_full(s):
    """[8.6] Full AST of MethodNode.java (372 nodes)."""
    q = """
    MATCH p = (f:File)-[:HAS_AST]->(root:ASTNode)-[:AST_CHILD*]->(child:ASTNode)
    WHERE f.id = 'src/main/org/codehaus/groovy/ast/MethodNode.java'
    RETURN p
    UNION
    MATCH p = (f:File)-[:HAS_AST]->(root:ASTNode)
    WHERE f.id = 'src/main/org/codehaus/groovy/ast/MethodNode.java'
    RETURN p
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s8_6_ast_methodnode_full.html",
         "[8.6] Full AST of MethodNode.java (372 nodes)")

def s8_7_ast_parser_depth3(s):
    """[8.7] AST of Parser.java (2,695 nodes total) at depth 3."""
    q = """
    MATCH p = (f:File)-[:HAS_AST]->(root:ASTNode)-[:AST_CHILD*..3]->(child:ASTNode)
    WHERE f.id = 'src/main/org/codehaus/groovy/syntax/parser/Parser.java'
    RETURN p
    UNION
    MATCH p = (f:File)-[:HAS_AST]->(root:ASTNode)
    WHERE f.id = 'src/main/org/codehaus/groovy/syntax/parser/Parser.java'
    RETURN p
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s8_7_ast_parser_depth3.html",
         "[8.7] AST of Parser.java (largest file, depth 3)")

def s8_11_master_full(s):
    """[8.11] COMPLETE base version: all structural nodes + all 44,701 ASTNodes."""
    q = """
    MATCH (c:Commit {id: $base})
    MATCH p1 = (proj:Project)<-[:BELONGS_TO]-(c)
    MATCH p2 = (branch:Branch)<-[:INSIDE_BRANCH]-(c)
    MATCH p3 = (c)-[:AUTHORED_BY]->(d:Developer)
    MATCH p4 = (c)-[:ADDED]->(f:File)-[:HAS_AST]->(root:ASTNode)
    MATCH p5 = (root)-[:AST_CHILD*]->(child:ASTNode)
    RETURN p1, p2, p3, p4, p5
    """
    net = make_net()
    ingest(net, s.run(q, base=BASE_COMMIT))
    save(net, "s8_11_master_full.html",
         "[8.11] COMPLETE base version — all 44,701 ASTNodes + structural nodes")

def s8_11_master_compact(s):
    """[8.11 compact] COMPLETE base version skeleton + AST nodes capped at 3000 total."""
    q_struct = """
    MATCH (c:Commit {id: $base})
    MATCH p1 = (proj:Project)<-[:BELONGS_TO]-(c)
    MATCH p2 = (branch:Branch)<-[:INSIDE_BRANCH]-(c)
    MATCH p3 = (c)-[:AUTHORED_BY]->(d:Developer)
    MATCH p4 = (c)-[:ADDED]->(f:File)-[:HAS_AST]->(root:ASTNode)
    RETURN p1, p2, p3, p4
    """
    q_ast_sample = """
    MATCH p = (f:File)-[:HAS_AST]->(root:ASTNode)-[:AST_CHILD*..3]->(child:ASTNode)
    RETURN p LIMIT 3000
    """
    net = make_net()
    sn, se = set(), set()
    ingest(net, s.run(q_struct, base=BASE_COMMIT), sn, se)
    ingest(net, s.run(q_ast_sample), sn, se)
    save(net, "s8_11_master_compact.html",
         "[8.11 compact] Base version skeleton + AST sampled to 3000 nodes")

def s8_10_ast_comparison(s):
    """[8.10] Side-by-side AST: MethodNode.java vs MethodTest.java (depth 3)."""
    q = """
    MATCH p1 = (f1:File)-[:HAS_AST]->(r1:ASTNode)-[:AST_CHILD*..3]->(c1:ASTNode)
    WHERE f1.id = 'src/main/org/codehaus/groovy/ast/MethodNode.java'
    MATCH p2 = (f2:File)-[:HAS_AST]->(r2:ASTNode)-[:AST_CHILD*..3]->(c2:ASTNode)
    WHERE f2.id = 'src/test/org/codehaus/groovy/classgen/MethodTest.java'
    RETURN p1, p2
    """
    net = make_net()
    ingest(net, s.run(q))
    save(net, "s8_10_ast_comparison.html",
         "[8.10] AST comparison: MethodNode.java vs MethodTest.java (depth 3)")

# =============================================================================
#  MAIN
# =============================================================================

EXPORTS = [
    # Section 1
    s1_3_whole_graph_sample,
    # Section 4
    s4_4_ast_bitwise_full,
    s4_5_ast_groovyc_depth3,
    # Section 5
    s5_2_commit_file_ast_chain,
    # Section 7
    s7_1_schema_map,
    s7_2_branch_commits,
    s7_3_developer_universe,
    s7_4_issue_network,
    s7_5_file_ast_roots,
    s7_7_file_lifecycle_invoker,
    s7_8_developer_impact,
    s7_9_commit_file_ast_history,
    s7_10_issue_fix_chain,
    s7_11_churned_file_ast,
    s7_12_commit_neighbourhood,
    s7_13_project_two_hop,
    # Section 8
    s8_1_base_master,
    s8_2_base_commit_files,
    s8_4_ast_roots_depth1,
    s8_5_ast_small_files,
    s8_6_ast_methodnode_full,
    s8_7_ast_parser_depth3,
    s8_10_ast_comparison,
    s8_11_master_compact,
    s8_11_master_full,
]

def main():
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            print(f"Exporting {len(EXPORTS)} graphs to {OUT_DIR}\n")
            for fn in EXPORTS:
                try:
                    fn(session)
                except Exception as e:
                    print(f"  SKIP  {fn.__name__}  ({e})")
    finally:
        driver.close()
    print(f"\nDone. Open any .html file in {OUT_DIR} in your browser.")

if __name__ == "__main__":
    main()
