"""
Standalone visual gallery of the KG (pyvis HTML — open in any browser, no Neo4j
Browser needed). Covers BOTH layers:

  1. base_skeleton.html   - base commit -> sample files -> AST roots (the base
                            "huge graph" structure, readable subset)
  2. file_ast_<name>.html - one file's FULL AST tree (a base component drilled in)
  3. commit_delta_<sha>.html - a commit's delta IN AST CONTEXT (the commit-level
                            delta graph: ADDS/REMOVES/UPDATES/MOVES on real nodes)
  4. evolution_<name>.html (also produced by validate_online_kg.py)

Run:  python viz_kg.py
"""
from pathlib import Path
from neo4j import GraphDatabase
from pyvis.network import Network

NEO4J_URI="bolt://localhost:7687"; NEO4J_AUTH=("neo4j","password1234")
BASE="408b29851d7bbe4d343340832297e4be7e0c5578"
OUT=Path("outputs/kg_gallery"); OUT.mkdir(parents=True, exist_ok=True)
EDGE_COLOR={"ADDS":"#2ca02c","REMOVES":"#d62728","UPDATES":"#ff7f0e","MOVES":"#9467bd"}
GROUP_COLOR={"declaration":"#1f77b4","statement":"#ff7f0e","expression":"#2ca02c",
             "literal":"#9467bd","leaf":"#8c8c8c"}

def net(h="800px"):
    n=Network(height=h,width="100%",bgcolor="#ffffff",font_color="#222",directed=True)
    n.barnes_hut(gravity=-7000,spring_length=110); return n

def base_skeleton(s, n_files=18):
    g=net()
    g.add_node("BASE",label="BASE\nCommit",shape="star",size=44,color="#c0392b",
               title=f"base commit {BASE[:10]}")
    files=s.run("""MATCH (c:Commit {id:$b})-[:ADDED]->(f:File)-[:HAS_AST]->(r:ASTNode)
                   RETURN f.id AS f, r.id AS root, count{(:ASTNode {file:f.id})} AS sz
                   ORDER BY sz DESC LIMIT $n""", b=BASE, n=n_files).data()
    for r in files:
        fn=r["f"].split("/")[-1]
        g.add_node("F:"+r["f"],label=fn,shape="box",size=20,color="#34495e",
                   title=f"{r['f']}\n{r['sz']} AST nodes")
        g.add_edge("BASE","F:"+r["f"],color="#bbb",label="ADDED")
        g.add_node(r["root"],label="AST root",shape="dot",size=12,color="#1f77b4")
        g.add_edge("F:"+r["f"],r["root"],color="#3498db",label="HAS_AST")
    out=OUT/"base_skeleton.html"; g.write_html(str(out),notebook=False); return out,len(files)

def file_ast(s, file_substr, depth=4):
    g=net()
    rows=s.run("""MATCH (f:File)-[:HAS_AST]->(root:ASTNode)
                  WHERE f.id CONTAINS $q
                  MATCH p=(root)-[:AST_CHILD*0..%d]->(a:ASTNode)
                  WITH DISTINCT a LIMIT 600
                  OPTIONAL MATCH (a)-[:AST_CHILD]->(ch:ASTNode)
                  RETURN a.id AS id, a.ast_type AS t, a.group AS grp, a.value AS v,
                         ch.id AS child""" % depth, q=file_substr).data()
    seen=set()
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            g.add_node(r["id"],label=(r["t"] or "?")+(f"\n{str(r['v'])[:14]}" if r["v"] else ""),
                       shape="dot",size=11,color=GROUP_COLOR.get(r["grp"],"#888"),
                       title=f"{r['t']} [{r['grp']}]")
        if r["child"]:
            if r["child"] not in seen:
                seen.add(r["child"]); g.add_node(r["child"],label="",size=8,color="#ccc")
            g.add_edge(r["id"],r["child"],color="#ddd")
    fn=file_substr.replace(".java","")
    out=OUT/f"file_ast_{fn}.html"; g.write_html(str(out),notebook=False); return out,len(seen)

def commit_delta(s, commit_prefix):
    g=net()
    rows=s.run("""MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
                  WHERE c.id STARTS WITH $p
                  RETURN c.id AS commit, c.buggy AS buggy, type(r) AS et,
                         a.id AS aid, a.ast_type AS t, a.value AS v LIMIT 200""",
               p=commit_prefix).data()
    if not rows: return None,0
    cid=rows[0]["commit"]; buggy=rows[0]["buggy"]
    g.add_node("C",label=("BUG " if buggy else "")+cid[:8],shape="star",size=40,
               color="#c0392b" if buggy else "#1f77b4")
    touched=[]
    for r in rows:
        g.add_node(r["aid"],label=(r["t"] or "?")+(f"\n{str(r['v'])[:12]}" if r["v"] else ""),
                   shape="dot",size=16,color=EDGE_COLOR[r["et"]],title=f"{r['et']} {r['t']}")
        g.add_edge("C",r["aid"],color=EDGE_COLOR[r["et"]],label=r["et"],width=2)
        touched.append(r["aid"])
    # one hop of AST context around touched nodes
    ctx=s.run("""MATCH (a:ASTNode)-[:AST_CHILD]-(nb:ASTNode)
                 WHERE a.id IN $ids RETURN DISTINCT a.id AS a, nb.id AS nb,
                 nb.ast_type AS t LIMIT 200""", ids=touched).data()
    for r in ctx:
        if r["nb"] not in touched:
            g.add_node(r["nb"],label=r["t"] or "",shape="dot",size=9,color="#ddd")
        g.add_edge(r["a"],r["nb"],color="#e0e0e0")
    out=OUT/f"commit_delta_{cid[:8]}.html"; g.write_html(str(out),notebook=False)
    return out,len(touched)

def main():
    d=GraphDatabase.driver(NEO4J_URI,auth=NEO4J_AUTH)
    with d.session() as s:
        print("Generating gallery ->", OUT)
        o,n=base_skeleton(s); print(f"  base_skeleton: {o.name} ({n} files)")
        for fq in ["MethodNode.java","Bitwise.java"]:
            o,n=file_ast(s,fq); print(f"  file_ast: {o.name} ({n} nodes)")
        # pick commits with a good delta mix
        cands=s.run("""MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->()
                       WITH c, count(r) AS n, count(DISTINCT type(r)) AS k
                       WHERE n>6 AND n<60 RETURN c.id AS id ORDER BY k DESC, n LIMIT 3""").data()
        for r in cands:
            o,n=commit_delta(s,r["id"][:8])
            if o: print(f"  commit_delta: {o.name} ({n} changed nodes)")
    d.close()
    print(f"\nOpen the .html files in {OUT} in any browser.")

if __name__=="__main__":
    main()
