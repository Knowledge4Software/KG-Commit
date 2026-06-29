"""
Commit-level KG story visualizations. For each example commit, produces:

  step_<sha>_1_file_ast.html  - the file's FULL AST BEFORE the commit (the
                                previous structure).
  step_<sha>_2_delta.html     - the commit + its delta graph + details
                                (author, issue, parent, buggy label, metrics).
  step_<sha>_3_aftertree.html - the file's AST AFTER the commit.
  step_<sha>_OVERLAY.html     - ★ ONE picture: the previous structure WITH the
                                commit's modifications layered on it — existing
                                nodes (grey), ADDED nodes+edges (green), REMOVED
                                (red), UPDATED (orange), anchored to the base
                                commit + File node. This is the view for
                                inspecting new relationships/modifications.

Plus base_graph_120.html once (base commit -> 120 files -> AST roots).

Several diverse examples are auto-picked (small change, varied change, buggy,
new-file). Override a single one:  python viz_commit_step.py <commit_prefix>
"""
import sys
from pathlib import Path
from neo4j import GraphDatabase
from pyvis.network import Network
import build_online_kg as bok
import online_ast_diff as oad

NEO4J_URI="bolt://localhost:7687"; NEO4J_AUTH=("neo4j","password1234")
BASE="408b29851d7bbe4d343340832297e4be7e0c5578"
OUT=Path("outputs/kg_story"); OUT.mkdir(parents=True, exist_ok=True)
EDGE_COLOR={"ADDS":"#2ca02c","REMOVES":"#d62728","UPDATES":"#ff7f0e","MOVES":"#9467bd"}
GROUP_COLOR={"declaration":"#1f77b4","statement":"#ff7f0e","expression":"#2ca02c",
             "literal":"#9467bd","leaf":"#9aa0a6"}

def net(h="850px"):
    n=Network(height=h,width="100%",bgcolor="#ffffff",font_color="#222",directed=True)
    n.barnes_hut(gravity=-7000,spring_length=120); return n

# ── base graph (120 files) ───────────────────────────────────────────────────
def base_graph(s):
    g=net()
    g.add_node("BASE",label="BASE commit",shape="star",size=46,color="#c0392b",
               title=f"base snapshot {BASE[:12]}")
    files=s.run("""MATCH (c:Commit {id:$b})-[:ADDED]->(f:File)-[:HAS_AST]->(r:ASTNode)
        RETURN f.id AS f, r.id AS root, count{(:ASTNode {file:f.id})} AS sz ORDER BY f.id""",
        b=BASE).data()
    for r in files:
        fid="F:"+r["f"]
        g.add_node(fid,label=r["f"].split("/")[-1],shape="box",size=14,color="#34495e",
                   title=f"{r['f']}\n{r['sz']} AST nodes")
        g.add_edge("BASE",fid,color="#cfd8dc")
        g.add_node(r["root"],label="",shape="dot",size=6,color="#1f77b4")
        g.add_edge(fid,r["root"],color="#90caf9")
    out=OUT/"base_graph_120.html"; g.write_html(str(out),notebook=False); return out,len(files)

# ── full AST of a blob (before or after) ─────────────────────────────────────
def _ast_view(blob, file_id, fname):
    ast=bok.build_full_ast(blob, file_id) if blob else None
    g=net()
    if ast:
        for n in ast["nodes"]:
            g.add_node(n["id"],label=(n["ast_type"] or "?")+(f"\n{str(n['value'])[:14]}" if n["value"] else ""),
                       shape="dot",size=10,color=GROUP_COLOR.get(n["group"],"#888"),
                       title=f"{n['ast_type']} [{n['group']}] line {n['line']}")
        for e in ast["edges"]:
            g.add_edge(e["parent"],e["child"],color="#dde")
    g.write_html(str(fname),notebook=False)
    return len(ast["nodes"]) if ast else 0

# ── commit + delta + details ─────────────────────────────────────────────────
def commit_delta(s, commit, file_id, fname):
    g=net()
    info=s.run("""MATCH (c:Commit {id:$c}) RETURN c.buggy AS buggy, c.is_fix AS fix,
        c.la AS la, c.ld AS ld, c.nf AS nf, c.ent AS ent, c.jit_year AS year,
        c.message AS msg""",c=commit).single()
    title=(f"commit {commit[:10]} year={info['year']} buggy={info['buggy']} "
           f"is_fix={info['fix']}\nla={info['la']} ld={info['ld']} nf={info['nf']} "
           f"ent={round(info['ent'] or 0,2)}\n{(info['msg'] or '')[:80]}")
    g.add_node("C",label=("BUG " if info["buggy"] else "")+commit[:8],shape="star",size=44,
               color="#c0392b" if info["buggy"] else "#1f77b4",title=title)
    g.add_node("FILE",label=file_id.split("/")[-1],shape="box",size=20,color="#34495e",title=file_id)
    g.add_edge("C","FILE",color="#888",label="MODIFIED")
    for q,lbl,col,cap in [
        ("MATCH (c:Commit {id:$c})-[:AUTHORED_BY]->(d:Developer) RETURN d.id AS x","author","#16a085","by"),
        ("MATCH (c:Commit {id:$c})-[:FIXES_ISSUE]->(i:Issue) RETURN i.id AS x","issue","#8e44ad","fixes"),
        ("MATCH (p:Commit)-[:PARENT_OF]->(c:Commit {id:$c}) RETURN p.id AS x","parent","#7f8c8d","parent")]:
        for r in s.run(q,c=commit).data()[:2]:
            nid=lbl+":"+str(r["x"]); g.add_node(nid,label=str(r["x"])[:18],shape="dot",size=14,color=col)
            g.add_edge("C",nid,color=col,label=cap)
    rows=s.run("""MATCH (c:Commit {id:$c})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode {file:$f})
        RETURN type(r) AS et, a.id AS aid, a.ast_type AS t, a.value AS v,
               r.old_value AS ov, r.new_value AS nv""",c=commit,f=file_id).data()
    touched=[]
    for r in rows:
        tip=f"{r['et']} {r['t']}"+(f"\n{r['ov']} -> {r['nv']}" if r["et"]=="UPDATES" else "")
        g.add_node(r["aid"],label=(r["t"] or "?"),shape="dot",size=15,color=EDGE_COLOR[r["et"]],title=tip)
        g.add_edge("C",r["aid"],color=EDGE_COLOR[r["et"]],label=r["et"],width=2); touched.append(r["aid"])
    if touched:
        for r in s.run("""MATCH (a:ASTNode)-[:AST_CHILD]-(nb:ASTNode) WHERE a.id IN $ids
            RETURN DISTINCT a.id AS a, nb.id AS nb, nb.ast_type AS t LIMIT 200""",ids=touched).data():
            if r["nb"] not in touched: g.add_node(r["nb"],label=r["t"] or "",shape="dot",size=8,color="#dadada")
            g.add_edge(r["a"],r["nb"],color="#e8e8e8")
    g.write_html(str(fname),notebook=False); return len(touched)

# ── ★ OVERLAY: previous structure + this commit's modifications in ONE view ──
def overlay(s, commit, file_id, fname):
    before=bok.git_bytes(f"{commit}^1", file_id)
    after =bok.git_bytes(commit, file_id)
    if after is None: return 0
    new_file=(before is None)
    bast=bok.build_full_ast(before, file_id) if before else None
    aast=bok.build_full_ast(after, file_id)
    info=s.run("""MATCH (c:Commit {id:$c}) RETURN c.buggy AS b, c.la AS la, c.ld AS ld,
        c.is_fix AS f, c.message AS m""",c=commit).single()
    g=net()
    # anchor: BASE -> File
    g.add_node("BASE",label="BASE",shape="star",size=26,color="#c0392b")
    g.add_node("FILE",label=file_id.split("/")[-1],shape="box",size=22,color="#34495e",title=file_id)
    g.add_edge("BASE","FILE",color="#cfd8dc",label="base file")
    # modifying commit
    g.add_node("C",label=("BUG " if info["b"] else "")+commit[:8],shape="star",size=40,
               color="#c0392b" if info["b"] else "#1f77b4",
               title=f"commit {commit[:10]}\nbuggy={info['b']} is_fix={info['f']} "
                     f"la={info['la']} ld={info['ld']}\n{(info['m'] or '')[:70]}")
    g.add_edge("C","FILE",color="#888",label="MODIFIED")

    if new_file:
        # whole file is new: all green
        for n in aast["nodes"]:
            g.add_node("N:"+n["id"],label=n["ast_type"] or "?",shape="dot",size=11,color="#2ca02c",
                       title=f"ADDED {n['ast_type']}")
        for e in aast["edges"]:
            g.add_edge("N:"+e["parent"],"N:"+e["child"],color="#2ca02c")
        g.add_edge("FILE","N:"+aast["root"],color="#90caf9",label="HAS_AST (new)")
        g.add_edge("C","N:"+aast["root"],color="#2ca02c",label="ADDS file",width=2)
        g.write_html(str(fname),notebook=False); return len(aast["nodes"])

    d=oad.diff_sources(before, after, file_id)
    if d is None: return 0
    after_by={n["lid"]:n for n in d["after_nodes"]}
    a2b={a:b for b,a,_,_ in d["matched"]}
    updated={b for b,a,vc,_ in d["matched"] if vc}
    upd_new={b:nv for b,a,vc,nv in d["matched"] if vc}
    deleted=set(d["deleted"]); inserted=set(d["inserted"])

    # previous structure (before AST): grey, mark removed=red, updated=orange
    bnodes={n["id"]:n for n in bast["nodes"]}
    for nid,n in bnodes.items():
        if nid in deleted:
            col,tip="#d62728","REMOVED "+str(n["ast_type"])
        elif nid in updated:
            col,tip="#ff7f0e",f"UPDATED {n['ast_type']}\n{n['value']} -> {upd_new.get(nid)}"
        else:
            col,tip="#c8ccd0",f"{n['ast_type']} (unchanged)"
        g.add_node("B:"+nid,label=n["ast_type"] or "?",shape="dot",size=10,color=col,title=tip)
    for e in bast["edges"]:
        col="#d62728" if (e["child"] in deleted or e["parent"] in deleted) else "#e2e4e8"
        g.add_edge("B:"+e["parent"],"B:"+e["child"],color=col)
    g.add_edge("FILE","B:"+bast["root"],color="#90caf9",label="HAS_AST")

    # added nodes (green) wired into the previous structure via their parent
    def gid(after_lid):
        if after_lid in inserted: return "A:"+after_lid          # new node id
        if after_lid in a2b:      return "B:"+a2b[after_lid]      # existing before node
        return None
    for alid in inserted:
        an=after_by[alid]
        g.add_node("A:"+alid,label=an["ast_type"] or "?",shape="dot",size=12,color="#2ca02c",
                   title=f"ADDED {an['ast_type']}"+(f"\n{str(an['value'])[:20]}" if an["value"] else ""))
    for alid in inserted:                                        # green NEW edges parent->new
        an=after_by[alid]; pg=gid(an["parent_lid"])
        if pg: g.add_edge(pg,"A:"+alid,color="#2ca02c",width=2)

    # commit -> changed nodes (delta edges)
    for nid in deleted: g.add_edge("C","B:"+nid,color="#d62728",label="REMOVES")
    for nid in updated: g.add_edge("C","B:"+nid,color="#ff7f0e",label="UPDATES")
    for alid in list(inserted)[:60]: g.add_edge("C","A:"+alid,color="#2ca02c",label="ADDS")
    g.write_html(str(fname),notebook=False)
    return len(inserted)+len(deleted)+len(updated)

# ── example selection ────────────────────────────────────────────────────────
def base_files(s):
    return set(r["f"] for r in s.run(
        "MATCH (:Commit {id:$b})-[:ADDED]->(f:File)-[:HAS_AST]->() RETURN f.id AS f",b=BASE).data())

def pick_examples(s):
    bf=base_files(s)
    sizes={r["f"]:r["sz"] for r in s.run("""MATCH (f:File)-[:HAS_AST]->() WHERE f.id IN $bf
        RETURN f.id AS f, count{(:ASTNode {file:f.id})} AS sz""",bf=list(bf)).data()}
    out=[]; seen=set()
    def is_modify(c,f):                              # parent version exists => real modify
        return bok.git_bytes(f"{c}^1", f) is not None
    def add_first(rows,kind,require_modify=True):
        for r in rows:
            c,f=r["c"],r["f"]
            if (c,f) in seen: continue
            if require_modify and not is_modify(c,f): continue
            out.append((c,f,kind)); seen.add((c,f)); return
    # 1) varied change: has UPDATES and/or MOVES, single file, moderate AST
    add_first(s.run("""MATCH (c:Commit {in_jit:true})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
        WITH c, a.file AS f, count(r) AS n, collect(DISTINCT type(r)) AS k
        WITH c, collect({f:f,n:n,k:k}) AS fs WHERE size(fs)=1
          AND ('UPDATES' IN fs[0].k OR 'MOVES' IN fs[0].k) AND fs[0].n>=6 AND fs[0].n<=40
        WITH c, fs[0].f AS f WHERE count{(:ASTNode {file:f})} <= 260
        RETURN c.id AS c, f ORDER BY count{(:ASTNode {file:f})} LIMIT 12""").data(), "varied")
    # 2) buggy commit (any tracked file), moderate AST, real modification
    add_first(s.run("""MATCH (c:Commit {in_jit:true, buggy:true})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
        WITH c, a.file AS f, count(r) AS n WHERE n>=5 AND n<=60
        WITH c, f WHERE count{(:ASTNode {file:f})} <= 300
        RETURN c.id AS c, f ORDER BY count{(:ASTNode {file:f})} LIMIT 15""").data(), "buggy")
    # 3) small clean modify on a moderate base file
    small=[r for r in s.run("""MATCH (c:Commit {in_jit:true})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
        WITH c, a.file AS f, count(r) AS n, count(DISTINCT type(r)) AS k
        WITH c, collect({f:f,n:n,k:k}) AS fs WHERE size(fs)=1 AND fs[0].n>=6 AND fs[0].n<=24 AND fs[0].k>=2
        RETURN c.id AS c, fs[0].f AS f ORDER BY fs[0].n LIMIT 40""").data()
        if r["f"] in bf and 50<=sizes.get(r["f"],9999)<=200]
    add_first(small, "small")
    # 4) a commit that ADDS a brand-new file (full AST born in the stream)
    add_first(s.run("""MATCH (c:Commit {in_jit:true})-[:ADDED]->(f:File)-[:HAS_AST]->()
        WHERE NOT (:Commit {id:$b})-[:ADDED]->(f)
        RETURN c.id AS c, f.id AS f LIMIT 5""",b=BASE).data(), "newfile", require_modify=False)
    return out

def make_story(s, commit, file_id, kind):
    sha=commit[:8]
    n1=_ast_view(bok.git_bytes(f"{commit}^1",file_id) or bok.git_bytes(BASE,file_id),
                 file_id, OUT/f"step_{sha}_1_file_ast.html")
    n2=commit_delta(s, commit, file_id, OUT/f"step_{sha}_2_delta.html")
    n3=_ast_view(bok.git_bytes(commit,file_id), file_id, OUT/f"step_{sha}_3_aftertree.html")
    n4=overlay(s, commit, file_id, OUT/f"step_{sha}_OVERLAY.html")
    print(f"  [{kind}] {sha} {file_id.split('/')[-1]}: before={n1} delta={n2} after={n3} overlay-changes={n4}")
    print(f"     -> step_{sha}_1_file_ast.html | _2_delta.html | _3_aftertree.html | _OVERLAY.html")

def main():
    d=GraphDatabase.driver(NEO4J_URI,auth=NEO4J_AUTH)
    with d.session() as s:
        o,n=base_graph(s); print(f"base graph: {o.name} ({n} files)\n")
        if len(sys.argv)>1:
            row=s.run("""MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
                WHERE c.id STARTS WITH $p RETURN c.id AS c, a.file AS f, count(r) AS n
                ORDER BY n DESC LIMIT 1""",p=sys.argv[1]).single()
            if row: make_story(s,row["c"],row["f"],"custom")
        else:
            ex=pick_examples(s)
            print(f"picked {len(ex)} example commits:\n")
            for c,f,k in ex: make_story(s,c,f,k)
    d.close()
    print(f"\nAll HTML in {OUT}. Open *_OVERLAY.html to see modifications on the previous structure.")

if __name__=="__main__":
    main()
