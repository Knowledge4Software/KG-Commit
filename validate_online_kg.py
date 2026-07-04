"""
Validate + visualize the ONLINE KG growth (build_online_kg.py output).

Two parts:

A. CORRECTNESS CHECK (the important one)
   For every file evolved during the run, rebuild the ground-truth AST of the
   file at its current stored state (git blob) and compare it to the graph's
   ALIVE nodes. If evolution + re-stamping is correct, the set of alive node
   positions in the graph should equal the ground-truth AST's positions.
   Reports per-file and overall position-set match.

B. VISUALIZATIONS (pyvis HTML in outputs/online_kg/)
   1. A file's evolution: its current alive AST + which commit added/removed
      each touched node (the multi-commit story).
   2. One commit's delta star in AST context.
   3. A new file (born in the stream) with its full AST.

Run:  python validate_online_kg.py
"""

import json
from pathlib import Path
from neo4j import GraphDatabase
from pyvis.network import Network
import build_online_kg as bok

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password1234")
OUT_DIR    = Path("outputs/online_kg"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT       = Path("outputs/online_kg_checkpoint.json")

EDGE_COLOR = {"ADDS":"#2ca02c","REMOVES":"#d62728","UPDATES":"#ff7f0e","MOVES":"#9467bd"}


# ── A. correctness ───────────────────────────────────────────────────────────

def positioned_set(nodes):
    """Set of (line,col) for nodes with a real position."""
    return {(n["line"], n["col"]) for n in nodes if n["line"] > 0}

def check_correctness(session, file_state):
    print("="*70)
    print("CORRECTNESS: graph alive-AST vs ground-truth AST at current state")
    print("="*70)
    base = bok.BASE_COMMIT
    # files that actually evolved (state != BASE)  -> the interesting ones
    evolved = {f: c for f, c in file_state.items() if c != base}
    print(f"{len(evolved)} files evolved past base; checking ground truth...\n")

    results = []
    for f, state_commit in sorted(evolved.items()):
        # graph alive positions
        rows = session.run("""
            MATCH (a:ASTNode {file:$f}) WHERE coalesce(a.alive,true) AND a.pos_line>0
            RETURN a.pos_line AS line, a.pos_col AS col
        """, f=f).data()
        graph_pos = {(r["line"], r["col"]) for r in rows}

        # ground truth from git blob at current state
        ref = state_commit
        blob = bok.git_bytes(ref, f)
        if blob is None:
            continue
        ast = bok.build_full_ast(blob, f)
        if ast is None:
            continue
        gt_pos = positioned_set(ast["nodes"])

        inter = graph_pos & gt_pos
        prec = len(inter)/len(graph_pos) if graph_pos else 0   # graph nodes that are real
        rec  = len(inter)/len(gt_pos) if gt_pos else 0          # real nodes captured
        results.append((f, len(graph_pos), len(gt_pos), prec, rec))

    if not results:
        print("  (no evolved files to check)")
        return []
    results.sort(key=lambda r: r[4])
    print(f"{'recall':>7} {'prec':>6} {'graph':>6} {'truth':>6}  file")
    for f, g, t, p, r in results:
        print(f"{r:7.2f} {p:6.2f} {g:6d} {t:6d}  {f.split('/')[-1]}")
    avg_p = sum(x[3] for x in results)/len(results)
    avg_r = sum(x[4] for x in results)/len(results)
    print(f"\nAVG precision={avg_p:.3f}  recall={avg_r:.3f}  over {len(results)} evolved files")
    return results


# ── A3 (V3): FULL-TREE correctness — includes identifier leaves, multiset, all files ─

def _signature_multiset(nodes):
    """Counter over (ast_type, value, is_leaf) for EVERY node (incl. leaves).

    Unlike positioned_set (structured nodes only, set of (line,col)), this covers
    the ~64% identifier leaves too and is a multiset, so a missing/extra/renamed
    node anywhere in the tree is detected."""
    from collections import Counter
    return Counter((str(n.get("ast_type")), str(n.get("value")), bool(n.get("is_leaf")))
                   for n in nodes)


def check_full_tree(session, file_state, limit=None):
    """For EVERY evolved file, compare the graph's ALIVE subtree to a fresh
    ground-truth parse at the file's current stored state, over the full node
    multiset (type,value,leaf) including leaves. Reports exact-match rate and the
    node coverage backing the claim (answers reviewer concern N4)."""
    print("="*70)
    print("FULL-TREE CORRECTNESS: graph alive-AST vs ground truth (incl. leaves)")
    print("="*70)
    base = bok.BASE_COMMIT
    evolved = {f: c for f, c in file_state.items() if c != base}
    items = sorted(evolved.items())
    if limit:
        items = items[:limit]
    print(f"{len(items)} evolved files; full-multiset check...\n")

    exact = 0; checked = 0; graph_nodes = 0; gt_nodes = 0; mism = []
    for f, state_commit in items:
        rows = session.run("""
            MATCH (a:ASTNode {file:$f}) WHERE coalesce(a.alive,true)
            RETURN a.ast_type AS ast_type, a.value AS value, a.is_leaf AS is_leaf
        """, f=f).data()
        blob = bok.git_bytes(state_commit, f)
        if blob is None:
            continue
        ast = bok.build_full_ast(blob, f)
        if ast is None:
            continue
        checked += 1
        g_ms = _signature_multiset(rows)
        t_ms = _signature_multiset(ast["nodes"])
        graph_nodes += sum(g_ms.values()); gt_nodes += sum(t_ms.values())
        if g_ms == t_ms:
            exact += 1
        else:
            # symmetric difference magnitude = number of node slots that disagree
            diff = sum((g_ms - t_ms).values()) + sum((t_ms - g_ms).values())
            mism.append((f.split("/")[-1], len(list((g_ms - t_ms).elements())),
                         len(list((t_ms - g_ms).elements())), diff))

    print(f"exact full-tree match: {exact}/{checked} files "
          f"({(exact/checked*100 if checked else 0):.1f}%)")
    print(f"nodes compared: graph={graph_nodes:,}  ground-truth={gt_nodes:,}")
    if mism:
        mism.sort(key=lambda r: -r[3])
        print(f"\n{len(mism)} files with a mismatch (top 10 by disagreement):")
        print(f"{'graph+':>7} {'truth+':>7} {'diff':>6}  file")
        for name, gextra, textra, diff in mism[:10]:
            print(f"{gextra:7d} {textra:7d} {diff:6d}  {name}")
    return dict(exact=exact, checked=checked, graph_nodes=graph_nodes,
                gt_nodes=gt_nodes, mismatched=len(mism))


# ── B. visualizations ────────────────────────────────────────────────────────

def viz_file_evolution(session, file_id):
    net = Network(height="800px", width="100%", bgcolor="#fff", directed=True)
    net.barnes_hut(gravity=-6000, spring_length=110)
    fname = file_id.split("/")[-1]
    # commits that touched this file's nodes + the touched nodes
    rows = session.run("""
        MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode {file:$f})
        RETURN c.id AS commit, c.buggy AS buggy, type(r) AS et,
               a.id AS aid, a.ast_type AS at, a.value AS v, coalesce(a.alive,true) AS alive
        LIMIT 250
    """, f=file_id).data()
    commits = {}
    for r in rows:
        cid = r["commit"]
        if cid not in commits:
            commits[cid] = True
            net.add_node("C:"+cid, label=("BUG " if r["buggy"] else "")+cid[:7],
                         shape="star", size=26,
                         color="#c0392b" if r["buggy"] else "#1f77b4")
        aid = r["aid"]
        lbl = r["at"] + (f"\n{str(r['v'])[:14]}" if r["v"] else "")
        net.add_node(aid, label=lbl, shape="dot", size=12,
                     color=EDGE_COLOR[r["et"]] if r["alive"] else "#999",
                     title=f"{r['at']} alive={r['alive']}")
        net.add_edge("C:"+cid, aid, color=EDGE_COLOR[r["et"]], label=r["et"])
    out = OUT_DIR / f"evolution_{fname.replace('.java','')}.html"
    net.write_html(str(out), notebook=False)
    return out, len(commits), len(rows)

def viz_new_file(session, file_id):
    net = Network(height="800px", width="100%", bgcolor="#fff", directed=True)
    net.barnes_hut(gravity=-5000, spring_length=90)
    rows = session.run("""
        MATCH (f:File {id:$f})-[:HAS_AST]->(root:ASTNode)
        MATCH p=(root)-[:AST_CHILD*0..4]->(a:ASTNode)
        WITH collect(DISTINCT a) AS ns
        UNWIND ns AS a
        OPTIONAL MATCH (a)-[:AST_CHILD]->(ch:ASTNode)
        RETURN a.id AS id, a.ast_type AS at, a.value AS v, ch.id AS child
    """, f=file_id).data()
    seen=set()
    for r in rows:
        for nid,at,v in [(r["id"],r["at"],r["v"])]:
            if nid not in seen:
                seen.add(nid)
                net.add_node(nid, label=(at or "?")+(f"\n{str(v)[:12]}" if v else ""),
                             shape="dot", size=11, color="#2ca02c")
        if r["child"]:
            if r["child"] not in seen:
                seen.add(r["child"]); net.add_node(r["child"], label="", size=9, color="#2ca02c")
            net.add_edge(r["id"], r["child"], color="#bbb")
    out = OUT_DIR / f"newfile_{file_id.split('/')[-1].replace('.java','')}.html"
    net.write_html(str(out), notebook=False)
    return out, len(seen)


def main():
    file_state = json.loads(CKPT.read_text())["file_state"] if CKPT.exists() else {}
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as s:
        # ---- summary ----
        print("="*70); print("ONLINE KG — SUMMARY"); print("="*70)
        for et in EDGE_COLOR:
            n=s.run(f"MATCH ()-[r:{et}]->() RETURN count(r) AS n").single()["n"]
            print(f"  {et:8s}: {n}")
        al=s.run("MATCH (a:ASTNode) WHERE a.alive=false RETURN count(a) AS n").single()["n"]
        tracked=len(file_state)
        commits_done=s.run("MATCH (c:Commit)-[:ADDS|REMOVES|UPDATES|MOVES]->() RETURN count(DISTINCT c) AS n").single()["n"]
        print(f"  removed (alive=false): {al}")
        print(f"  files tracked: {tracked}   commits with deltas: {commits_done}")

        # ---- correctness ----
        results = check_correctness(s, file_state)
        full = check_full_tree(s, file_state)

        # ---- pick interesting files for viz ----
        print("\n" + "="*70); print("VISUALIZATIONS"); print("="*70)
        # most-evolved file (most distinct commits touching it)
        top = s.run("""
            MATCH (c:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
            WITH a.file AS f, count(DISTINCT c) AS commits
            RETURN f, commits ORDER BY commits DESC LIMIT 1
        """).single()
        if top:
            out,nc,ne = viz_file_evolution(s, top["f"])
            print(f"  evolution: {out.name}  ({nc} commits, {ne} touched-node edges)  {top['f']}")

        # a new file born in the stream
        nf = s.run("""
            MATCH (c:Commit {in_jit:true})-[:ADDED]->(f:File)-[:HAS_AST]->()
            WHERE f.id <> $base RETURN f.id AS f LIMIT 1
        """, base="").data()
        newfiles = s.run("""
            MATCH (c:Commit {in_jit:true})-[:ADDED]->(f:File)-[:HAS_AST]->(:ASTNode {is_delta:true})
            RETURN DISTINCT f.id AS f LIMIT 1
        """).data()
        # fall back: any file whose creating commit is not base
        cand = s.run("""
            MATCH (f:File)-[:HAS_AST]->(r:ASTNode {is_delta:true})
            RETURN DISTINCT f.id AS f LIMIT 1
        """).data()
        pick = (newfiles or cand)
        if pick:
            out,n = viz_new_file(s, pick[0]["f"])
            print(f"  new-file AST: {out.name}  ({n} nodes)  {pick[0]['f']}")

        Path("outputs/online_kg_validation.json").write_text(json.dumps({
            "positioned_check": {
                "evolved_files_checked": len(results),
                "avg_precision": round(sum(x[3] for x in results)/len(results),3) if results else None,
                "avg_recall": round(sum(x[4] for x in results)/len(results),3) if results else None,
            },
            "full_tree_check": full,
        }, indent=2))
        print(f"\n  HTML -> {OUT_DIR}")
    driver.close()


if __name__ == "__main__":
    main()
