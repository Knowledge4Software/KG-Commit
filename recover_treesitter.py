"""
V2 / A3-plus: recover the modern-Java files that old `javalang` cannot parse, by
replaying their labelled-commit history with a tree-sitter Java parser.

These files have NO existing AST and NO deltas, so building their whole history
with a different (consistent) parser is safe and conflict-free. We reproduce the
engine's contract exactly:
  * node ids "{file}::A{i}" (pre-order over NAMED tree-sitter nodes);
  * the same subtree-hash + parent-propagation + positional diff;
  * the same edge types (HAS_AST, AST_CHILD, ADDS, REMOVES, UPDATES, MOVES) and
    node properties (alive / is_delta / pos_line,col), with tree-sitter type names
    mapped to javalang-style PascalCase so the schema stays uniform.

  python recover_treesitter.py            # DRY RUN (report per file)
  python recover_treesitter.py --apply     # write AST + deltas + update checkpoint
"""
import sys, json, hashlib
import tree_sitter_java as tsj
from tree_sitter import Language, Parser
from neo4j import GraphDatabase
import build_online_kg as bok

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")
APPLY = "--apply" in sys.argv
# Default = attach the AST only (solves the parse gap, clean +~64k nodes). The
# per-commit delta replay is opt-in: the cross-parser diff on the giant,
# heavily-refactored antlr4 files produces very large/noisy deltas (+~384k nodes,
# +~112k spurious MOVES) that would distort the graph, so it is OFF by default.
WITH_DELTAS = "--with-deltas" in sys.argv
JAVA = Language(tsj.language()); PARSER = Parser(JAVA)

# ── tree-sitter type -> javalang-style name / group ─────────────────────────
def ts_pascal(t):
    return "".join(w.capitalize() for w in t.split("_"))  # class_declaration->ClassDeclaration
def group_of(t, is_leaf):
    if "declaration" in t: return "declaration"
    if "statement" in t:   return "statement"
    if "literal" in t:     return "literal"
    if "expression" in t or "invocation" in t: return "expression"
    return "leaf" if is_leaf else "other"

# ── parse a source blob into (root_lid, nodes, children) over NAMED nodes ────
def parse_nodes(src_bytes, rel):
    root = PARSER.parse(src_bytes).root_node
    nodes, children = {}, {}; ctr = [0]
    def visit(n, parent_lid):
        lid = f"{rel}::A{ctr[0]}"; ctr[0] += 1
        named = n.named_children
        is_leaf = len(named) == 0
        nodes[lid] = dict(lid=lid, ast_type=ts_pascal(n.type),
                          group=group_of(n.type, is_leaf),
                          value=(n.text.decode("utf-8","replace")[:200] if is_leaf else ""),
                          is_leaf=is_leaf, line=n.start_point[0]+1, col=n.start_point[1]+1,
                          parent_lid=parent_lid)
        children[lid] = []
        if parent_lid is not None: children[parent_lid].append(lid)
        for ch in named: visit(ch, lid)
    visit(root, None)
    return (f"{rel}::A0" if nodes else None), nodes, children

def depths(root_lid, children):
    d = {}; stack = [(root_lid, 0)]
    while stack:
        lid, dep = stack.pop(); d[lid] = dep
        for c in children.get(lid, ()): stack.append((c, dep+1))
    return d

def attach_ast(root_lid, nodes, children):
    dep = depths(root_lid, children)
    ns = [dict(id=lid, ast_type=n["ast_type"], group=n["group"], value=n["value"],
               is_leaf=n["is_leaf"], depth=dep.get(lid,0), line=n["line"], col=n["col"])
          for lid, n in nodes.items()]
    edges = [dict(parent=p, child=c, pos=i) for p, cs in children.items() for i, c in enumerate(cs)]
    return dict(nodes=ns, edges=edges, root=root_lid)

# ── subtree-hash differ (mirrors online_ast_diff) ───────────────────────────
def _hash(lid, nodes, children, memo):
    if lid in memo: return memo[lid]
    parts = [nodes[lid]["ast_type"], str(nodes[lid]["value"])]
    for c in children[lid]: parts.append(_hash(c, nodes, children, memo))
    h = hashlib.md5("|".join(parts).encode()).hexdigest(); memo[lid] = h; return h

def diff(bn, bc, ba, an, ac, aa):
    from collections import defaultdict
    bh, ah = {}, {}; _hash(ba, bn, bc, bh); _hash(aa, an, ac, ah)
    ah_by = defaultdict(list)
    for lid in an: ah_by[ah[lid]].append(lid)
    b2a = {}; used = set()
    order = sorted(bn, key=lambda x: bn[x]["line"])
    for blid in order:
        if blid in b2a: continue
        cands = [a for a in ah_by.get(bh[blid], []) if a not in used]
        if not cands: continue
        bpos = (bn[blid]["line"], bn[blid]["col"])
        cands.sort(key=lambda a: 0 if (an[a]["line"], an[a]["col"]) == bpos else 1)
        pick = cands[0]
        def match_tree(b, a):
            b2a[b] = a; used.add(a)
            for cb, ca in zip(bc[b], ac[a]): match_tree(cb, ca)
        match_tree(blid, pick)
    changed = True
    while changed:
        changed = False
        for blid, alid in list(b2a.items()):
            bp, ap = bn[blid]["parent_lid"], an[alid]["parent_lid"]
            if bp and ap and bp not in b2a and ap not in used and bn[bp]["ast_type"] == an[ap]["ast_type"]:
                b2a[bp] = ap; used.add(ap); changed = True
    a_by_pos = defaultdict(list)
    for a in an:
        if a not in used: a_by_pos[(an[a]["ast_type"], an[a]["line"], an[a]["col"])].append(a)
    for blid in bn:
        if blid in b2a: continue
        k = (bn[blid]["ast_type"], bn[blid]["line"], bn[blid]["col"])
        if a_by_pos.get(k): pick = a_by_pos[k].pop(); b2a[blid] = pick; used.add(pick)
    matched = [[b, a, bn[b]["value"] != an[a]["value"], an[a]["value"]] for b, a in b2a.items()]
    moved = []
    for b, a in b2a.items():
        bp, ap = bn[b]["parent_lid"], an[a]["parent_lid"]
        if bp in b2a and b2a[bp] != ap: moved.append([a, an[ap]["ast_type"] if ap else ""])
    deleted = [b for b in bn if b not in b2a]
    inserted = [a for a in an if a not in used]
    return dict(matched=matched, deleted=deleted, inserted=inserted, moved=moved, an=an)

# ── apply a modify delta (mirrors build_online_kg.process_modify writes) ─────
def apply_modify(sess, commit, file_id, d, node_map):
    an = d["an"]; new_map = {}
    gid = lambda blid: node_map.get(blid, blid)
    restamp, updates = [], []
    for b, a, vc, nv in d["matched"]:
        g = gid(b); new_map[a] = g
        restamp.append({"id": g, "line": an[a]["line"], "col": an[a]["col"]})
        if vc: updates.append({"id": g, "new": nv})
    removes = [gid(b) for b in d["deleted"]]
    inserts = []
    for alid in d["inserted"]:
        n = an[alid]; gnew = f"{file_id}::D{commit[:8]}:{alid.rsplit('::A',1)[-1]}"
        new_map[alid] = gnew
        inserts.append(dict(id=gnew, atype=n["ast_type"], group=n["group"], val=n["value"],
                            is_leaf=n["is_leaf"], line=n["line"], col=n["col"],
                            depth=0, parent_lid=n["parent_lid"]))
    for row in inserts: row["parent"] = new_map.get(row["parent_lid"])
    moves = [{"id": new_map[a], "npt": npt} for a, npt in d["moved"] if a in new_map]
    if removes:
        sess.run("""MATCH (c:Commit {id:$cid}) UNWIND $ids AS gid MATCH (a:ASTNode {id:gid})
            SET a.alive=false, a.removed_by=$cid MERGE (c)-[:REMOVES]->(a)""", cid=commit, ids=removes)
    if updates:
        sess.run("""MATCH (c:Commit {id:$cid}) UNWIND $rows AS u MATCH (a:ASTNode {id:u.id})
            MERGE (c)-[r:UPDATES]->(a) ON CREATE SET r.old_value=a.value, r.new_value=u.new
            SET a.value=u.new""", cid=commit, rows=updates)
    if inserts:
        sess.run("""MATCH (c:Commit {id:$cid}) UNWIND $rows AS n
            MERGE (a:ASTNode {id:n.id}) ON CREATE SET a.file=$file, a.ast_type=n.atype,
              a.group=n.group, a.value=n.val, a.is_leaf=n.is_leaf, a.depth=n.depth,
              a.pos_line=n.line, a.pos_col=n.col, a.is_delta=true, a.alive=true
            MERGE (c)-[:ADDS]->(a)""", cid=commit, file=file_id, rows=inserts)
        sess.run("""UNWIND $rows AS n WITH n WHERE n.parent IS NOT NULL
            MATCH (p:ASTNode {id:n.parent}),(ch:ASTNode {id:n.id}) MERGE (p)-[:AST_CHILD]->(ch)""", rows=inserts)
    if moves:
        sess.run("""MATCH (c:Commit {id:$cid}) UNWIND $rows AS m MATCH (a:ASTNode {id:m.id})
            MERGE (c)-[r:MOVES]->(a) ON CREATE SET r.new_parent_type=m.npt""", cid=commit, rows=moves)
    if restamp:
        sess.run("""UNWIND $rows AS r MATCH (a:ASTNode {id:r.id}) SET a.pos_line=r.line, a.pos_col=r.col""",
                 rows=restamp)
    return new_map, dict(adds=len(inserts), removes=len(removes), updates=len(updates), moves=len(moves))

def main():
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with d.session(default_access_mode="READ") as s:
        rows = s.execute_read(lambda tx: [r.data() for r in tx.run("""
            MATCH (c:Commit {in_jit:true})-[:MODIFIED|ADDED]->(f:File)
            WHERE f.id ENDS WITH '.java' AND NOT (f)-[:HAS_AST]->()
            WITH f, c ORDER BY c.author_ts
            RETURN f.id AS f, collect(c.id) AS commits ORDER BY f.id""")])
    ck = json.loads(bok.CKPT_PATH.read_text())
    file_state, file_maps = ck["file_state"], ck.get("file_maps", {})

    sess = d.session() if APPLY else None
    grand = dict(files=0, commits=0, adds=0, removes=0, updates=0, moves=0)
    for r in rows:
        f, commits = r["f"], r["commits"]
        # first commit where the blob is non-empty
        first = None
        for c in commits:
            if bok.git_bytes(c, f): first = c; break
        if first is None:
            print(f"  SKIP {f.split('/')[-1]} (no retrievable blob)"); continue
        root, nodes, children = parse_nodes(bok.git_bytes(first, f), f)
        if not nodes:
            print(f"  SKIP {f.split('/')[-1]} (empty parse)"); continue
        if APPLY:
            bok.attach_full_ast(sess, f, attach_ast(root, nodes, children), first, as_new_file=True)
        node_map = {lid: lid for lid in nodes}              # identity at bootstrap
        file_state[f] = first; ndelta = dict(adds=0, removes=0, updates=0, moves=0)
        # replay the rest (opt-in; see WITH_DELTAS note)
        for c in (commits if WITH_DELTAS else []):
            if c == first: continue
            before = bok.git_bytes(file_state[f], f); after = bok.git_bytes(c, f)
            if before is None or after is None: continue
            br, bn, bc = parse_nodes(before, f); ar, an_, ac = parse_nodes(after, f)
            if not bn or not an_: continue
            dd = diff(bn, bc, br, an_, ac, ar)
            if APPLY:
                node_map, st = apply_modify(sess, c, f, dd, node_map)
                file_state[f] = c; file_maps[f] = node_map
                for k in ndelta: ndelta[k] += st[k]
            else:
                st = dict(adds=len(dd["inserted"]), removes=len(dd["deleted"]),
                          updates=sum(1 for m in dd["matched"] if m[2]), moves=len(dd["moved"]))
                for k in ndelta: ndelta[k] += st[k]
        grand["files"] += 1; grand["commits"] += len(commits)
        for k in ("adds","removes","updates","moves"): grand[k] += ndelta[k]
        print(f"  {'OK' if APPLY else 'DRY'} {f.split('/')[-1]:42s} {len(nodes):>5} nodes, "
              f"{len(commits)} commits  deltas adds={ndelta['adds']} rem={ndelta['removes']} "
              f"upd={ndelta['updates']} mov={ndelta['moves']}")
    if APPLY:
        ck["file_state"], ck["file_maps"] = file_state, file_maps
        bok.CKPT_PATH.write_text(json.dumps(ck)); sess.close()
    d.close()
    print(f"\n{'APPLIED' if APPLY else 'DRY RUN'}: {grand['files']} files, "
          f"deltas adds={grand['adds']} removes={grand['removes']} updates={grand['updates']} moves={grand['moves']}")
    if not APPLY: print("Re-run with --apply to write to Neo4j + checkpoint.")

if __name__ == "__main__":
    main()
