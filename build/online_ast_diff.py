"""
javalang-native AST differ for the online KG engine.

GumTree parses a DIFFERENT tree (JavaParser/JDT) than our javalang-built graph,
so matching GumTree actions back to graph nodes by position is ambiguous and
drifts.  This differ parses BOTH versions with javalang and builds them with the
SAME _build_ast_subgraph scheme the graph uses, so node identity is exact:

  - both parses use deterministic ids "{rel}::A{i}" (same source => same ids)
  - we match before<->after nodes structurally (subtree hash bottom-up, then
    parent propagation, then positional fallback)
  - output drives evolve: matched keep their graph id (reposition / value
    update), deleted are removed, inserted are created under a matched parent

Because the after-parse of commit N == the before-parse of commit N+1 (identical
source => identical "::A{i}" ids), the per-file map {parse_id -> graph_id} can be
threaded across commits with no positional ambiguity.

Public API:
  diff_sources(before_bytes, after_bytes, rel_path) -> dict | None
    {
      "root": "<rel>::A0",
      "after_nodes": [{lid, ast_type, group, value, is_leaf, line, col,
                       parent_lid, depth}],   # the full AFTER tree
      "matched":  [[before_lid, after_lid, value_changed, new_value]],
      "deleted":  [before_lid, ...],
      "inserted": [after_lid, ...],           # subset of after_nodes (new)
      "moved":    [[after_lid, new_parent_type], ...]
    }
"""

import subprocess, sys, json, tempfile, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_WORKER = r"""
import sys, json, hashlib
import javalang, javalang.tree as jlt
import networkx as nx
import build_ast_features as bf

def clean(v):
    # strip lone UTF-16 surrogates that Neo4j's driver cannot encode to UTF-8
    if isinstance(v, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in v):
        return ''.join(c for c in v if not (0xD800 <= ord(c) <= 0xDFFF))
    return v

def build(path, rel):
    src = open(path, encoding='utf-8', errors='replace').read()
    tree = javalang.parse.parse(src)
    Ag, root = bf._build_ast_subgraph(tree, rel, jlt, nx)
    depths = nx.shortest_path_length(Ag, source=root)

    # positions: mirror the EXACT pre-order DFS of _build_ast_subgraph
    positions = []
    def walk(node):
        pos = getattr(node, 'position', None)
        positions.append([int(pos.line), int(pos.column)] if pos else [-1, -1])
        for child in node.children:
            if child is None: continue
            items = child if isinstance(child, (list, tuple, set, frozenset)) else [child]
            for it in items:
                if isinstance(it, jlt.Node): walk(it)
                elif isinstance(it, str) and it.strip(): positions.append([-1, -1])
    walk(tree)

    parent = {}
    childpos = {}
    for u, v, d in Ag.edges(data=True):
        parent[v] = u
        childpos[v] = d.get('pos', 0)

    nodes = {}
    for nid, dat in Ag.nodes(data=True):
        i = int(nid.rsplit('::A', 1)[1])
        line, col = (positions[i] if i < len(positions) else [-1, -1])
        nodes[nid] = {
            'lid': nid, 'ast_type': dat.get('ast_type'), 'group': dat.get('group'),
            'value': clean(dat.get('value')), 'is_leaf': bool(dat.get('is_leaf')),
            'line': line, 'col': col, 'depth': int(depths.get(nid, 0)),
            'parent_lid': parent.get(nid), 'childpos': childpos.get(nid, 0),
        }
    # children lists (ordered by childpos)
    children = {nid: [] for nid in nodes}
    for nid, p in parent.items():
        children[p].append(nid)
    for p in children:
        children[p].sort(key=lambda c: nodes[c]['childpos'])
    return root, nodes, children

def subtree_hash(root, nodes, children, memo):
    if root in memo: return memo[root]
    n = nodes[root]
    parts = [str(n['ast_type']), str(n['value'])]
    for c in children[root]:
        parts.append(subtree_hash(c, nodes, children, memo))
    h = hashlib.md5('|'.join(parts).encode()).hexdigest()
    memo[root] = h
    return h

bp, bn, bc = build(sys.argv[1], sys.argv[3])
ap, an, ac = build(sys.argv[2], sys.argv[3])

bh, ah = {}, {}
subtree_hash(bp, bn, bc, bh)
subtree_hash(ap, an, ac, ah)

# group after-nodes by hash for greedy identical-subtree matching
from collections import defaultdict
ah_by = defaultdict(list)
for nid in an: ah_by[ah[nid]].append(nid)

b2a = {}   # before_lid -> after_lid
a_used = set()

# 1) bottom-up: identical subtrees (prefer same position), largest first
border = sorted(bn, key=lambda x: -bn[x]['depth'])   # deep (small) first is wrong; want big subtrees -> low depth first
border = sorted(bn, key=lambda x: bn[x]['depth'])     # shallow (big subtrees) first
for blid in border:
    if blid in b2a: continue
    cands = [a for a in ah_by.get(bh[blid], []) if a not in a_used]
    if not cands: continue
    bpos = (bn[blid]['line'], bn[blid]['col'])
    cands.sort(key=lambda a: 0 if (an[a]['line'], an[a]['col']) == bpos else 1)
    pick = cands[0]
    # match whole subtree pairwise (same shape since identical hash)
    def match_tree(b, a):
        b2a[b] = a; a_used.add(a)
        for cb, ca in zip(bc[b], ac[a]):
            match_tree(cb, ca)
    match_tree(blid, pick)

# 2) top-down propagation: if parents of a matched pair are unmatched and share
#    type, match them; repeat to cover changed-but-aligned ancestors
changed = True
while changed:
    changed = False
    for blid, alid in list(b2a.items()):
        bp_, ap_ = bn[blid]['parent_lid'], an[alid]['parent_lid']
        if bp_ and ap_ and bp_ not in b2a and ap_ not in a_used:
            if bn[bp_]['ast_type'] == an[ap_]['ast_type']:
                b2a[bp_] = ap_; a_used.add(ap_); changed = True

# 3) positional fallback for still-unmatched (same type+line+col)
a_by_pos = defaultdict(list)
for a in an:
    if a not in a_used:
        a_by_pos[(an[a]['ast_type'], an[a]['line'], an[a]['col'])].append(a)
for blid in bn:
    if blid in b2a: continue
    key = (bn[blid]['ast_type'], bn[blid]['line'], bn[blid]['col'])
    if bn[blid]['line'] > 0 and a_by_pos.get(key):
        pick = a_by_pos[key].pop()
        b2a[blid] = pick; a_used.add(pick)

# assemble result
matched, moved = [], []
for blid, alid in b2a.items():
    vc = (bn[blid]['value'] != an[alid]['value'])
    matched.append([blid, alid, vc, an[alid]['value']])
    # move: matched parents differ
    bp_, ap_ = bn[blid]['parent_lid'], an[alid]['parent_lid']
    if bp_ in b2a and b2a[bp_] != ap_:
        moved.append([alid, an[ap_]['ast_type'] if ap_ else ''])

deleted  = [b for b in bn if b not in b2a]
inserted = [a for a in an if a not in a_used]

print(json.dumps({
    'root': ap,
    'after_nodes': list(an.values()),
    'matched': matched,
    'deleted': deleted,
    'inserted': inserted,
    'moved': moved,
}))
"""


def diff_sources(before_bytes, after_bytes, rel_path, timeout=90):
    bt = before_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
    at = after_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
    fb = tempfile.NamedTemporaryFile(suffix=".java", delete=False); fb.write(bt.encode()); fb.close()
    fa = tempfile.NamedTemporaryFile(suffix=".java", delete=False); fa.write(at.encode()); fa.close()
    try:
        r = subprocess.run([sys.executable, "-c", _WORKER, fb.name, fa.name, rel_path],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(PROJECT_ROOT))
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    finally:
        os.unlink(fb.name); os.unlink(fa.name)
