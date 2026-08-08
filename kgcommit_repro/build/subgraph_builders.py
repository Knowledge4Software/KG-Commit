"""
Per-file subgraph builders for the v4 "which subgraph is best?" experiment.

Reuses the graph-construction cores already written for the offline static
feature extractors -- CFGBuilder (build_cfg_features), DefUseBuilder
(build_dfg_features), CPGBuilder (build_cpg_features) -- and adds a deliberately
weak SeqBuilder (flat statement sequence). Every builder is wrapped so it emits
ONE uniform JSON shape, identical in spirit to the AST worker in
build_online_kg.py:

  {
    "nodes": [{id, type, group, method, line, col, value, is_leaf}],
    "edges": [{src, dst, etype}],
    "roots": [lid, ...]          # per-method attach points (File-[:HAS_X]->root)
  }

Local node ids ("lid") are deterministic per source -- "{rel}::{PREFIX}{i}" over
a file-global counter that advances in (method-order, builder-ctr-order). So the
after-parse of commit N has byte-identical lids to the before-parse of N+1 when
the source is unchanged, letting the differ thread node identity across commits
(same trick the AST layer uses).

Run standalone (used as a killable subprocess by the online engine):
  python subgraph_builders.py <kind> <src.java> <rel/path.java>
    -> prints the JSON above, or {"error": "..."} on parse failure.
"""
import sys, json
import networkx as nx
import javalang, javalang.tree as jlt

import subgraph_spec as spec
from build_cfg_features import CFGBuilder
from build_dfg_features import DefUseBuilder
from build_cpg_features import CPGBuilder


def _clean(v):
    if isinstance(v, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in v):
        return "".join(c for c in v if not (0xD800 <= ord(c) <= 0xDFFF))
    return v


def _enclosing_class(path_nodes, jlt):
    cls = "<unknown>"
    for n in path_nodes:
        if isinstance(n, (jlt.ClassDeclaration, jlt.InterfaceDeclaration,
                          jlt.EnumDeclaration)) and getattr(n, "name", None):
            cls = n.name
    return cls


def _local_sort_key(nid):
    """Order a builder's node ids by their trailing integer (ctr order)."""
    for i, ch in enumerate(nid):
        if ch.isdigit():
            return (nid[:i], int(nid[i:]))
    return (nid, 0)


# ── per-method graph producers → (nx graph, root_local_id, node_type_fn) ─────

def _cfg_method(node):
    b = CFGBuilder(nx, jlt)
    G, entry = b.build(node)
    return G, entry, (lambda nid, d: d.get("kind", "?"))


def _pdg_method(node):
    b = CPGBuilder(nx, jlt)
    G, entry = b.build(node)
    return G, entry, (lambda nid, d: d.get("kind", "?"))


def _dfg_method(node):
    b = DefUseBuilder(nx, jlt)
    G = b.build(node)
    root = min(G.nodes(), key=_local_sort_key) if G.number_of_nodes() else None
    # richer, DFG-native token type: def_type for DEFs, 'USE' for uses
    def tfn(nid, d):
        return d.get("def_type") or ("USE" if d.get("kind") == "USE" else "DEF")
    return G, root, tfn


def _seq_method(node):
    """Deliberately weak: flat pre-order sequence of statement nodes typed by
    their javalang class, chained by NEXT edges. No control/data structure."""
    G = nx.DiGraph(); ctr = [0]; prev = [None]; root = [None]

    def visit(st):
        if not isinstance(st, jlt.Statement):
            return
        nid = f"N{ctr[0]}"; ctr[0] += 1
        p = getattr(st, "position", None)
        G.add_node(nid, kind=type(st).__name__, group="stmt",
                   line=(p.line if p else -1))
        if root[0] is None:
            root[0] = nid
        if prev[0] is not None:
            G.add_edge(prev[0], nid, rel="NEXT")
        prev[0] = nid
        for child in st.children:                    # DIRECT children only
            if child is None: continue
            items = child if isinstance(child, (list, set, frozenset, tuple)) else [child]
            for it in items:
                if isinstance(it, jlt.Statement):
                    visit(it)
    body = node.body if isinstance(node.body, (list, tuple)) else [node.body]
    for st in body:
        if st is not None:
            visit(st)
    return G, root[0], (lambda nid, d: d.get("kind", "?"))


# ── per-method AST builder (scope-matched to CFG/DFG/PDG) ───────────────
# Uses the SAME method-discovery gate and SAME json shape. The ONLY
# difference vs CFG/DFG/PDG is the representation: a full syntax tree
# instead of a control/data/dependence graph. This is the point of the
# comparison — same scope, different representation.

# AST node grouping (imported from the existing build_ast_features module
# to reuse the proven classification; lazy-loaded to avoid import-time cost
# when this module is used as a worker for other kinds).
_ast_group_fn = None

def _get_ast_group(name):
    global _ast_group_fn
    if _ast_group_fn is None:
        import build_ast_features as bf
        _ast_group_fn = bf.ast_group
    return _ast_group_fn(name)


def _ast_method_method(node):
    """Per-method AST: full syntax tree of the method body, using javalang
    node-type names as ``kind`` (matching CFG/DFG/PDG's property name).
    Scope-matched to the other builders: only method-body constructs.
    Edges use AST_CHILD (the natural tree parent→child relationship)."""
    G = nx.DiGraph(); ctr = [0]
    root_nid = None
    stack = [(node, None, 0)]
    while stack:
        jval, parent_nid, child_pos = stack.pop()
        nid = f"N{ctr[0]}"; ctr[0] += 1
        if parent_nid is None:
            root_nid = nid
        if isinstance(jval, jlt.Node):
            t = type(jval).__name__
            vs = ''
            if getattr(jval, 'name', None):                 vs = str(jval.name)[:24]
            elif getattr(jval, 'value', None) is not None:  vs = str(jval.value)[:24]
            elif getattr(jval, 'operator', None):           vs = str(jval.operator)[:24]
            p = getattr(jval, 'position', None)
            G.add_node(nid, kind=t, group=_get_ast_group(t),
                       line=(p.line if p else -1), var=vs)
            if parent_nid is not None:
                G.add_edge(parent_nid, nid, rel='AST_CHILD')
            children = []; cp = 0
            for child in jval.children:
                if child is None:
                    continue
                if isinstance(child, jlt.Node):
                    children.append((child, nid, cp)); cp += 1
                elif isinstance(child, (list, frozenset, set)):
                    for it in child:
                        if isinstance(it, jlt.Node):
                            children.append((it, nid, cp)); cp += 1
                        elif isinstance(it, str) and it.strip():
                            children.append((it, nid, cp)); cp += 1
                elif isinstance(child, str) and child.strip():
                    children.append((child, nid, cp)); cp += 1
            stack.extend(reversed(children))
        else:
            val = jval[:24] if isinstance(jval, str) else str(jval)[:24]
            G.add_node(nid, kind='Identifier', group='leaf',
                       line=-1, var=val)
            if parent_nid is not None:
                G.add_edge(parent_nid, nid, rel='AST_CHILD')
    return G, root_nid, (lambda nid, d: d.get("kind", "?"))


_METHOD = {"cfg": _cfg_method, "pdg": _pdg_method, "dfg": _dfg_method,
           "seq": _seq_method, "ast_method": _ast_method_method}



def build_file_graph(kind, src_text, rel):
    """Parse a whole file, build the per-method subgraph for every concrete
    method/constructor, and merge into one file-level typed graph as uniform
    JSON. Returns dict (with 'error' key on parse failure)."""
    sp = spec.get(kind)
    method_fn = _METHOD[kind]
    try:
        tree = javalang.parse.parse(src_text)
    except Exception as e:                       # javalang parse / tokenizer error
        return {"error": f"{type(e).__name__}: {e}"}

    nodes, edges, roots = [], [], []
    gid = [0]
    for path_nodes, node in tree:                # walk whole tree to find methods
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, "body", None) is None:
            continue
        cls = _enclosing_class(path_nodes, jlt)
        method = f"{cls}::{node.name}"           # no line -> stable across commits
        try:
            G, root_local, tfn = method_fn(node)
        except Exception:
            continue                             # skip a pathological method, keep file
        if G is None or G.number_of_nodes() == 0:
            continue

        local2gid = {}
        for lid in sorted(G.nodes(), key=_local_sort_key):
            d = G.nodes[lid]
            g = f"{rel}::{sp.prefix}{gid[0]}"; gid[0] += 1
            local2gid[lid] = g
            ttype = str(tfn(lid, d))
            nodes.append({
                "id": g, "type": ttype, "group": str(d.get("group", "?")),
                "method": method, "line": int(d.get("line", -1)), "col": -1,
                "value": _clean(str(d.get("var", "") or d.get("label", ""))[:24]),
                "is_leaf": False,
            })
        for u, v, ed in G.edges(data=True):
            edges.append({"src": local2gid[u], "dst": local2gid[v],
                          "etype": str(ed.get("rel", "EDGE"))})
        if root_local is not None and root_local in local2gid:
            roots.append(local2gid[root_local])

    return {"nodes": nodes, "edges": edges, "roots": roots}


def serve():
    """Persistent worker: read one JSON request per line {kind, path, rel} from
    stdin, write one JSON response per line to stdout. Amortizes interpreter +
    import startup across thousands of files. The parent enforces a per-request
    timeout and restarts this process if a pathological file hangs it."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            src = open(req["path"], encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
            out = build_file_graph(req["kind"], src, req["rel"])
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve(); return
    kind, src_path, rel = sys.argv[1], sys.argv[2], sys.argv[3]
    src = open(src_path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    print(json.dumps(build_file_graph(kind, src, rel)))


if __name__ == "__main__":
    main()
