"""
Shared uniform differ for the alternative subgraphs (CFG / DFG / PDG / SEQ).

The AST layer diffs TREES with bottom-up subtree hashing (online_ast_diff.py).
CFG/DFG/PDG are general typed graphs, so we use a node-identity-lite matcher that
is applied IDENTICALLY to every subgraph -- this is the fairness guarantee: the
only thing that varies across the RQ variants is the subgraph itself, never the
difference model.

Matching (before -> after), all keyed WITHIN a method (class::name, stable
across commits) so unrelated methods never cross-match. Identity is by ORDINAL,
not line: the k-th node of a given type in a method maps to the k-th such node in
the other version. This is robust to line shifts (a top-of-file insert moves
every line but not the ordinals), so a one-line edit yields a handful of
adds/removes instead of cascading false matches. Line is used only for restamping.
  1. exact  (method, type, ordinal)   -- ordinal = rank among same-type nodes
                                          in the method, ordered by (line, id)
  2. greedy (method, type)            -- leftovers when the same-type count changed

Output contract (consumed by build_subgraph_online_kg.py) mirrors
online_ast_diff.diff_sources so the engine code is uniform:
  {
    "after_nodes": [{lid,type,group,method,line,col,value,is_leaf,parent_lid}],
    "after_edges": [{src,dst,etype}],
    "matched":  [[before_lid, after_lid, value_changed, new_value]],
    "deleted":  [before_lid, ...],
    "inserted": [after_lid, ...],
    "moved":    [[after_lid, new_parent_type], ...],
  }

diff_multiset() is the documented lightweight fallback (decision #2): pure typed
count difference, no node identity / edges / moves -- used only if a subgraph's
identity matching proves unstable, and the choice is recorded per kind.
"""
from collections import defaultdict


def _primary_pred(edges):
    """lid -> its primary predecessor lid (source of the first in-edge)."""
    pred = {}
    for e in edges:
        pred.setdefault(e["dst"], e["src"])
    return pred


def _index(nodes):
    return {n["id"]: n for n in nodes}


def _ordinals(nodes):
    """lid -> rank among same-(method,type) nodes, ordered by (line, id)."""
    groups = defaultdict(list)
    for n in nodes:
        groups[(n["method"], n["type"])].append(n)
    ordinal = {}
    for ns in groups.values():
        for rank, n in enumerate(sorted(ns, key=lambda x: (x["line"], x["id"]))):
            ordinal[n["id"]] = rank
    return ordinal


def diff_graphs(before, after):
    bnodes, anodes = before["nodes"], after["nodes"]
    bidx, aidx = _index(bnodes), _index(anodes)
    bpred, apred = _primary_pred(before["edges"]), _primary_pred(after["edges"])
    bord, aord = _ordinals(bnodes), _ordinals(anodes)

    b2a = {}          # before_lid -> after_lid
    a_used = set()

    # ---- pass 1: exact (method, type, ordinal) -- stable under line shifts ----
    a_by_key = {}
    for n in anodes:
        a_by_key[(n["method"], n["type"], aord[n["id"]])] = n["id"]
    for n in bnodes:
        a = a_by_key.get((n["method"], n["type"], bord[n["id"]]))
        if a is not None and a not in a_used:
            b2a[n["id"]] = a; a_used.add(a)

    # ---- pass 2: greedy (method, type) for leftovers (count changed) ----
    a_by_mt = defaultdict(list)
    for n in anodes:
        if n["id"] not in a_used:
            a_by_mt[(n["method"], n["type"])].append(n["id"])
    for k in a_by_mt:                                 # deterministic order
        a_by_mt[k].sort(key=lambda x: aord[x], reverse=True)
    for n in sorted(bnodes, key=lambda x: bord[x["id"]]):
        if n["id"] in b2a:
            continue
        cands = a_by_mt.get((n["method"], n["type"]))
        while cands:
            a = cands.pop()
            if a not in a_used:
                b2a[n["id"]] = a; a_used.add(a); break

    # ---- assemble ----
    matched, moved = [], []
    for blid, alid in b2a.items():
        bv, av = bidx[blid]["value"], aidx[alid]["value"]
        matched.append([blid, alid, bv != av, av])
        # move = matched node reparented onto a predecessor of a DIFFERENT type
        # (mirrors the AST layer's "new_parent_type"; requiring a type change
        # keeps mere sibling reordering from being counted as a move).
        bp, ap = bpred.get(blid), apred.get(alid)
        if bp in b2a and b2a[bp] != ap:
            bpt = bidx[bp]["type"] if bp in bidx else ""
            apt = aidx[ap]["type"] if ap in aidx else ""
            if bpt != apt:
                moved.append([alid, apt])

    deleted = [n["id"] for n in bnodes if n["id"] not in b2a]
    inserted = [n["id"] for n in anodes if n["id"] not in a_used]

    after_nodes = [dict(n, parent_lid=apred.get(n["id"])) for n in anodes]
    return {
        "after_nodes": after_nodes,
        "after_edges": after["edges"],
        "matched": matched,
        "deleted": deleted,
        "inserted": inserted,
        "moved": moved,
    }


def diff_multiset(before, after):
    """Fallback: typed count difference only (no identity, edges, or moves)."""
    from collections import Counter
    bc = Counter((n["method"], n["type"]) for n in before["nodes"])
    ac = Counter((n["method"], n["type"]) for n in after["nodes"])
    # keep the after graph whole (as inserts) so tokens = ADDS/REMOVES by type;
    # engine treats matched=[] -> everything alive is re-created / removed.
    return {
        "after_nodes": [dict(n, parent_lid=None) for n in after["nodes"]],
        "after_edges": after["edges"],
        "matched": [],
        "deleted": [n["id"] for n in before["nodes"]],
        "inserted": [n["id"] for n in after["nodes"]],
        "moved": [],
        "_multiset_delta": {f"{k[1]}": ac[k] - bc[k] for k in (set(ac) | set(bc))},
    }
