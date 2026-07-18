"""
Registry of the alternative structural subgraphs evaluated in the v4
"Which subgraph is best?" research question.

Each SubgraphSpec describes ONE structural layer uniformly so a single online
growth engine (build_subgraph_online_kg.py), a single differ (subgraph_diff.py)
and a single inference loader (inference/advanced_infer.load_kg) can serve all of
them. The AST layer (build_online_kg.py / :ASTNode) is the incumbent V3 and is
NOT described here -- it is left completely untouched and reused as-is.

All subgraphs share the SAME delta-edge vocabulary as AST -- ADDS / REMOVES /
UPDATES / MOVES from :Commit -- typed only by the node LABEL they point at, so
the downstream token code ("{edge}:{type}") is identical across layers.

kind        cli name / checkpoint suffix
node_label  Neo4j label for this layer's nodes (kept distinct from :ASTNode)
attach_rel  File-[:attach_rel]->root  (per-method root(s))
type_prop   node property used as the token "type" (the thing swapped for ast_type)
group_prop  coarse group property (for stats/figures)
prefix      deterministic local-id prefix, mirrors AST's "::A{i}"
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SubgraphSpec:
    kind: str
    node_label: str
    attach_rel: str
    type_prop: str
    group_prop: str
    prefix: str


REGISTRY = {
    "cfg": SubgraphSpec("cfg", "CFGNode", "HAS_CFG", "kind", "group", "C"),
    "dfg": SubgraphSpec("dfg", "DFGNode", "HAS_DFG", "kind", "group", "F"),
    "pdg": SubgraphSpec("pdg", "PDGNode", "HAS_PDG", "kind", "group", "P"),
    "seq": SubgraphSpec("seq", "SEQNode", "HAS_SEQ", "type", "group", "S"),
}

DELTA_RELS = ("ADDS", "REMOVES", "UPDATES", "MOVES")


def get(kind: str) -> SubgraphSpec:
    if kind not in REGISTRY:
        raise SystemExit(f"unknown subgraph kind {kind!r}; choose from {list(REGISTRY)}")
    return REGISTRY[kind]
