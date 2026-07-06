"""
Shared helpers for the V4 scalability / complexity / statistical analysis suite.

Scope: everything here measures ONLY the final (V4) methodology as documented in
docs/KG-Commit evolution.tex (read end-first; V4 supersedes V1-V3):

  * Graph family (6):  Core, Core+AST, Core+CFG, Core+DFG, Core+PDG,
                       and the deployed final Core+AST+CSTG.
                       (Token-seq is NOT in the final family.)
  * Five graph-inference methods: RN, PPR, LP, DW, KGE  (inference/kg_methods.py).
  * Deployed fusion: F = RN+PPR ; deployed model F+G = RN+PPR+CSTG (graph-only).
  * Dropped and therefore never measured as "final": the commit-text channel X,
    the JIT-metrics fusion channel M, the V3-era M+T+R+P ablation, standalone
    embeddings (node2vec/TransE), and the legacy ASTDiff/ASTEdit GumTree layer.

All DB access is READ-ONLY. The expensive hub dictionaries are pulled from Neo4j
once and cached to outputs/scalability/final_graph_cache.pkl, after which the
whole graph family is rebuilt purely in-memory (no live DB) for the timing and
significance experiments -- so nothing here rebuilds the KG.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
INFERENCE = ROOT / "inference"
OUT = ROOT / "outputs" / "scalability"
FIG = ROOT / "docs" / "figures" / "v4" / "scalability"
OUT.mkdir(parents=True, exist_ok=True)

# make the existing final-methodology code importable:
#  - inference/ holds the inference stack (kg_methods, run_final_experiments, ...)
#  - the repo root holds the online build engines and differs (online_ast_diff,
#    subgraph_builders, subgraph_diff, build_online_kg, ...)
for _p in (INFERENCE, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password1234")

# The final graph family (columns of the V4 study). Token-seq intentionally absent.
FINAL_GRAPHS = ["core", "ast", "cfg", "dfg", "pdg", "final"]
GRAPH_NAME = {"core": "Core", "ast": "Core+AST", "cfg": "Core+CFG",
              "dfg": "Core+DFG", "pdg": "Core+PDG", "final": "Core+AST+CSTG"}
FINAL_METHODS = ["RN", "PPR", "LP", "DW", "KGE"]

# Structural layers that were materialised in the KG (for the size/profile table).
# PDG is the statement-level program-dependence graph (not a canonical CPG).
STRUCT_LAYERS = [  # (id, pretty, node label, type property)
    ("ast", "AST",       "ASTNode", "ast_type"),
    ("cfg", "CFG",       "CFGNode", "atype"),
    ("dfg", "DFG",       "DFGNode", "atype"),
    ("pdg", "PDG",       "PDGNode", "atype"),
    ("seq", "Token-seq", "SEQNode", "atype"),
]
DELTA_RELS = ["ADDS", "REMOVES", "UPDATES", "MOVES"]


# ---- Neo4j (read-only) -----------------------------------------------------

def driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def read_session(d):
    """A strictly read-only session (guards against accidental writes)."""
    return d.session(default_access_mode="READ")


def assert_db_complete():
    """Confirm the online build finished (next_index==8059 in every checkpoint)
    so the profiled graph is the final state, not a partial build."""
    ck = ROOT / "outputs"
    files = {"ast": ck / "online_kg_checkpoint.json"}
    for k in ("cfg", "dfg", "pdg", "seq"):
        files[k] = ck / f"online_kg_checkpoint_{k}.json"
    status = {}
    for k, p in files.items():
        status[k] = json.load(open(p)).get("next_index") if p.exists() else None
    ok = all(v == 8059 for v in status.values())
    return ok, status


# ---- cached final-graph loader (Neo4j once -> in-memory forever) ------------

def load_final_graph_cache(force=False):
    """Load the hub dictionaries for the whole final graph family and cache them.

    Reuses inference/run_final_experiments.load_all() (read-only Cypher). After the
    first call the cache is a plain pickle and NO database is touched again, which is
    what lets the timing (E4) and significance (E5) experiments replay the final
    methodology without rebuilding or re-querying the KG.

    Returns a dict: commits, cids, y, files, devs, tok (per structural layer),
    cstg (Commit->[Term] mentions).
    """
    cache = OUT / "final_graph_cache.pkl"
    if cache.exists() and not force:
        return pickle.load(open(cache, "rb"))
    import run_final_experiments as rfe
    commits, cids, y, files, devs, tok, cstg = rfe.load_all()
    bundle = dict(commits=commits, cids=list(cids), y=np.asarray(y),
                  files=files, devs=devs, tok=tok, cstg=cstg)
    pickle.dump(bundle, open(cache, "wb"))
    return bundle


# ---- small numeric utilities (no statsmodels / powerlaw dependency) ---------

def describe(x):
    """Robust distribution summary used across the size/growth tables."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return dict(n=0, mean=0.0, std=0.0, min=0.0, p50=0.0, p95=0.0,
                    max=0.0, sum=0.0, gini=0.0, skew=0.0)
    return dict(
        n=int(x.size), mean=float(x.mean()), std=float(x.std()),
        min=float(x.min()), p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)), max=float(x.max()),
        sum=float(x.sum()), gini=float(_gini(x)), skew=float(_skew(x)),
    )


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _skew(x):
    x = np.asarray(x, float)
    m = x.mean(); s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / s ** 3)


def powerlaw_tail(deg, xmin=None):
    """Discrete power-law tail exponent via Clauset-style MLE + a KS statistic
    against the fitted law -- pure numpy, no external `powerlaw` package.

    Returns dict(alpha, xmin, n_tail, ks). alpha is the ML exponent for the
    heavy tail (degrees >= xmin); ks is the Kolmogorov-Smirnov distance between
    the empirical tail CDF and the fitted power-law CDF (smaller = better fit).
    """
    deg = np.asarray(deg, float)
    deg = deg[deg > 0]
    if deg.size < 20:
        return dict(alpha=float("nan"), xmin=float("nan"), n_tail=int(deg.size),
                    ks=float("nan"))
    if xmin is None:
        # pick xmin minimising the KS distance over a small candidate set
        cands = np.unique(np.percentile(deg, np.linspace(50, 95, 10)).astype(int))
        cands = cands[cands >= 1]
        best = None
        for xm in cands:
            r = _fit_pl(deg, xm)
            if r is not None and (best is None or r["ks"] < best["ks"]):
                best = r
        return best if best is not None else _fit_pl(deg, int(np.median(deg)))
    return _fit_pl(deg, xmin)


def _fit_pl(deg, xmin):
    tail = deg[deg >= xmin]
    n = tail.size
    if n < 10:
        return None
    # continuous MLE (good approximation for the discrete case at these scales)
    alpha = 1.0 + n / np.sum(np.log(tail / (xmin - 0.5)))
    xs = np.sort(tail)
    emp = np.arange(1, n + 1) / n
    fit = 1.0 - (xs / xmin) ** (1.0 - alpha)
    ks = float(np.max(np.abs(emp - fit)))
    return dict(alpha=float(alpha), xmin=float(xmin), n_tail=int(n), ks=ks)


def save_json(obj, name):
    p = OUT / name
    json.dump(obj, open(p, "w"), indent=1, default=_json_default)
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")
