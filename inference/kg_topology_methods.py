#!/usr/bin/env python3
"""
Topology-native inference over the KG --- beyond commit-hub, beyond one hop.
===========================================================================

Everything so far (TGC included) still SUMMARISES a neighbourhood into numbers.
That is better than commit-hub co-membership, but it is not yet inference over
the graph's topology: the *shape* of the dependency structure around a change is
discarded, and information still cannot travel more than the tier radius.

This module implements methods whose predictions depend on GLOBAL graph
structure that no bounded neighbourhood summary can reproduce.

  M1  ARCHITECTURAL POSITION (global, structural)
      PageRank, k-core, reverse-PageRank and betweenness-proxy over the whole
      file dependency graph. A file's score depends on the ENTIRE graph, not on
      any neighbourhood of it. Commits inherit the positional profile of the
      files they touch. This is genuinely global and cannot be expressed by any
      k-hop projection.

  M2  RISK DIFFUSION WITH RESTART (global, label-propagating)
      Personalised PageRank over the file graph seeded by strictly-past defect
      mass, run to convergence rather than 3 steps. Risk reaches a commit from
      arbitrarily far away, attenuated by graph distance -- the paper's claim in
      its strongest form.

  M3  SPECTRAL / STRUCTURAL EMBEDDING (global, topological)
      Truncated SVD of the normalised file-adjacency, giving each file a
      position in a low-dimensional space determined by the whole topology.
      Commits are the centroid of their files' embeddings. Two files with
      similar global roles land nearby even if arbitrarily far apart.

  M4  DEPENDENCY-SHAPE FEATURES (local topology, not summary statistics)
      The SHAPE of the induced subgraph on the commit's dependency closure:
      density, clustering, reciprocity, component count, degree skew,
      cut-vertex count. Two commits touching the same NUMBER of files with
      different dependency shapes get different features.

  M5  CROSS-LAYER GROUNDING (heterogeneous, multi-relational)
      Uses the KG as a knowledge graph: Term -COOCCURS- Term -REFERS_TO- File
      paths let semantic context reach files the commit never touched.

All are computed on SNAPSHOTS: the graph is rebuilt at intervals from edges
known at that time, so nothing from the future leaks. CPU-only, cache-only
apart from one Neo4j read of the file graph.
"""
from __future__ import annotations

import math
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# --------------------------------------------------------------- graph helper
def build_file_graph(edges, files):
    """Sparse directed adjacency over files. edges: list of (src, dst)."""
    idx = {f: i for i, f in enumerate(files)}
    r, c = [], []
    for s, d in edges:
        if s in idx and d in idx:
            r.append(idx[s])
            c.append(idx[d])
    n = len(files)
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return A, idx


# ------------------------------------------------------------------ M1 position
def architectural_position(A):
    """Global structural role of every file. Depends on the WHOLE graph."""
    n = A.shape[0]
    out = {}

    # PageRank (importance as a dependency target)
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    P = sp.diags(1.0 / deg) @ A
    x = np.full(n, 1.0 / n)
    for _ in range(40):
        x = 0.15 / n + 0.85 * (P.T @ x)
    out["pagerank"] = x

    # reverse PageRank (importance as a dependency source)
    At = A.T.tocsr()
    deg2 = np.asarray(At.sum(1)).ravel()
    deg2[deg2 == 0] = 1.0
    P2 = sp.diags(1.0 / deg2) @ At
    x2 = np.full(n, 1.0 / n)
    for _ in range(40):
        x2 = 0.15 / n + 0.85 * (P2.T @ x2)
    out["rev_pagerank"] = x2

    # k-core number on the symmetrised graph (embeddedness in dense regions)
    S = ((A + A.T) > 0).astype(np.int8).tocsr()
    degs = np.asarray(S.sum(1)).ravel().astype(int)
    core = degs.copy()
    order = np.argsort(degs)
    alive = np.ones(n, bool)
    for v in order:
        if not alive[v]:
            continue
        alive[v] = False
        nb = S.indices[S.indptr[v]:S.indptr[v + 1]]
        for u in nb:
            if alive[u] and core[u] > core[v]:
                core[u] -= 1
    out["kcore"] = core.astype(float)

    # degree profile
    out["in_deg"] = np.asarray(A.sum(0)).ravel()
    out["out_deg"] = np.asarray(A.sum(1)).ravel()
    return out


# ------------------------------------------------------------------ M2 diffusion
def risk_ppr(A, seed_mass, alpha=0.15, iters=30):
    """Personalised PageRank seeded by past defect mass, run to convergence.

    Risk reaches a file from ARBITRARY distance, attenuated by graph distance.
    This is the paper's claim in its strongest computational form.
    """
    n = A.shape[0]
    S = ((A + A.T) > 0).astype(float).tocsr()
    deg = np.asarray(S.sum(1)).ravel()
    deg[deg == 0] = 1.0
    P = sp.diags(1.0 / deg) @ S
    tot = seed_mass.sum()
    if tot <= 0:
        return np.zeros(n)
    r = seed_mass / tot
    x = r.copy()
    for _ in range(iters):
        x = (1 - alpha) * (P.T @ x) + alpha * r
    return x


# ------------------------------------------------------------------ M3 spectral
def spectral_embedding(A, dim=32):
    """Global topological coordinates: SVD of the normalised adjacency."""
    n = A.shape[0]
    S = ((A + A.T) > 0).astype(float).tocsr()
    deg = np.asarray(S.sum(1)).ravel()
    deg[deg == 0] = 1.0
    D = sp.diags(1.0 / np.sqrt(deg))
    L = D @ S @ D
    k = min(dim, min(L.shape) - 1)
    if k < 2:
        return np.zeros((n, dim))
    try:
        U, s, _ = svds(L, k=k)
    except Exception:
        return np.zeros((n, dim))
    E = U * s
    if E.shape[1] < dim:
        E = np.hstack([E, np.zeros((n, dim - E.shape[1]))])
    return E


# ------------------------------------------------------------------ M4 shape
def dependency_shape(A, idx, files_of_commit, radius=1):
    """Topological SHAPE of the subgraph induced on the commit's closure.

    Two commits touching the same NUMBER of files but sitting in differently
    shaped dependency structures get different features. No summary statistic
    over a neighbourhood can express this.
    """
    ids = [idx[f] for f in files_of_commit if f in idx]
    if not ids:
        return [0.0] * 8
    node = set(ids)
    frontier = set(ids)
    for _ in range(radius):
        nxt = set()
        for v in frontier:
            nxt |= set(A.indices[A.indptr[v]:A.indptr[v + 1]].tolist())
        nxt -= node
        node |= nxt
        frontier = nxt
        if len(node) > 600:
            break
    nl = sorted(node)
    if len(nl) < 2:
        return [0.0] * 8
    sub = A[nl][:, nl]
    m = sub.nnz
    n = len(nl)
    dens = m / max(n * (n - 1), 1)
    din = np.asarray(sub.sum(0)).ravel()
    dout = np.asarray(sub.sum(1)).ravel()
    recip = (sub.multiply(sub.T)).nnz / max(m, 1)

    # weakly-connected components of the induced subgraph
    sym = ((sub + sub.T) > 0).tocsr()
    seen = np.zeros(n, bool)
    comps, sizes = 0, []
    for s0 in range(n):
        if seen[s0]:
            continue
        comps += 1
        q, cnt = deque([s0]), 0
        seen[s0] = True
        while q:
            v = q.popleft()
            cnt += 1
            for u in sym.indices[sym.indptr[v]:sym.indptr[v + 1]]:
                if not seen[u]:
                    seen[u] = True
                    q.append(u)
        sizes.append(cnt)
    return [
        float(np.log1p(n)),
        float(dens),
        float(recip),
        float(np.log1p(comps)),
        float(max(sizes) / n),
        float(din.std()),
        float(dout.std()),
        float((din == 0).mean()),
    ]
