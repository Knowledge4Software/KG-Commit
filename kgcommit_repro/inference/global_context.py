#!/usr/bin/env python3
"""
Tiered Global Context (TGC) --- project context beyond the commit's neighbourhood.
=================================================================================

The commit--hub bipartite projection can only express "commit c touched hub h".
A file the commit never touched is unreachable by construction, which is exactly
the paper's central claim and exactly what the projection cannot support.

This module extracts, for each commit, nested tiers of increasing graph radius
over the PROJECT graph, and summarises each tier by history-aware statistics
computed from strictly-past information only.

  T0  files the commit modifies                       (commit-local; baseline)
  T1  files that import, or are imported by, T0       (1 hop, UNTOUCHED files)
  T2  two-hop dependency closure of T0                (2 hops)
  TP  package siblings of T0                          (structural, non-dependency)
  TC  files co-changed with T0 in the past            (historical, non-structural)

TP and TC exist so the design carries GENERAL project context rather than
degenerating into an Omega (cross-file-import) detector. Requirement: the design
must lift ordinary commits as well as Omega commits.

Every statistic uses only commits strictly earlier than the one being scored, so
the prequential/no-leakage guarantee is identical to the deployed protocol.

Stage 1 (this module) is a one-off Neo4j extraction of the STRUCTURE (which
files are in which tier). Stage 2 computes the history-aware statistics in
pure numpy at scoring time -- no Neo4j on the prediction path, preserving the
paper's real-time claim.

Out: outputs/<project>/global_context/tiers.pkl
Run: KGC_PROJECT=kafka python inference/global_context.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT, NEO4J_URI, NEO4J_AUTH  # noqa: E402

DEST = OUT / "global_context"

# a tier larger than this is treated as uninformative (mega-commits / hub files)
TIER_CAP = 400
CO_CHANGE_TOPK = 25       # strongest past co-change partners per file


# --------------------------------------------------------------- extraction
def _q(session, cypher, **kw):
    return session.execute_read(lambda tx: [r.data() for r in tx.run(cypher, **kw)])


def extract(uri=None, auth=None):
    """Pull the raw structure once. Everything downstream is numpy."""
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(uri or NEO4J_URI, auth=auth or NEO4J_AUTH)
    t0 = time.time()
    with drv.session(default_access_mode="READ") as s:
        print("  commits ...", flush=True)
        commits = _q(s, """
            MATCH (c:Commit {in_jit:true})
            RETURN c.id AS id, coalesce(c.author_ts,0) AS ts,
                   toInteger(coalesce(c.buggy,0)) AS buggy
            ORDER BY ts, id
        """)

        print("  commit -> files ...", flush=True)
        cf = _q(s, """
            MATCH (c:Commit {in_jit:true})-[:MODIFIED|ADDED|DELETED]->(f:File)
            RETURN c.id AS cid, collect(DISTINCT f.id) AS files
        """)

        print("  file import graph ...", flush=True)
        imp = _q(s, """
            MATCH (a:File)-[:IMPORTS]->(b:File)
            RETURN a.id AS src, b.id AS dst
        """)
    drv.close()
    print(f"  extraction {time.time()-t0:.1f}s", flush=True)

    touched = {r["cid"]: list(r["files"]) for r in cf}
    return commits, touched, imp


# --------------------------------------------------------------- tier building
def build_tiers(commits, touched, imp):
    """Nested tiers per commit. Structure only -- no labels are read here."""
    # directed import adjacency, both directions kept separate: the dependant
    # side (who breaks if this changes) and the dependency side (what this
    # relies on) are different signals and are tested separately.
    importers = defaultdict(set)     # f -> files that import f  (dependants)
    imports_of = defaultdict(set)    # f -> files that f imports (dependencies)
    for r in imp:
        importers[r["dst"]].add(r["src"])
        imports_of[r["src"]].add(r["dst"])

    # package = directory prefix of the path (File carries only `id`)
    def pkg(p):
        i = p.rfind("/")
        return p[:i] if i > 0 else ""

    pkg_members = defaultdict(set)
    for f in set(importers) | set(imports_of) | {x for v in touched.values() for x in v}:
        pkg_members[pkg(f)].add(f)

    # past co-change, accumulated causally as we sweep the ordered stream
    co = defaultdict(lambda: defaultdict(int))

    tiers = {}
    order = [c["id"] for c in commits]
    for cid in order:
        T0 = set(touched.get(cid, ()))
        if not T0:
            tiers[cid] = None
            # a commit with no files still updates nothing
            continue

        dep = set()      # dependants of T0 (untouched)
        dry = set()      # dependencies of T0 (untouched)
        for f in T0:
            dep |= importers.get(f, set())
            dry |= imports_of.get(f, set())
        dep -= T0
        dry -= T0
        T1 = dep | dry

        # two-hop closure, capped: bounded cost is a hard requirement
        T2 = set()
        if len(T1) <= TIER_CAP:
            for f in T1:
                T2 |= importers.get(f, set())
                T2 |= imports_of.get(f, set())
            T2 -= (T0 | T1)

        TP = set()
        for f in T0:
            TP |= pkg_members.get(pkg(f), set())
        TP -= T0

        TC = set()
        for f in T0:
            part = co.get(f)
            if part:
                top = sorted(part.items(), key=lambda kv: -kv[1])[:CO_CHANGE_TOPK]
                TC |= {g for g, _ in top}
        TC -= T0

        tiers[cid] = {
            "T0": sorted(T0),
            "T1": sorted(T1)[:TIER_CAP],
            "T1_dep": sorted(dep)[:TIER_CAP],
            "T1_dry": sorted(dry)[:TIER_CAP],
            "T2": sorted(T2)[:TIER_CAP],
            "TP": sorted(TP)[:TIER_CAP],
            "TC": sorted(TC)[:TIER_CAP],
        }

        # update co-change AFTER using it -- strictly past
        fl = list(T0)
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                co[fl[i]][fl[j]] += 1
                co[fl[j]][fl[i]] += 1

    return tiers


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"[{PROJECT}] extracting global context")
    commits, touched, imp = extract()
    print(f"  commits={len(commits)} with-files={len(touched)} import-edges={len(imp)}")

    t0 = time.time()
    tiers = build_tiers(commits, touched, imp)
    print(f"  tiers built in {time.time()-t0:.1f}s")

    ok = {k: v for k, v in tiers.items() if v}
    stats = {}
    for name in ("T0", "T1", "T1_dep", "T1_dry", "T2", "TP", "TC"):
        sizes = np.array([len(v[name]) for v in ok.values()], float)
        stats[name] = {"mean": float(sizes.mean()), "max": float(sizes.max()),
                       "nonzero_share": float((sizes > 0).mean())}
        print(f"  {name:<7} mean={sizes.mean():7.1f}  max={sizes.max():5.0f}  "
              f"non-empty on {(sizes>0).mean():5.1%} of commits")

    pickle.dump({"project": PROJECT, "commits": commits, "tiers": tiers,
                 "touched": touched, "n_import_edges": len(imp),
                 "tier_stats": stats, "tier_cap": TIER_CAP},
                open(DEST / "tiers.pkl", "wb"))
    print(f"\nsaved -> {DEST/'tiers.pkl'}")


if __name__ == "__main__":
    main()
