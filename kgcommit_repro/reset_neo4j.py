"""
Full Neo4j wipe -- run this BEFORE building a DIFFERENT project.

The Neo4j database is single-tenant: nodes are NOT project-tagged, and the online
build streams `MATCH (c:Commit {in_jit:true})` globally. The DB therefore holds
exactly ONE project's knowledge graph at a time. To switch from project A to
project B you must first wipe A entirely; this script deletes every node and
relationship (Core process layer, AST delta layer, CFG/DFG/PDG/SEQ layers, and
the CSTG Term/Intent layer) so the next build starts clean.

It also removes the active project's on-disk build checkpoints so the online
engines restart from commit 0 rather than resuming a stale state.

This is deliberately explicit and guarded (requires --yes) because it is
destructive. It does NOT touch data/ or repos/ (read in place) or the per-project
outputs/ artifacts (those are namespaced and safe to keep).

Run:
    (PowerShell)  $env:KGC_PROJECT='zookeeper'; python reset_neo4j.py --yes
    (bash)        KGC_PROJECT=zookeeper python reset_neo4j.py --yes
"""
import argparse
import os
import subprocess
import sys
import time

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (NEO4J_URI, NEO4J_AUTH, PROJECT,
                                    CKPT_PATH, ckpt_path)

CONTAINER = os.environ.get("KGC_NEO4J_CONTAINER", "neo4j-local")
IMAGE = os.environ.get("KGC_NEO4J_IMAGE", "neo4j:latest")
DB = os.environ.get("KGC_NEO4J_DB", "neo4j")


def _sh(*a, check=True, capture=False, env=None):
    r = subprocess.run(list(a), text=True, capture_output=capture, env=env)
    if check and r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(a)}")
    return r


def _host_datadir():
    env = os.environ.get("KGC_NEO4J_DATADIR")
    if env:
        return env
    fmt = '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}'
    return _sh("docker", "inspect", CONTAINER, "--format", fmt, capture=True).stdout.strip()


def _docker_vol(datadir):
    p = str(datadir).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "//" + p[0].lower() + p[2:]
    return f"{p}://data"


def _store_wipe():
    """Reliable wipe of any size: stop container, delete DB store files, restart."""
    datadir = _host_datadir()
    if not datadir:
        sys.exit("Could not find the Neo4j /data bind mount; set KGC_NEO4J_DATADIR, "
                 "or use --cypher for a (small-graph) in-database delete.")
    print(f"Store-level wipe of database '{DB}' (container {CONTAINER}) ...")
    _sh("docker", "stop", CONTAINER, capture=True)
    env = dict(os.environ); env["MSYS_NO_PATHCONV"] = "1"
    _sh("docker", "run", "--rm", "--user", "root", "-v", _docker_vol(datadir), IMAGE,
        "bash", "-c", f"rm -rf /data/databases/{DB} /data/transactions/{DB}", env=env)
    _sh("docker", "start", CONTAINER, capture=True)
    # wait for the fresh empty DB to come online
    from neo4j import GraphDatabase
    for _ in range(40):
        try:
            d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            with d.session() as s:
                n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            d.close()
            print(f"Neo4j online; empty database (node count {n}).")
            return
        except Exception:
            time.sleep(3)
    print("Neo4j restarted (could not confirm node count within timeout).")


def _cypher_wipe():
    """Batched in-database delete -- only for SMALL graphs (can OOM otherwise)."""
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with d.session() as s:
        n0 = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        print(f"Cypher wipe -- {n0:,} nodes present ...")
        total = 0
        while True:
            k = s.run("MATCH (n) WITH n LIMIT 20000 DETACH DELETE n "
                      "RETURN count(n) AS k").single()["k"]
            if not k:
                break
            total += k; print(f"  deleted {total:,} nodes ...")
        for kind in ("INDEXES", "CONSTRAINTS"):
            try:
                for row in s.run(f"SHOW {kind} YIELD name RETURN name").data():
                    s.run(f"DROP {kind[:-1]} `{row['name']}` IF EXISTS")
            except Exception:
                pass
    d.close()
    print("Neo4j is empty.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                    help="confirm the destructive full-DB wipe")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="do not delete this project's build checkpoints")
    ap.add_argument("--cypher", action="store_true",
                    help="force in-database batched delete (small graphs only) "
                         "instead of the default store-level wipe")
    args = ap.parse_args()

    if not args.yes:
        sys.exit("Refusing to wipe Neo4j without --yes. This deletes the ENTIRE "
                 "graph (one project resident at a time). Re-run with --yes.\n"
                 "TIP: to KEEP the current project's graph, snapshot it first:\n"
                 "     python snapshot_neo4j.py dump")

    # A full multi-million-node graph cannot be cleared by a batched Cypher
    # DETACH DELETE (it exhausts the container heap -> OutOfMemoryError). The
    # reliable wipe is store-level: stop the Neo4j container, delete the database
    # store files in a transient container, restart. Use --cypher to force the old
    # batched delete for a small graph instead.
    if args.cypher:
        _cypher_wipe()
    else:
        _store_wipe()

    if not args.keep_checkpoints:
        removed = []
        for p in [CKPT_PATH] + [ckpt_path(k) for k in ("cfg", "dfg", "pdg", "seq")]:
            if p.exists():
                p.unlink(); removed.append(p.name)
        if removed:
            print(f"Removed {PROJECT} build checkpoints: {', '.join(removed)}")


if __name__ == "__main__":
    main()
