#!/usr/bin/env python3
"""
Step 3: populate apache/groovy commit history into the local Neo4j KG using
the teammate's own JITCommitKnowledgeGraph pipeline, then attach the
file-level AST graphs built by build_file_ast_graphs.py under each File node
(File as root), then run post-merge statistical analysis.

Neo4j schema added here:
  (:ASTNode {id, file, ast_type, group, value, is_leaf, depth})
  (:File)-[:HAS_AST]->(:ASTNode)            one root per base-snapshot file
  (:ASTNode)-[:AST_CHILD {pos}]->(:ASTNode) tree edges within one file's AST

Run:
  python ingest_base_kg_and_ast.py                          # full pipeline
  python ingest_base_kg_and_ast.py --skip-commits            # AST attach + stats only
  python ingest_base_kg_and_ast.py --skip-commits --skip-ast  # stats only
"""

import sys, json, pickle, csv, argparse
from pathlib import Path

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (OUT, REPO_PATH, REPO_PATHS, PROJECT_KEY,
                                    CSV_PATH, NEO4J_URI, NEO4J_AUTH)

from kg_commit.knowledge.dataloader import CommitDataLoader
from kg_commit.knowledge.parsers import FilteredCommitParser
from kg_commit.knowledge.graph import JITCommitKnowledgeGraph

OUT_DIR   = OUT                          # outputs/<project>/  (was flat outputs/)
GRAPH_DIR = OUT_DIR / "file_ast_graphs"
NEO4J_USER, NEO4J_PASSWORD = NEO4J_AUTH


# ── Step 3.1: core-entity ingestion (teammate's pipeline, groovy only) ──────
def ingest_commits(kg: JITCommitKnowledgeGraph) -> None:
    from tqdm import tqdm

    # Multi-repository ingestion (see docs/Critical_notes.tex). A project's
    # labelled commits may be split across several repos (e.g. hadoop-mapreduce:
    # spin-off + monorepo). Policy:
    #   * PRIMARY repo (REPO_PATHS[0]): stream its FULL history (as always), so
    #     the base graph keeps its unlabelled context commits.
    #   * SECONDARY repos: stream ONLY the labelled commits present there
    #     (git log --no-walk over the labelled SHA set), so a large monorepo does
    #     not add tens of thousands of irrelevant commit nodes.
    # Commits are DEDUPLICATED by id across repos (primary wins). For single-repo
    # projects REPO_PATHS == [REPO_PATH] and behaviour is byte-for-byte unchanged.
    import csv as _csv
    labelled = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            cid = (r.get("commit_id") or r.get("commit") or "").strip()
            if cid:
                labelled.add(cid)
    print(f"{len(labelled)} labelled commits in the ApacheJIT CSV.", flush=True)

    print("Purging database for a fresh chronological run...", flush=True)
    counters = kg.purge_database()
    print(f"  purged: {counters}", flush=True)

    seen = set()

    def multi_repo_stream():
        for i, repo_path in enumerate(REPO_PATHS):
            is_primary = (i == 0)
            dl = CommitDataLoader(repo_map={PROJECT_KEY: str(repo_path)})
            # secondary repos: only the labelled SHAs not already ingested
            only = None
            if not is_primary:
                only = [s for s in labelled if s not in seen]
                if not only:
                    print(f"  {repo_path.name}: no remaining labelled commits, "
                          f"skipping.", flush=True)
                    continue
            n_repo = 0
            with tqdm(desc=f"[{PROJECT_KEY}] ingest {repo_path.name}"
                           f"{'' if is_primary else ' (labelled-only)'}",
                      unit=" commit", dynamic_ncols=True) as pbar:
                for parsed in dl.fetch_all_commits_fast(project=PROJECT_KEY,
                                                        limit=-1, only_commits=only):
                    cid = parsed.get("commit_id", "")
                    if not cid or cid in seen:
                        continue                       # dedup across repos
                    seen.add(cid)
                    n_repo += 1
                    pbar.set_postfix(new_from_repo=n_repo, total=len(seen),
                                     commit=cid[:8])
                    pbar.update(1)
                    yield parsed
            print(f"  {repo_path.name}: {n_repo} new commits "
                  f"(running total {len(seen)})", flush=True)

    ingested = kg.ingest_fast(multi_repo_stream())
    got = len(labelled & seen)
    print(f"Ingested {ingested} commits for {PROJECT_KEY} across "
          f"{len(REPO_PATHS)} repo(s); labelled coverage {got}/{len(labelled)}.",
          flush=True)
    if got < len(labelled):
        print(f"  WARNING: {len(labelled)-got} labelled commits were NOT found "
              f"in any configured repo.", flush=True)


# ── Step 3.2: attach each file's AST graph under its File node ─────────────
def attach_ast_graphs(kg: JITCommitKnowledgeGraph) -> None:
    with kg.driver.session() as session:
        session.execute_write(lambda tx: tx.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:ASTNode) REQUIRE a.id IS UNIQUE"
        ).consume())

    def _attach_one(tx, file_id, nodes, edges, root_id):
        tx.run(
            """
            MERGE (f:File {id: $file_id})
            WITH f
            UNWIND $nodes AS n
            MERGE (a:ASTNode {id: n.id})
            ON CREATE SET a.file = $file_id, a.ast_type = n.ast_type,
                          a.group = n.group, a.value = n.value,
                          a.is_leaf = n.is_leaf, a.depth = n.depth
            """,
            file_id=file_id, nodes=nodes,
        )
        if edges:
            tx.run(
                """
                UNWIND $edges AS e
                MATCH (p:ASTNode {id: e.parent}), (c:ASTNode {id: e.child})
                MERGE (p)-[r:AST_CHILD]->(c)
                ON CREATE SET r.pos = e.pos
                """,
                edges=edges,
            )
        tx.run(
            """
            MATCH (f:File {id: $file_id}), (r:ASTNode {id: $root_id})
            MERGE (f)-[:HAS_AST]->(r)
            """,
            file_id=file_id, root_id=root_id,
        )

    pkl_files = sorted(GRAPH_DIR.glob("*.pkl"))
    print(f"Attaching {len(pkl_files)} file AST graphs to Neo4j...", flush=True)

    with kg.driver.session() as session:
        for i, pf in enumerate(pkl_files, 1):
            G, root_nid = pickle.loads(pf.read_bytes())
            file_id = G.graph["file"]
            nodes = [
                {"id": nid, "ast_type": d.get("ast_type"), "group": d.get("group"),
                 "value": d.get("value"), "is_leaf": bool(d.get("is_leaf")),
                 "depth": int(d.get("depth", 0))}
                for nid, d in G.nodes(data=True)
            ]
            edges = [
                {"parent": u, "child": v, "pos": d.get("pos")}
                for u, v, d in G.edges(data=True)
            ]
            session.execute_write(_attach_one, file_id, nodes, edges, root_nid)
            print(f"  [{i}/{len(pkl_files)}] {file_id}  ({len(nodes)} nodes, {len(edges)} edges)", flush=True)

    print(f"Attached AST for {len(pkl_files)} files.", flush=True)


# ── Step 3.3: post-merge statistical analysis ───────────────────────────────
def post_merge_stats(kg: JITCommitKnowledgeGraph) -> None:
    with kg.driver.session() as session:
        label_counts = session.execute_read(lambda tx: [r.data() for r in tx.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS total ORDER BY total DESC"
        )])
        rel_counts = session.execute_read(lambda tx: [r.data() for r in tx.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS total ORDER BY total DESC"
        )])
        file_count = session.execute_read(lambda tx: tx.run(
            "MATCH (f:File) RETURN count(f) AS c"
        ).single()["c"])
        has_ast_count = session.execute_read(lambda tx: tx.run(
            "MATCH (f:File)-[:HAS_AST]->() RETURN count(f) AS c"
        ).single()["c"])
        node_per_file = {r["file"]: r["n"] for r in session.execute_read(lambda tx: [r.data() for r in tx.run(
            "MATCH (a:ASTNode) RETURN a.file AS file, count(a) AS n ORDER BY file"
        )])}
        edge_per_file = {r["file"]: r["n"] for r in session.execute_read(lambda tx: [r.data() for r in tx.run(
            "MATCH (a:ASTNode)-[:AST_CHILD]->() RETURN a.file AS file, count(*) AS n ORDER BY file"
        )])}

    # Cross-check against the Step-2 CSV (outputs/file_ast_subgraph_stats.csv)
    step2_csv = OUT_DIR / "file_ast_subgraph_stats.csv"
    step2_rows = {}
    with open(step2_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step2_rows[row["file"]] = row

    out_rows = []
    mismatches = 0
    for file_id in sorted(set(step2_rows) | set(node_per_file)):
        expected_n = int(step2_rows.get(file_id, {}).get("n_ast_nodes", -1))
        expected_e = int(step2_rows.get(file_id, {}).get("n_edges", -1))
        actual_n = node_per_file.get(file_id, 0)
        actual_e = edge_per_file.get(file_id, 0)
        ok = (expected_n == actual_n) and (expected_e == actual_e)
        if not ok:
            mismatches += 1
        out_rows.append({
            "file": file_id,
            "n_ast_nodes_csv": expected_n, "n_ast_nodes_neo4j": actual_n,
            "n_edges_csv": expected_e, "n_edges_neo4j": actual_e,
            "match": ok,
        })

    with open(OUT_DIR / "file_ast_neo4j_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    sizes = sorted(node_per_file.values())
    summary = {
        "label_counts": label_counts,
        "relationship_counts": rel_counts,
        "file_node_count": file_count,
        "file_with_ast_count": has_ast_count,
        "ast_files_expected": len(step2_rows),
        "csv_vs_neo4j_mismatches": mismatches,
        "ast_size_per_file": {
            "min": sizes[0] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "mean": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "max": sizes[-1] if sizes else 0,
        },
    }
    with open(OUT_DIR / "file_ast_neo4j_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nFile coverage: {has_ast_count}/{file_count} File nodes have a HAS_AST root "
          f"(expected ~{len(step2_rows)} base-snapshot files).", flush=True)
    print(f"CSV vs Neo4j mismatches: {mismatches}/{len(out_rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-commits", action="store_true", help="Skip core-entity (Commit/Project/Developer/File) ingestion.")
    ap.add_argument("--skip-ast", action="store_true", help="Skip attaching file AST graphs.")
    ap.add_argument("--skip-stats", action="store_true", help="Skip post-merge statistical analysis.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kg = JITCommitKnowledgeGraph(
        uri=NEO4J_URI, auth_user=NEO4J_USER, auth_pass=NEO4J_PASSWORD,
        parser=FilteredCommitParser(),
    )
    try:
        if not args.skip_commits:
            ingest_commits(kg)
        if not args.skip_ast:
            attach_ast_graphs(kg)
        if not args.skip_stats:
            post_merge_stats(kg)
    finally:
        kg.close()


if __name__ == "__main__":
    main()