"""
Step 1 (revised): Add pos_line / pos_col to existing ASTNodes in Neo4j.

Node IDs in Neo4j are  "{rel_path}::A{counter}"  where counter is a
deterministic DFS index produced by _build_ast_subgraph.  We re-run the
SAME traversal logic (in a subprocess for safety) and capture the javalang
position for every node at the exact counter it will receive, then bulk-update
Neo4j.

jlt.Node instances get their pos from node.position (line, col) if available.
Raw-string 'Identifier' leaf nodes get pos_line=-1, pos_col=-1 (no source pos).

Run:
  python add_positions_to_astnodes.py
"""

import subprocess, sys, json, pickle
from pathlib import Path

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (REPO_PATH, OUT, base_commit,
                                    NEO4J_URI, NEO4J_AUTH)

GRAPHS_DIR   = OUT / "file_ast_graphs"   # outputs/<project>/file_ast_graphs
BASE_COMMIT  = base_commit()             # loud check: build step needs a base commit
NEO4J_USER, NEO4J_PASSWORD = NEO4J_AUTH

# ── subprocess worker: mirrors _build_ast_subgraph DFS, returns positions ────
# Returns a JSON list of [line, col] in the exact DFS counter order,
# so index i in the list corresponds to ASTNode id "{rel_path}::A{i}".
WORKER = r"""
import javalang, javalang.tree as jlt, sys, json

src_path = sys.argv[1]
src = open(src_path, encoding='utf-8', errors='replace').read()
try:
    tree = javalang.parse.parse(src)
except Exception:
    print(json.dumps([]))
    sys.exit(0)

positions = []   # index = DFS counter value
stack = [(tree, None, 0)]   # (node_or_str, parent_idx, child_pos)

while stack:
    jval, parent_idx, child_pos = stack.pop()
    idx = len(positions)

    if isinstance(jval, jlt.Node):
        pos = getattr(jval, 'position', None)
        positions.append([int(pos.line), int(pos.column)] if pos else [-1, -1])
        children = []
        cp = 0
        for child in jval.children:
            if child is None:
                continue
            if isinstance(child, jlt.Node):
                children.append((child, idx, cp)); cp += 1
            elif isinstance(child, (list, frozenset, set)):
                for it in child:
                    if isinstance(it, jlt.Node):
                        children.append((it, idx, cp)); cp += 1
                    elif isinstance(it, str) and it.strip():
                        children.append((it, idx, cp)); cp += 1
            elif isinstance(child, str) and child.strip():
                children.append((child, idx, cp)); cp += 1
        stack.extend(reversed(children))
    else:
        positions.append([-1, -1])   # Identifier leaf — no source position

print(json.dumps(positions))
"""

def get_positions(src_bytes: bytes, timeout: int = 45) -> list:
    """Return list of [line, col] in DFS counter order."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
        f.write(src_bytes)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, "-c", WORKER, tmp],
            capture_output=True, text=True, timeout=timeout
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    finally:
        os.unlink(tmp)
    return []

def main():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    pkl_files = sorted(GRAPHS_DIR.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pkl files.")

    total_updated = 0
    total_skipped = 0

    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(
            "CREATE INDEX astnode_pos IF NOT EXISTS "
            "FOR (a:ASTNode) ON (a.file, a.pos_line, a.pos_col)"
        ).consume())

        for pkl_path in pkl_files:
            G, root_nid = pickle.load(open(pkl_path, "rb"))
            rel_path    = G.graph.get("file", "")
            if not rel_path:
                total_skipped += 1
                continue

            # Fetch source from git
            r = subprocess.run(
                ["git", "-C", str(REPO_PATH), "show",
                 f"{BASE_COMMIT}:{rel_path}"],
                capture_output=True
            )
            if r.returncode != 0:
                print(f"  SKIP (git): {rel_path}")
                total_skipped += 1
                continue

            positions = get_positions(r.stdout)
            n_nodes   = G.number_of_nodes()

            if not positions:
                print(f"  SKIP (parse): {rel_path}")
                total_skipped += 1
                continue

            if len(positions) != n_nodes:
                print(f"  WARN count mismatch: DFS={len(positions)} "
                      f"pkl={n_nodes}  {rel_path}")

            # Build update list — node id = f"{rel_path}::A{counter}"
            limit   = min(len(positions), n_nodes)
            updates = [
                {"id": f"{rel_path}::A{i}",
                 "line": positions[i][0],
                 "col":  positions[i][1]}
                for i in range(limit)
            ]

            def _write(tx, upd=updates):
                tx.run("""
                    UNWIND $rows AS r
                    MATCH (a:ASTNode {id: r.id})
                    SET a.pos_line = r.line, a.pos_col = r.col
                """, rows=upd)

            session.execute_write(_write)
            total_updated += len(updates)

            with_pos = sum(1 for p in positions[:limit] if p[0] > 0)
            print(f"  OK  {len(updates):5d} nodes  "
                  f"({with_pos} with pos, {len(updates)-with_pos} leaf)  "
                  f"{rel_path}")

    driver.close()
    print(f"\nDone. {total_updated} ASTNodes updated, "
          f"{total_skipped} files skipped.")

if __name__ == "__main__":
    main()
