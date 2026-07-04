"""
V3 A2 -- LIVE-tree multi-parent repair (alive-scoped, principled).

Invariant: every ALIVE ASTNode must have exactly one ALIVE parent (the live tree
is a tree). Dead history nodes may keep multiple parents from delete/re-add -- that
is preserved history, never traversed as a tree, so it is left untouched.

After the rename-retirement, only ~145 alive nodes (4 files) still have >1 alive
parent -- stale cross-edges from files that were deleted-and-re-added, which reused
the deterministic {file}::A{i} ids. The correct parent is the NEAREST ancestor
(largest depth; the extra edge points to a shallower old-incarnation node). We keep
the deepest alive parent (tie-break by source position) and delete the other
alive-parent edges. All deletions are backed up for undo.

  python v3_fix_multiparent.py            # DRY RUN
  python v3_fix_multiparent.py --apply
  python v3_fix_multiparent.py --undo --apply
"""
import sys, json
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")
BACKUP = Path("outputs/v3_multiparent_backup.json")
APPLY = "--apply" in sys.argv
UNDO = "--undo" in sys.argv


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    if UNDO:
        if not BACKUP.exists():
            print("no backup; nothing to undo."); driver.close(); return
        edges = json.loads(BACKUP.read_text())
        if APPLY:
            with driver.session() as s:
                for i in range(0, len(edges), 2000):
                    b = edges[i:i+2000]
                    s.execute_write(lambda tx: tx.run("""
                        UNWIND $b AS e MATCH (p:ASTNode {id:e.parent}), (c:ASTNode {id:e.child})
                        MERGE (p)-[r:AST_CHILD]->(c) ON CREATE SET r.pos=e.pos
                    """, b=b))
            print(f"UNDONE: restored {len(edges)} edges.")
        else:
            print(f"would restore {len(edges)} edges. add --apply.")
        driver.close(); return

    with driver.session() as s:
        # alive nodes with >1 ALIVE parent, with each parent's depth/pos/pos to choose
        rows = s.run("""
            MATCH (a:ASTNode)<-[rc:AST_CHILD]-(p)
            WHERE coalesce(a.alive,true) AND coalesce(p.alive,true)
            WITH a, collect({pid:p.id, depth:coalesce(p.depth,0),
                             line:coalesce(p.pos_line,-1), pos:rc.pos}) AS ps
            WHERE size(ps) > 1
            RETURN a.id AS child, a.file AS file, ps AS parents
        """).data()
        print(f"{len(rows)} alive nodes with >1 alive parent, "
              f"over {len({r['file'] for r in rows})} files")

        backup, deletions = [], []
        for r in rows:
            ps = r["parents"]
            # keep the NEAREST ancestor: max depth, then max line (closest above), then pid
            keep = sorted(ps, key=lambda x: (-x["depth"], -x["line"], x["pid"]))[0]
            for x in ps:
                if x["pid"] != keep["pid"]:
                    backup.append({"child": r["child"], "parent": x["pid"], "pos": x["pos"]})
                    deletions.append((x["pid"], r["child"]))
        print(f"surplus alive-parent edges to delete: {len(deletions)}")
        for r in list({row["file"] for row in rows}):
            fdel = sum(1 for d in deletions if d[1] in
                       {row["child"] for row in rows if row["file"] == r})
            print(f"   {fdel:4d}  {r.split('/')[-1]}")

        if not APPLY:
            print("\nDRY RUN. add --apply (edges backed up first)."); driver.close(); return

        BACKUP.write_text(json.dumps(backup))
        print(f"backed up {len(backup)} edges -> {BACKUP}")
        for i in range(0, len(deletions), 2000):
            b = [{"p": p, "c": c} for p, c in deletions[i:i+2000]]
            s.execute_write(lambda tx: tx.run("""
                UNWIND $b AS e MATCH (p:ASTNode {id:e.p})-[r:AST_CHILD]->(c:ASTNode {id:e.c})
                DELETE r
            """, b=b))
        print(f"APPLIED: deleted {len(deletions)} stale alive-parent edges.")
    driver.close()


if __name__ == "__main__":
    main()
