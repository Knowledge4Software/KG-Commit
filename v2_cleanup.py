"""
V2 safe in-place data-quality fixes (idempotent). Modifies the live Neo4j graph;
the build is NOT re-run.

  1. Normalize the `alive` flag: NULL (never-touched base nodes) -> true.
  2. Delete no-op UPDATES edges (old_value == new_value).
  3. Remove the deprecated legacy layer (ASTDiff / ASTEdit + their edges).

Each step prints before/after counts. Re-running is harmless (already-clean
steps report 0 changes).

Run:  python v2_cleanup.py
"""
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")

def main():
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with d.session() as s:
        def count(c): return s.run(c).single()[0]

        # 1. alive flag
        before = count("MATCH (a:ASTNode) WHERE a.alive IS NULL RETURN count(a)")
        s.run("MATCH (a:ASTNode) WHERE a.alive IS NULL SET a.alive = true")
        print(f"[1] alive=NULL normalized to true : {before} nodes")

        # 2. no-op UPDATES
        noop = count("MATCH ()-[r:UPDATES]->() WHERE r.old_value = r.new_value RETURN count(r)")
        s.run("MATCH ()-[r:UPDATES]->() WHERE r.old_value = r.new_value DELETE r")
        print(f"[2] no-op UPDATES deleted          : {noop} edges")

        # 3. legacy layer
        diffs = count("MATCH (n:ASTDiff) RETURN count(n)")
        edits = count("MATCH (n:ASTEdit) RETURN count(n)")
        s.run("MATCH (n:ASTDiff) DETACH DELETE n")
        s.run("MATCH (n:ASTEdit) DETACH DELETE n")
        print(f"[3] legacy layer removed           : {diffs} ASTDiff + {edits} ASTEdit (and edges)")

        # verify
        print("\nverify:")
        print("  alive=NULL remaining :", count("MATCH (a:ASTNode) WHERE a.alive IS NULL RETURN count(a)"))
        print("  no-op UPDATES        :", count("MATCH ()-[r:UPDATES]->() WHERE r.old_value=r.new_value RETURN count(r)"))
        print("  ASTDiff / ASTEdit    :", count("MATCH (n:ASTDiff) RETURN count(n)"),
              "/", count("MATCH (n:ASTEdit) RETURN count(n)"))
    d.close()
    print("\nDone. Re-run v2_audit.py to confirm the before/after.")

if __name__ == "__main__":
    main()
