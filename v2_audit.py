"""
V2 diagnostic audit of the finalized KG-V1 (read-only).

Produces a precise accounting of every known data-quality gap, as the "before"
snapshot for the V2 debugging work, and writes outputs/v2_audit.json. Re-run
after each fix to confirm the gap counts drop.

Run:  python v2_audit.py
"""
import json
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

def main():
    import sys
    fast = "--fast" in sys.argv        # skip the two slow full group-scans
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    def R(c):
        with d.session(default_access_mode="READ") as s:
            return s.execute_read(lambda tx: [r.data() for r in tx.run(c)])
    def one(c, k):
        return R(c)[0][k]

    A = {}

    # ---- overview (slow full group-scans; skipped in --fast) ----
    if not fast:
        A["nodes_by_label"] = {r["l"]: r["n"] for r in
            R("MATCH (n) RETURN labels(n)[0] AS l, count(n) AS n ORDER BY n DESC")}
        A["rels_by_type"] = {r["t"]: r["n"] for r in
            R("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS n ORDER BY n DESC")}

    # ---- A1 targets ----
    A["alive_flag"] = {str(r["al"]): r["n"] for r in R(
        "MATCH (a:ASTNode) RETURN coalesce(toString(a.alive),'NULL') AS al, count(a) AS n")}
    A["noop_updates"] = one(
        "MATCH ()-[r:UPDATES]->() WHERE r.old_value = r.new_value RETURN count(r) AS n", "n")
    A["legacy"] = dict(
        ASTDiff=one("MATCH (n:ASTDiff) RETURN count(n) AS n", "n"),
        ASTEdit=one("MATCH (n:ASTEdit) RETURN count(n) AS n", "n"),
        HAS_AST_DIFF=one("MATCH ()-[r:HAS_AST_DIFF]->() RETURN count(r) AS n", "n"),
        EDIT=one("MATCH ()-[r:EDIT]->() RETURN count(r) AS n", "n"),
        DIFFED_AT=one("MATCH ()-[r:DIFFED_AT]->() RETURN count(r) AS n", "n"),
    )

    # ---- A3: unparseable java files (touched by a labelled commit, no AST) ----
    gaps = R("""MATCH (c:Commit {in_jit:true})-[:MODIFIED|ADDED]->(f:File)
        WHERE f.id ENDS WITH '.java' AND NOT (f)-[:HAS_AST]->()
        RETURN f.id AS f, count(DISTINCT c) AS commits ORDER BY f""")
    A["unparseable_java_files"] = gaps

    # ---- A2: delta-less labelled commits that modified java ----
    A["in_jit_total"] = one("MATCH (c:Commit {in_jit:true}) RETURN count(c) AS n", "n")
    A["in_jit_with_delta"] = one(
        "MATCH (c:Commit {in_jit:true}) WHERE (c)-[:ADDS|REMOVES|UPDATES|MOVES]->() "
        "RETURN count(c) AS n", "n")
    A["deltaless_modified_java"] = one("""MATCH (c:Commit {in_jit:true})
        WHERE NOT (c)-[:ADDS|REMOVES|UPDATES|MOVES]->()
          AND EXISTS { MATCH (c)-[:MODIFIED]->(f:File) WHERE f.id ENDS WITH '.java' }
        RETURN count(DISTINCT c) AS n""", "n")
    A["suspect_skips"] = one("""MATCH (c:Commit {in_jit:true})
        WHERE NOT (c)-[:ADDS|REMOVES|UPDATES|MOVES]->()
          AND EXISTS { MATCH (c)-[:MODIFIED]->(f:File) WHERE (f)-[:HAS_AST]->() }
        RETURN count(DISTINCT c) AS n""", "n")
    A["deltaless_only_astless_java"] = A["deltaless_modified_java"] - A["suspect_skips"]

    # ---- coverage sanity ----
    A["astnode_total"] = (A["nodes_by_label"].get("ASTNode", 0) if not fast
                          else one("MATCH (a:ASTNode) RETURN count(a) AS n", "n"))
    A["files_with_ast"] = one("MATCH (f:File)-[:HAS_AST]->() RETURN count(DISTINCT f) AS n", "n")

    # ---- A4: structural integrity (added in V2 finalization) ----
    # inserted (is_delta) nodes that are neither a HAS_AST root nor a child of any
    # node -> disconnected from the tree (the insert-parent-None path).
    A["disconnected_inserted"] = one("""MATCH (a:ASTNode {is_delta:true})
        WHERE NOT ( ()-[:AST_CHILD]->(a) ) AND NOT ( (:File)-[:HAS_AST]->(a) )
        RETURN count(a) AS n""", "n")
    # LIVE-tree invariant: an ALIVE node should have exactly one ALIVE parent.
    # (Dead history nodes may retain multiple parents from delete/re-add -- that is
    # preserved history, never traversed as a tree, so it is not an integrity fault.)
    A["multi_parent_nodes"] = one("""MATCH (a:ASTNode)<-[:AST_CHILD]-(p)
        WHERE coalesce(a.alive,true) AND coalesce(p.alive,true)
        WITH a, count(p) AS np WHERE np > 1 RETURN count(a) AS n""", "n")
    # informational: multi-parent among ALL nodes incl. dead history (not a fault)
    A["multi_parent_incl_dead"] = one("""MATCH (a:ASTNode)<-[:AST_CHILD]-(p)
        WITH a, count(p) AS np WHERE np > 1 RETURN count(a) AS n""", "n")
    # rename ghosts (V3/N1): a file renamed away DURING the study window (earliest
    # rename's committed_at < the last labelled commit's) that still carries an alive
    # AST -> should have been retired. Files renamed only AFTER the cutoff are
    # legitimately alive at end-of-window and are counted separately (not a fault).
    A["rename_ghost_asts"] = one("""
        MATCH (ic:Commit {in_jit:true}) WITH max(ic.committed_at) AS cut
        MATCH (c:Commit)-[:RENAMED_FROM]->(f:File)
        WHERE (f)-[:HAS_AST]->()
          AND EXISTS { MATCH (a:ASTNode {file:f.id}) WHERE coalesce(a.alive,true) }
        WITH cut, f, min(c.committed_at) AS fr WHERE fr < cut
        RETURN count(DISTINCT f) AS n""", "n")
    A["rename_kept_post_cutoff"] = one("""
        MATCH (ic:Commit {in_jit:true}) WITH max(ic.committed_at) AS cut
        MATCH (c:Commit)-[:RENAMED_FROM]->(f:File)
        WHERE (f)-[:HAS_AST]->()
          AND EXISTS { MATCH (a:ASTNode {file:f.id}) WHERE coalesce(a.alive,true) }
        WITH cut, f, min(c.committed_at) AS fr WHERE fr >= cut
        RETURN count(DISTINCT f) AS n""", "n")
    # alive nodes whose file has no HAS_AST root at all (fully orphaned subtree).
    A["orphan_alive_no_root"] = one("""MATCH (a:ASTNode) WHERE coalesce(a.alive,true)
          AND NOT EXISTS { MATCH (:File {id:a.file})-[:HAS_AST]->() }
        RETURN count(a) AS n""", "n")
    A["integrity_ok"] = (A["disconnected_inserted"] == 0 and A["multi_parent_nodes"] == 0
                         and A["rename_ghost_asts"] == 0 and A["orphan_alive_no_root"] == 0)

    (OUT/"v2_audit.json").write_text(json.dumps(A, indent=2))
    d.close()

    # ---- console summary ----
    print("="*64); print("KG-V1 DATA-QUALITY AUDIT (V2 before-snapshot)"); print("="*64)
    print(f"ASTNodes {A['astnode_total']:,}  | files with AST {A['files_with_ast']:,}")
    print(f"\n[A1] alive flag      : {A['alive_flag']}")
    print(f"[A1] no-op UPDATES   : {A['noop_updates']}")
    print(f"[A1] legacy layer    : {A['legacy']}")
    print(f"\n[A3] unparseable .java files touched by labelled commits: {len(gaps)}")
    for g in gaps: print(f"        {g['commits']:>3} commit(s)  {g['f']}")
    print(f"\n[A2] labelled commits with deltas      : {A['in_jit_with_delta']:,} / {A['in_jit_total']:,}")
    print(f"[A2] delta-less but modified .java     : {A['deltaless_modified_java']}")
    print(f"        of which file had NO AST (expected): {A['deltaless_only_astless_java']}")
    print(f"        of which file HAD AST (SUSPECT SKIP): {A['suspect_skips']}")
    ok = lambda v: "OK" if v == 0 else "FAIL"
    print(f"\n[A4] structural integrity (each should be 0):")
    print(f"        disconnected inserted nodes : {A['disconnected_inserted']:>7}  [{ok(A['disconnected_inserted'])}]")
    print(f"        multi-parent AST_CHILD      : {A['multi_parent_nodes']:>7}  [{ok(A['multi_parent_nodes'])}]")
    print(f"        rename-ghost alive ASTs     : {A['rename_ghost_asts']:>7}  [{ok(A['rename_ghost_asts'])}]")
    print(f"        (post-cutoff renames kept   : {A['rename_kept_post_cutoff']:>7}  informational)")
    print(f"        orphan alive (no root)      : {A['orphan_alive_no_root']:>7}  [{ok(A['orphan_alive_no_root'])}]")
    print(f"        => integrity_ok = {A['integrity_ok']}")
    print(f"\nwrote -> {OUT/'v2_audit.json'}")

if __name__ == "__main__":
    main()
