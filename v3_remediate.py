"""
V3 Track-A graph remediation (in-place, reversible).

Every mutation is TAGGED so it can be undone surgically:
  * rename-retirement -> nodes get v3_retired=true; REMOVES edges get v3_op='rename_retire'
  * multi-parent repair -> deleted AST_CHILD edges are backed up to
    outputs/v3_multiparent_backup.json before deletion.

Subcommands (dry-run unless --apply):
  python v3_remediate.py diagnose            # inspect ghosts + multi-parent (read-only)
  python v3_remediate.py retire-renames --apply
  python v3_remediate.py repair-multiparent --apply
  python v3_remediate.py undo-renames --apply
  python v3_remediate.py undo-multiparent --apply

Safety net: the user is advised to also take a full offline dump
  neo4j stop && neo4j-admin database dump neo4j --to-path=<dir> && neo4j start
before the first --apply (belt-and-suspenders in addition to the tags).
"""
import sys, json, subprocess
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")
REPO = Path(__file__).resolve().parent / "repos" / "apache" / "groovy"
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
MP_BACKUP = OUT / "v3_multiparent_backup.json"
APPLY = "--apply" in sys.argv


def exists_at_head(path):
    """True if `path` exists in the repo's current HEAD tree (not a ghost)."""
    r = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"HEAD:{path}"],
                       capture_output=True)
    return r.returncode == 0


# ── ghost / rename retirement (A1 / N1) ──────────────────────────────────────
def _ghost_candidates(session):
    """Files that were a rename SOURCE, still carry an alive AST, and do NOT exist
    at HEAD -> genuine ghosts. Returns [(file, retire_commit, alive_count)]."""
    # ANY commit that renamed the file away (not just in_jit): the online build
    # only processed in_jit commits, so files renamed by non-labelled commits were
    # never retired and are the bulk of the ghosts.
    #
    # CRUCIAL window guard: only retire files renamed away DURING the study window
    # (earliest rename's committed_at < the last labelled commit's committed_at).
    # Files renamed only AFTER the cutoff were legitimately alive at end-of-window
    # and must be kept. Attribute the retirement to the rename commit (prefer an
    # in_jit one so its REMOVES delta is recorded; else the within-window rename).
    cut = session.run("MATCH (c:Commit {in_jit:true}) "
                      "RETURN max(c.committed_at) AS m").single()["m"]
    rows = session.run("""
        MATCH (c:Commit)-[:RENAMED_FROM]->(f:File)
        WHERE (f)-[:HAS_AST]->()
          AND EXISTS { MATCH (a:ASTNode {file:f.id}) WHERE coalesce(a.alive,true) }
        WITH f, min(c.committed_at) AS first_ren,
             collect({id:c.id, injit:coalesce(c.in_jit,false), ts:c.committed_at}) AS cs
        WHERE first_ren < $cut
        WITH f, [x IN cs WHERE x.injit][0] AS injit_c,
                [x IN cs WHERE x.ts < $cut][0] AS win_c
        WITH f, coalesce(injit_c, win_c) AS rc
        MATCH (a:ASTNode {file:f.id}) WHERE coalesce(a.alive,true)
        RETURN f.id AS f, rc.id AS c, coalesce(rc.injit,false) AS injit, count(a) AS alive
        ORDER BY alive DESC
    """, cut=cut).data()
    return rows


def retire_renames(driver):
    with driver.session() as s:
        cands = _ghost_candidates(s)
        ghosts = [r for r in cands if not exists_at_head(r["f"])]
        skipped = [r for r in cands if exists_at_head(r["f"])]
        tot_alive = sum(r["alive"] for r in ghosts)
        print(f"rename-source files with alive AST : {len(cands)}")
        print(f"  -> genuine ghosts (absent at HEAD): {len(ghosts)}  ({tot_alive:,} alive nodes)")
        print(f"  -> still exist at HEAD (skip)     : {len(skipped)}")
        print(f"  -> attributed to in_jit / non-in_jit rename commits: "
              f"{sum(1 for r in ghosts if r['injit'])} / {sum(1 for r in ghosts if not r['injit'])}")
        if not APPLY:
            print("\nDRY RUN. Re-run with --apply to retire ghost ASTs."); return
        n = 0
        for r in ghosts:
            # retire the alive nodes; add a REMOVES delta ONLY when the rename commit
            # is labelled (non-in_jit commits do not carry deltas).
            cy = ("""MATCH (c:Commit {id:$cid}), (a:ASTNode {file:$f}) WHERE coalesce(a.alive,true)
                     SET a.alive=false, a.removed_by=$cid, a.v3_retired=true
                     MERGE (c)-[rr:REMOVES]->(a) SET rr.v3_op='rename_retire'
                     RETURN count(a) AS n""" if r["injit"] else
                  """MATCH (a:ASTNode {file:$f}) WHERE coalesce(a.alive,true)
                     SET a.alive=false, a.removed_by=$cid, a.v3_retired=true
                     RETURN count(a) AS n""")
            res = s.execute_write(lambda tx: tx.run(cy, cid=r["c"], f=r["f"]).single()["n"])
            n += res
        print(f"\nAPPLIED: retired {n:,} ghost nodes over {len(ghosts)} files.")


def undo_renames(driver):
    with driver.session() as s:
        if not APPLY:
            n = s.run("MATCH (a:ASTNode {v3_retired:true}) RETURN count(a) AS n").single()["n"]
            print(f"would restore {n:,} retired nodes. Re-run with --apply."); return
        s.run("MATCH ()-[r:REMOVES {v3_op:'rename_retire'}]->() DELETE r")
        n = s.run("""MATCH (a:ASTNode {v3_retired:true})
            SET a.alive=true REMOVE a.removed_by, a.v3_retired RETURN count(a) AS n
        """).single()["n"]
        print(f"UNDONE: restored {n:,} nodes.")


# ── disconnected inserts (A5 / N3) ───────────────────────────────────────────
def fix_disconnected(driver):
    """Wire the few is_delta nodes that have neither a parent AST_CHILD nor a
    HAS_AST root to their file's root (a connected, reversible placement). Tagged
    v3_reparented for undo."""
    with driver.session() as s:
        rows = s.run("""
            MATCH (a:ASTNode {is_delta:true})
            WHERE NOT ( ()-[:AST_CHILD]->(a) ) AND NOT ( (:File)-[:HAS_AST]->(a) )
            RETURN a.id AS id, a.file AS file
        """).data()
        print(f"disconnected inserted nodes: {len(rows)}")
        for r in rows:
            print(f"   {r['id']}")
        if not APPLY:
            print("\nDRY RUN. add --apply to reparent them to their file root."); return
        done = 0
        for r in rows:
            done += s.execute_write(lambda tx: tx.run("""
                MATCH (a:ASTNode {id:$id})
                MATCH (:File {id:$file})-[:HAS_AST]->(root:ASTNode)
                MERGE (root)-[e:AST_CHILD]->(a) ON CREATE SET e.pos=-1, e.v3_reparented=true
                RETURN count(a) AS n
            """, id=r["id"], file=r["file"]).single()["n"])
        print(f"APPLIED: reparented {done} disconnected nodes to their file root.")


def undo_disconnected(driver):
    with driver.session() as s:
        if not APPLY:
            n = s.run("MATCH ()-[e:AST_CHILD {v3_reparented:true}]->() RETURN count(e) AS n").single()["n"]
            print(f"would delete {n} reparent edges. add --apply."); return
        n = s.run("MATCH ()-[e:AST_CHILD {v3_reparented:true}]->() DELETE e RETURN count(e) AS n").single()["n"]
        print(f"UNDONE: removed {n} reparent edges.")


# ── multi-parent repair (A2 / N2) ────────────────────────────────────────────
def diagnose_multiparent(session, sample=8):
    rows = session.run("""
        MATCH (a:ASTNode)<-[:AST_CHILD]-(p) WITH a, count(p) AS np WHERE np>1
        RETURN a.id AS id, a.file AS file, a.is_delta AS is_delta,
               coalesce(a.alive,true) AS alive, np ORDER BY np DESC LIMIT $k
    """, k=sample).data()
    total = session.run("""MATCH (a:ASTNode)<-[:AST_CHILD]-(p)
        WITH a, count(p) AS np WHERE np>1 RETURN count(a) AS n""").single()["n"]
    print(f"multi-parent nodes: {total:,}. Sample:")
    for r in rows:
        parents = session.run("""MATCH (p)-[:AST_CHILD]->(a:ASTNode {id:$id})
            RETURN p.id AS pid, coalesce(p.alive,true) AS palive""", id=r["id"]).data()
        print(f"  {r['id']}  np={r['np']} delta={r['is_delta']} alive={r['alive']}")
        for p in parents:
            print(f"       parent {p['pid']}  alive={p['palive']}")
    return total


def repair_multiparent(driver):
    """Keep exactly one incoming AST_CHILD per node (prefer an ALIVE parent; break
    ties by lexicographically smallest parent id for determinism); back up and
    delete the rest."""
    with driver.session() as s:
        total = diagnose_multiparent(s)
        if not APPLY:
            print("\nDRY RUN. Re-run with --apply to repair (edges backed up first)."); return
        # collect all surplus edges to delete, with full info for undo
        surplus = s.run("""
            MATCH (a:ASTNode)<-[rc:AST_CHILD]-(p)
            WITH a, collect({pid:p.id, palive:coalesce(p.alive,true), pos:rc.pos}) AS ps
            WHERE size(ps) > 1
            RETURN a.id AS child, ps AS parents
        """).data()
        backup, to_delete = [], []
        for row in surplus:
            ps = row["parents"]
            # keep: an alive parent first, then smallest pid
            keep = sorted(ps, key=lambda x: (0 if x["palive"] else 1, x["pid"]))[0]
            for x in ps:
                if x["pid"] != keep["pid"]:
                    backup.append({"child": row["child"], "parent": x["pid"], "pos": x["pos"]})
                    to_delete.append((x["pid"], row["child"]))
        MP_BACKUP.write_text(json.dumps(backup))
        print(f"backed up {len(backup)} surplus edges -> {MP_BACKUP}")
        for i in range(0, len(to_delete), 5000):
            batch = [{"p": p, "c": c} for p, c in to_delete[i:i+5000]]
            s.execute_write(lambda tx: tx.run("""
                UNWIND $b AS e MATCH (p:ASTNode {id:e.p})-[r:AST_CHILD]->(c:ASTNode {id:e.c})
                DELETE r
            """, b=batch))
        print(f"APPLIED: deleted {len(to_delete)} surplus AST_CHILD edges "
              f"over {len(surplus)} nodes.")


def undo_multiparent(driver):
    if not MP_BACKUP.exists():
        print("no backup file; nothing to undo."); return
    backup = json.loads(MP_BACKUP.read_text())
    if not APPLY:
        print(f"would restore {len(backup)} edges from {MP_BACKUP}. Re-run --apply."); return
    with driver.session() as s:
        for i in range(0, len(backup), 5000):
            b = backup[i:i+5000]
            s.execute_write(lambda tx: tx.run("""
                UNWIND $b AS e MATCH (p:ASTNode {id:e.parent}), (c:ASTNode {id:e.child})
                MERGE (p)-[r:AST_CHILD]->(c) ON CREATE SET r.pos=e.pos
            """, b=b))
    print(f"UNDONE: restored {len(backup)} AST_CHILD edges.")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        if cmd == "diagnose":
            with driver.session() as s:
                print("== ghosts =="); _ = _ghost_candidates(s)
                print(f"rename-source-with-alive-AST files: {len(_)}")
                print("\n== multi-parent =="); diagnose_multiparent(s)
        elif cmd == "retire-renames":   retire_renames(driver)
        elif cmd == "undo-renames":     undo_renames(driver)
        elif cmd == "fix-disconnected": fix_disconnected(driver)
        elif cmd == "undo-disconnected": undo_disconnected(driver)
        elif cmd == "repair-multiparent": repair_multiparent(driver)
        elif cmd == "undo-multiparent": undo_multiparent(driver)
        else: print(f"unknown command: {cmd}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
