"""
V2 A2: recover the 'suspect-skip' commits -- labelled commits that modified an
AST-bearing file yet carry no delta edges.

A suspect can be SAFELY recovered only if the commit is the LAST labelled commit
to modify that file: then the graph's stored state for the file is exactly the
commit's 'before', so re-diffing stored-state -> commit and applying it gives the
commit its delta without double-counting (a non-last suspect's change was already
absorbed by a later commit's delta -- the documented 'bundling' behaviour).

We reuse the online engine verbatim: build_online_kg.process_modify with the
file's node-identity map from the checkpoint, so identity threading stays exact.

  python recover_skipped_deltas.py            # DRY RUN (classify only)
  python recover_skipped_deltas.py --apply     # apply safe recoveries + save checkpoint
"""
import sys, json
from pathlib import Path
from neo4j import GraphDatabase
import build_online_kg as bok
import online_ast_diff as oad

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")
APPLY = "--apply" in sys.argv

def diff_nonempty(d):
    return d is not None and (d["deleted"] or d["inserted"] or d["moved"]
                              or any(m[2] for m in d["matched"]))

def main():
    ck = json.loads(bok.CKPT_PATH.read_text())
    file_state, file_maps = ck["file_state"], ck.get("file_maps", {})

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(default_access_mode="READ") as s:
        rows = s.execute_read(lambda tx: [r.data() for r in tx.run("""
            MATCH (c:Commit {in_jit:true})
            WHERE NOT (c)-[:ADDS|REMOVES|UPDATES|MOVES]->()
            MATCH (c)-[:MODIFIED]->(f:File) WHERE (f)-[:HAS_AST]->()
            WITH c, f, c.author_ts AS cts
            OPTIONAL MATCH (c2:Commit {in_jit:true})-[:MODIFIED]->(f)
                WHERE c2.author_ts > cts
            RETURN c.id AS c, cts AS ts, f.id AS f, count(c2) AS later
            ORDER BY cts
        """)])
    print(f"{len(rows)} suspect (commit,file) pairs over "
          f"{len({r['c'] for r in rows})} commits")

    safe   = [r for r in rows if r["later"] == 0]
    bundled = [r for r in rows if r["later"] > 0]
    print(f"  safe (commit is last toucher of file) : {len(safe)}")
    print(f"  bundled into a later commit (residual): {len(bundled)}\n")

    stats = dict(recovered=0, noop=0, gitgap=0, parsefail=0)
    recovered_commits = set()
    sess = driver.session()        # write session for --apply
    for r in sorted(safe, key=lambda x: x["ts"]):
        c, f = r["c"], r["f"]
        before = bok.git_bytes(file_state.get(f, ""), f) if f in file_state else None
        after  = bok.git_bytes(c, f)
        if before is None or after is None:
            stats["gitgap"] += 1; continue
        d = oad.diff_sources(before, after, f)
        if d is None:
            stats["parsefail"] += 1; continue
        if not diff_nonempty(d):
            stats["noop"] += 1; continue
        # real change the build missed
        if APPLY:
            res = bok.process_modify(sess, c, f, before, after, file_maps.get(f, {}))
            if res is None:
                stats["parsefail"] += 1; continue
            _, new_map = res
            file_state[f] = c; file_maps[f] = new_map
        stats["recovered"] += 1; recovered_commits.add(c)
    sess.close(); driver.close()

    print("classification of SAFE candidates:")
    print(f"  recoverable (non-empty diff) : {stats['recovered']}  "
          f"over {len(recovered_commits)} commits")
    print(f"  genuine no-op (empty diff)   : {stats['noop']}")
    print(f"  git/parse gaps               : {stats['gitgap']} / {stats['parsefail']}")
    print(f"  residual (bundled, documented): {len(bundled)} pairs")

    if APPLY:
        ck["file_state"], ck["file_maps"] = file_state, file_maps
        bok.CKPT_PATH.write_text(json.dumps(ck))
        print("\nAPPLIED. Checkpoint saved. Re-run v2_audit.py to confirm fewer suspect skips.")
    else:
        print("\nDRY RUN only. Re-run with --apply to write the recoverable deltas.")

if __name__ == "__main__":
    main()
