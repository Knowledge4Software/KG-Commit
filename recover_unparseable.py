"""
V2 A3: best-effort recovery of the Java files that have no AST because the
build's parser (old `javalang`) could not parse them.

Diagnosis (this script): for each gap file, scan the labelled commits that
touched it and find the first version `javalang` can parse. If one exists, build
its full AST and attach it (so the file is no longer a structural gap). Files
with NO parseable labelled version are documented as a bounded known-gap --
they are modern-Java (antlr4 / Java-8 functional-interface) sources whose
recovery needs a newer parser; introducing one into the online pipeline would
break the deterministic `::A{i}` identity the differ relies on, so it is deferred.

  python recover_unparseable.py            # DRY RUN (diagnose only)
  python recover_unparseable.py --apply     # attach AST for parseable gap files
"""
import sys, json
import javalang
from pathlib import Path
from neo4j import GraphDatabase
import build_online_kg as bok

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")
APPLY = "--apply" in sys.argv

def main():
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    # ---- read phase: gap files + their labelled commits in ONE query ----
    with d.session(default_access_mode="READ") as s:
        rows = s.execute_read(lambda tx: [r.data() for r in tx.run("""
            MATCH (c:Commit {in_jit:true})-[:MODIFIED|ADDED]->(f:File)
            WHERE f.id ENDS WITH '.java' AND NOT (f)-[:HAS_AST]->()
            WITH f, c ORDER BY c.author_ts
            RETURN f.id AS f, collect(c.id) AS commits ORDER BY f.id""")])
    gaps = [r["f"] for r in rows]
    commits_by_file = {r["f"]: r["commits"] for r in rows}

    # ---- process: find first parseable version + build AST ----
    report = {"recovered": [], "residual": []}
    to_attach = []
    for f in gaps:
        first_ok = None
        for c in commits_by_file[f]:
            blob = bok.git_bytes(c, f)
            if blob is None: continue
            try:
                javalang.parse.parse(blob.decode("utf-8", "replace")); first_ok = c; break
            except Exception:
                pass
        short = "/".join(f.split("/")[-2:])
        ast = bok.build_full_ast(bok.git_bytes(first_ok, f), f) if first_ok else None
        if ast:
            to_attach.append((f, ast, first_ok))
            report["recovered"].append({"file": f, "at": first_ok[:8], "nodes": len(ast["nodes"])})
            print(f"  RECOVER {short}  ({len(ast['nodes'])} nodes @ {first_ok[:8]})")
        else:
            report["residual"].append({"file": f, "commits": len(commits_by_file[f]),
                                       "reason": "no javalang-parseable version (modern Java)"})
            print(f"  RESIDUAL {short}  ({len(commits_by_file[f])} commits) -- needs modern parser")

    # ---- write phase ----
    if APPLY and to_attach:
        with d.session() as sess:
            for f, ast, c in to_attach:
                bok.attach_full_ast(sess, f, ast, c, as_new_file=True)
    d.close()

    Path("outputs/v2_unparseable_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nrecovered {len(report['recovered'])} / {len(gaps)} gap files; "
          f"{len(report['residual'])} documented residual")
    print("report -> outputs/v2_unparseable_report.json")
    if not APPLY:
        print("DRY RUN. Re-run with --apply to attach the recoverable AST(s).")

if __name__ == "__main__":
    main()
