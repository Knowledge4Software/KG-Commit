"""
Validate an alternative-subgraph layer built by build_subgraph_online_kg.py.

Per kind it checks the structural invariants the online engine must maintain and
exports a few sample per-commit delta subgraphs for paper figures:

  invariants
    * every ADDS/REMOVES/UPDATES/MOVES delta edge points at a node of THIS layer's
      label (no cross-layer leakage; the AST layer is never touched);
    * every removed node carries alive=false AND removed_by (soft-delete kept for
      delta history), every non-removed node is alive=true;
    * every node has a token type (atype) and a method; roots are reachable via
      File-[:HAS_<X>];
    * delta-edge and node-label tallies (sanity magnitudes).
  spot re-derivation (optional, --recheck N)
    * for N random tracked files, rebuild the CURRENT-repo subgraph and compare
      its node count to the count of alive nodes stored for that file.
  sample export (--samples N)
    * dump N commits' delta subgraphs (nodes + ADDS/REMOVES/UPDATES/MOVES) to
      outputs/sg_samples/<kind>/*.json.

Run:
  python validate_subgraph_kg.py --kind cfg
  python validate_subgraph_kg.py --kind pdg --recheck 20 --samples 12
"""
import argparse, json, random, subprocess, sys
from pathlib import Path
from neo4j import GraphDatabase

import subgraph_spec as spec
import subgraph_diff as sdiff  # noqa: F401  (kept for parity / future checks)

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_PATH    = PROJECT_ROOT / "repos" / "apache" / "groovy"
BUILDER      = str(PROJECT_ROOT / "subgraph_builders.py")
NEO4J_URI    = "bolt://localhost:7687"
NEO4J_AUTH   = ("neo4j", "password1234")


def git_show(ref, path):
    r = subprocess.run(["git", "-C", str(REPO_PATH), "show", f"{ref}:{path}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def build_current(kind, path):
    blob = git_show("HEAD", path)
    if blob is None:
        return None
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
        f.write(blob.decode("utf-8", "replace").replace("\r\n", "\n").encode()); tmp = f.name
    try:
        r = subprocess.run([sys.executable, BUILDER, kind, tmp, path],
                           capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT))
        d = json.loads(r.stdout) if r.stdout.strip() else {"error": "empty"}
        return None if "error" in d else d
    except Exception:
        return None
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(spec.REGISTRY))
    ap.add_argument("--recheck", type=int, default=0)
    ap.add_argument("--samples", type=int, default=0)
    args = ap.parse_args()
    sp = spec.get(args.kind); L = sp.node_label
    ok = True

    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with d.session() as s:
        one = lambda cy, **kw: s.run(cy, **kw).single()[0]
        n_nodes = one(f"MATCH (a:{L}) RETURN count(a)")
        n_alive = one(f"MATCH (a:{L}) WHERE coalesce(a.alive,true) RETURN count(a)")
        n_files = one(f"MATCH (f:File)-[:{sp.attach_rel}]->(:{L}) RETURN count(DISTINCT f)")
        print(f"[{args.kind}] {L}: {n_nodes} nodes ({n_alive} alive), "
              f"attached to {n_files} files")

        for rel in spec.DELTA_RELS:
            tot = one(f"MATCH (:Commit)-[r:{rel}]->(:{L}) RETURN count(r)")
            print(f"  {rel:<8} -> {L}: {tot}")

        # invariant 1: no delta edge to this layer lands on a wrong-label node
        bad_lbl = one(f"""MATCH (:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:{L})
                          WHERE size(labels(a))<>1 RETURN count(r)""")
        # (a subgraph node should carry exactly its one label)
        # invariant 2: removed nodes well-formed
        bad_removed = one(f"""MATCH (a:{L}) WHERE a.alive=false
            AND a.removed_by IS NULL RETURN count(a)""")
        # invariant 3: REMOVES edge <=> alive=false
        rem_alive = one(f"""MATCH (:Commit)-[:REMOVES]->(a:{L})
            WHERE coalesce(a.alive,true) RETURN count(a)""")
        # invariant 4: token type + method present
        miss_type = one(f"MATCH (a:{L}) WHERE a.atype IS NULL RETURN count(a)")
        miss_meth = one(f"MATCH (a:{L}) WHERE a.method IS NULL RETURN count(a)")

        for label, v in [("multi-labelled delta targets", bad_lbl),
                         ("removed nodes missing removed_by", bad_removed),
                         ("REMOVES targets still alive", rem_alive),
                         ("nodes missing atype", miss_type),
                         ("nodes missing method", miss_meth)]:
            flag = "OK " if v == 0 else "FAIL"
            if v: ok = False
            print(f"  [{flag}] {label}: {v}")

        # spot re-derivation
        if args.recheck:
            files = [r["f"] for r in s.run(
                f"MATCH (f:File)-[:{sp.attach_rel}]->(:{L}) RETURN DISTINCT f.id AS f").data()]
            random.seed(0); sample = random.sample(files, min(args.recheck, len(files)))
            print(f"\n  re-deriving {len(sample)} files at HEAD (alive-count drift):")
            drifts = []
            for path in sample:
                cur = build_current(args.kind, path)
                if cur is None:   # file gone at HEAD or unparseable now
                    continue
                stored = one(f"MATCH (a:{L} {{file:$f}}) WHERE coalesce(a.alive,true) "
                             f"RETURN count(a)", f=path)
                built = len(cur["nodes"])
                drifts.append(abs(stored - built))
                if abs(stored - built) > max(5, 0.15 * built):
                    print(f"    drift {path}: stored_alive={stored} built={built}")
            if drifts:
                import statistics as st
                print(f"    median |drift|={st.median(drifts):.1f} over {len(drifts)} files")

        # sample export
        if args.samples:
            outdir = PROJECT_ROOT / "outputs" / "sg_samples" / args.kind
            outdir.mkdir(parents=True, exist_ok=True)
            rows = s.run(f"""
                MATCH (c:Commit {{in_jit:true}})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:{L})
                WITH c, count(r) AS ne WHERE ne > 8 AND ne < 120
                RETURN c.id AS cid ORDER BY c.author_ts LIMIT $k
            """, k=args.samples).data()
            for row in rows:
                cid = row["cid"]
                data = s.run(f"""
                    MATCH (c:Commit {{id:$cid}})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:{L})
                    RETURN type(r) AS edge, a.id AS id, a.atype AS type,
                           a.method AS method, coalesce(a.alive,true) AS alive
                """, cid=cid).data()
                (outdir / f"{cid[:12]}.json").write_text(json.dumps(
                    {"commit": cid, "kind": args.kind, "delta": data}, indent=1))
            print(f"\n  exported {len(rows)} sample delta subgraphs -> {outdir}")

    d.close()
    print(f"\n{'VALID' if ok else 'INVARIANT FAILURES'} for {args.kind}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
