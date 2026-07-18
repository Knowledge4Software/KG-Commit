"""
Apply ApacheJIT bug-labels + change metrics to Commit nodes in Neo4j.

Source: data/apachejit/projects/apache_groovy.csv  (8,059 labelled commits)
Sets on each matching (:Commit {id}) node:
  buggy      : bool   - the JIT defect label (introduced a bug later fixed)
  is_fix     : bool   - this commit is itself a bug-fix
  jit_year   : int
  author_ts  : int    - author UNIX timestamp (for chronological online order)
  la, ld     : int    - lines added / deleted
  nf, nd, ns : int    - files / dirs / subsystems touched
  ent        : float  - change entropy
  ndev, age, nuc, aexp, arexp, asexp : float  - JIT process metrics
  in_jit     : bool   - marks the commit as part of the labelled study set

Commits NOT in the CSV are left untouched (in_jit unset).

Run:  python apply_commit_labels.py
"""

import csv
from pathlib import Path
from neo4j import GraphDatabase

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import CSV_PATH, NEO4J_URI, NEO4J_AUTH  # per-project

INT_COLS   = ["la", "ld", "nf", "nd", "ns"]
FLOAT_COLS = ["ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]


def load_rows():
    rows = []
    for r in csv.DictReader(open(CSV_PATH)):
        rec = {
            "id":        r["commit_id"],
            "buggy":     r["buggy"] == "True",
            "is_fix":    r["fix"]   == "True",
            "jit_year":  int(r["year"]),
            "author_ts": int(r["author_date"]),
        }
        for c in INT_COLS:
            rec[c] = int(float(r[c]))
        for c in FLOAT_COLS:
            rec[c] = float(r[c])
        rows.append(rec)
    return rows


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} labelled commits from CSV.")
    buggy = sum(1 for r in rows if r["buggy"])
    print(f"  buggy={buggy}  benign={len(rows)-buggy}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as s:
        # index for fast online chronological queries
        s.run("CREATE INDEX commit_author_ts IF NOT EXISTS "
              "FOR (c:Commit) ON (c.author_ts)")

        # Bulk update in batches
        BATCH = 1000
        matched = 0
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i+BATCH]
            res = s.run("""
                UNWIND $rows AS r
                MATCH (c:Commit {id: r.id})
                SET c.buggy=r.buggy, c.is_fix=r.is_fix, c.jit_year=r.jit_year,
                    c.author_ts=r.author_ts, c.la=r.la, c.ld=r.ld, c.nf=r.nf,
                    c.nd=r.nd, c.ns=r.ns, c.ent=r.ent, c.ndev=r.ndev,
                    c.age=r.age, c.nuc=r.nuc, c.aexp=r.aexp, c.arexp=r.arexp,
                    c.asexp=r.asexp, c.in_jit=true
                RETURN count(c) AS n
            """, rows=chunk).single()["n"]
            matched += res
            print(f"  batch {i//BATCH+1}: matched {res}/{len(chunk)}")

        print(f"\nLabelled {matched}/{len(rows)} commits in Neo4j.")

        # Verify
        v = s.run("""
            MATCH (c:Commit {in_jit:true})
            RETURN count(c) AS total,
                   sum(CASE WHEN c.buggy THEN 1 ELSE 0 END) AS buggy,
                   sum(CASE WHEN c.is_fix THEN 1 ELSE 0 END) AS fixes
        """).single()
        print(f"Verify in Neo4j: total={v['total']}  buggy={v['buggy']}  "
              f"fixes={v['fixes']}")
    driver.close()


if __name__ == "__main__":
    main()
