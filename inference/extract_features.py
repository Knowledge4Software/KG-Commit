"""
Extract per-commit features from the KG for buggy/benign classification.

Two feature families:
  A. JIT change metrics  - already on the Commit nodes (classic ApacheJIT
                           baseline): la, ld, nf, nd, ns, ent, ndev, age, nuc,
                           aexp, arexp, asexp.
  B. Delta-graph features - the KG's contribution, from the commit's AST delta:
        n_adds, n_removes, n_updates, n_moves, delta_total,
        n_delta_files (distinct files touched at AST level),
        n_changed_types (distinct ast_type touched),
        grp_declaration/statement/expression/literal/leaf (group histogram of
        changed nodes), and the add/remove ratio.

Output: outputs/commit_features.csv  (one row per in_jit commit), with
        commit, author_ts, buggy, plus all features. Commits not yet grown by
        the online engine simply get zero delta features (robust to a partial
        build).

Run:  python inference/extract_features.py
"""
import csv
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase

NEO4J_URI="bolt://localhost:7687"; NEO4J_AUTH=("neo4j","password1234")
OUT=Path("outputs/commit_features.csv")
METRICS=["la","ld","nf","nd","ns","ent","ndev","age","nuc","aexp","arexp","asexp"]
GROUPS=["declaration","statement","expression","literal","leaf"]

def main():
    d=GraphDatabase.driver(NEO4J_URI,auth=NEO4J_AUTH)
    with d.session() as s:
        print("Pulling commit metrics...")
        base={}
        for r in s.run(f"""MATCH (c:Commit {{in_jit:true}})
            RETURN c.id AS id, c.author_ts AS ts, c.buggy AS buggy, c.is_fix AS is_fix,
                   {", ".join(f"c.{m} AS {m}" for m in METRICS)}""").data():
            base[r["id"]]={"commit":r["id"],"author_ts":r["ts"],
                           "buggy":int(bool(r["buggy"])),"is_fix":int(bool(r["is_fix"]))}
            for m in METRICS: base[r["id"]][m]=r[m] if r[m] is not None else 0
        print(f"  {len(base)} labelled commits")

        print("Pulling delta-graph features (this scans the delta layer)...")
        # per-commit, per-type, per-group counts of changed AST nodes
        agg=defaultdict(lambda: defaultdict(float))
        files=defaultdict(set); types=defaultdict(set)
        n=0
        for r in s.run("""MATCH (c:Commit {in_jit:true})-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:ASTNode)
            RETURN c.id AS id, type(r) AS et, a.group AS grp, a.file AS f,
                   a.ast_type AS t, count(*) AS n""").data():
            cid=r["id"]; et=r["et"]; n+=1
            agg[cid]["n_"+et.lower()]+=r["n"]
            agg[cid]["delta_total"]+=r["n"]
            if r["grp"] in GROUPS: agg[cid]["grp_"+r["grp"]]+=r["n"]
            files[cid].add(r["f"]); types[cid].add(r["t"])
        print(f"  aggregated {n} (commit,type,group,file) rows")

    d.close()

    # assemble rows
    delta_cols=["n_adds","n_removes","n_updates","n_moves","delta_total",
                "n_delta_files","n_changed_types","add_remove_ratio"]+\
               ["grp_"+g for g in GROUPS]
    rows=[]
    for cid,row in base.items():
        a=agg.get(cid,{})
        for c in ["n_adds","n_removes","n_updates","n_moves","delta_total"]+["grp_"+g for g in GROUPS]:
            row[c]=a.get(c,0.0)
        row["n_delta_files"]=len(files.get(cid,()))
        row["n_changed_types"]=len(types.get(cid,()))
        adds=a.get("n_adds",0.0); rem=a.get("n_removes",0.0)
        row["add_remove_ratio"]=adds/(rem+1.0)
        rows.append(row)
    rows.sort(key=lambda r:(r["author_ts"] or 0))

    cols=["commit","author_ts","buggy","is_fix"]+METRICS+delta_cols
    with open(OUT,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,0) for k in cols})
    with_delta=sum(1 for r in rows if r["delta_total"]>0)
    print(f"\nWrote {len(rows)} rows -> {OUT}")
    print(f"  commits with delta features: {with_delta} "
          f"({100*with_delta//max(1,len(rows))}%)  "
          f"[grows to ~100% once the online build finishes]")
    print(f"  buggy={sum(r['buggy'] for r in rows)}  benign={sum(1-r['buggy'] for r in rows)}")

if __name__=="__main__":
    main()
