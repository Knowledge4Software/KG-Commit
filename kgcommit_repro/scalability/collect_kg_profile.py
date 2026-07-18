"""
E1 -- Final-KG statistical profile (read-only, no rebuild).

Profiles ONLY the final (V4) methodology graph as it exists in the built Neo4j DB:
  * Core relational/process layer (Commit / File / Developer / Issue + edges)
  * the structural layers of the final family (AST, CFG, DFG, PDG) and the
    Token-seq layer (reported for completeness though not in the final family)
  * the final CSTG semantic layer (Term / MENTIONS / COOCCURS / GROUNDS_IN and
    the change-intent typing Intent / HAS_INTENT)

For each layer we record graph-size totals (nodes, delta edges, alive/removed/
inserted), type/token vocabulary, the per-file size distribution, the commit
change-token degree distribution with a heavy-tail (power-law) fit, and the
on-disk store footprint. Everything is one pass of read-only Cypher against the
already-built DB; the legacy ASTDiff/ASTEdit layer is explicitly excluded.

Output: outputs/scalability/kg_profile.json

Run:  python scalability/collect_kg_profile.py
"""
import numpy as np

import _common as C


def _one(s, cy, **kw):
    return s.run(cy, **kw).single()[0]


def profile_core(s):
    """Core process/relational layer: node populations and edge multiplicities."""
    core = {}
    for lbl in ["Commit", "File", "Developer", "Issue", "Branch", "Project"]:
        core[f"n_{lbl}"] = _one(s, f"MATCH (a:{lbl}) RETURN count(a)")
    core["n_Commit_in_jit"] = _one(
        s, "MATCH (c:Commit {in_jit:true}) RETURN count(c)")
    core["n_Commit_buggy"] = _one(
        s, "MATCH (c:Commit {in_jit:true}) WHERE c.buggy RETURN count(c)")
    for rt in ["PARENT_OF", "AUTHORED_BY", "FIXES_ISSUE",
               "MODIFIED", "ADDED", "DELETED", "RENAMED_FROM", "RENAMED_TO"]:
        core[f"e_{rt}"] = _one(s, f"MATCH ()-[r:{rt}]->() RETURN count(r)")
    # files per commit and commits per file (process-layer degree distributions)
    fpc = [r[0] for r in s.run(
        "MATCH (c:Commit {in_jit:true})-[:MODIFIED|ADDED|DELETED]->(f:File) "
        "RETURN count(f) AS k")]
    core["files_per_commit"] = C.describe(fpc)
    return core


def profile_struct_layer(s, lbl, tp):
    """One structural layer (AST/CFG/DFG/PDG/SEQ): sizes, vocab, per-file sizes,
    and the commit change-token degree distribution with a power-law tail fit."""
    row = {"label": lbl}
    row["nodes"] = _one(s, f"MATCH (a:{lbl}) RETURN count(a)")
    row["alive"] = _one(
        s, f"MATCH (a:{lbl}) WHERE a.alive=true RETURN count(a)")
    row["removed"] = row["nodes"] - row["alive"]
    row["delta_inserted"] = _one(
        s, f"MATCH (a:{lbl}) WHERE a.is_delta RETURN count(a)")
    row["files"] = _one(s, f"MATCH (a:{lbl}) RETURN count(DISTINCT a.file)")
    for rel in C.DELTA_RELS:
        row[f"e_{rel}"] = _one(
            s, f"MATCH (:Commit)-[r:{rel}]->(:{lbl}) RETURN count(r)")
    row["delta_total"] = sum(row[f"e_{r}"] for r in C.DELTA_RELS)
    row["n_node_types"] = _one(
        s, f"MATCH (a:{lbl}) RETURN count(DISTINCT coalesce(a.{tp},'?'))")
    row["n_token_types"] = _one(
        s, f"MATCH (:Commit)-[r:ADDS|REMOVES|UPDATES|MOVES]->(a:{lbl}) "
           f"RETURN count(DISTINCT type(r)+':'+coalesce(a.{tp},'?'))")
    row["commits_with_tokens"] = _one(
        s, f"MATCH (c:Commit {{in_jit:true}})-[:ADDS|REMOVES|UPDATES|MOVES]->(:{lbl}) "
           f"RETURN count(DISTINCT c)")
    # per-file node-count distribution (materialised graph size, skew/Gini)
    per_file = [r[0] for r in s.run(
        f"MATCH (a:{lbl}) RETURN count(a) AS k")] if False else \
        [r["k"] for r in s.run(
            f"MATCH (a:{lbl}) WITH a.file AS f, count(a) AS k RETURN k")]
    row["per_file_nodes"] = C.describe(per_file)
    # commit change-token degree distribution (# delta edges emitted per commit)
    deg = [r["k"] for r in s.run(
        f"MATCH (c:Commit {{in_jit:true}})-[r:ADDS|REMOVES|UPDATES|MOVES]->(:{lbl}) "
        f"WITH c, count(r) AS k RETURN k")]
    row["delta_per_commit"] = C.describe(deg)
    row["delta_per_commit_powerlaw"] = C.powerlaw_tail(deg)
    return row


def profile_cstg(s):
    """The final CSTG semantic-text layer + change-intent typing."""
    cstg = {}
    cstg["n_Term"] = _one(s, "MATCH (t:Term) RETURN count(t)")
    cstg["n_Term_kinds"] = _one(
        s, "MATCH (t:Term) RETURN count(DISTINCT t.kind)")
    kinds = {r["k"]: r["n"] for r in s.run(
        "MATCH (t:Term) RETURN coalesce(t.kind,'?') AS k, count(*) AS n")}
    cstg["term_kind_hist"] = kinds
    cstg["n_Intent"] = _one(s, "MATCH (i:Intent) RETURN count(i)")
    for rt in ["MENTIONS", "COOCCURS", "GROUNDS_IN", "HAS_INTENT"]:
        cstg[f"e_{rt}"] = _one(s, f"MATCH ()-[r:{rt}]->() RETURN count(r)")
    # MENTIONS degree distributions: terms per commit, commits per term
    tpc = [r["k"] for r in s.run(
        "MATCH (c:Commit {in_jit:true})-[:MENTIONS]->(t:Term) "
        "WITH c, count(t) AS k RETURN k")]
    cstg["terms_per_commit"] = C.describe(tpc)
    cpt = [r["k"] for r in s.run(
        "MATCH (c:Commit)-[:MENTIONS]->(t:Term) "
        "WITH t, count(c) AS k RETURN k")]
    cstg["commits_per_term"] = C.describe(cpt)
    cstg["commits_per_term_powerlaw"] = C.powerlaw_tail(cpt)
    # COOCCURS term-graph degree (semantic connectivity)
    cod = [r["k"] for r in s.run(
        "MATCH (t:Term)-[:COOCCURS]-(:Term) WITH t, count(*) AS k RETURN k")]
    cstg["cooccurs_degree"] = C.describe(cod)
    return cstg


def store_footprint(s):
    """On-disk store size from Neo4j's jmx/sysinfo, best-effort (read-only)."""
    out = {}
    try:
        for r in s.run("CALL dbms.database.state('neo4j')"):
            pass  # not universally available; ignored below
    except Exception:
        pass
    try:
        rec = s.run(
            "CALL apoc.monitor.store() YIELD totalStoreSize, nodeStoreSize, "
            "relStoreSize, propStoreSize RETURN totalStoreSize, nodeStoreSize, "
            "relStoreSize, propStoreSize").single()
        if rec:
            out = dict(rec)
    except Exception:
        out = {"note": "store size unavailable (apoc.monitor.store not present); "
                       "see docs for the ~3.3 GB reported footprint"}
    return out


def main():
    ok, status = C.assert_db_complete()
    print(f"DB complete (next_index==target): {ok}  {status}")
    d = C.driver()
    prof = {"_meta": {"scope": "final V4 methodology only; legacy ASTDiff/ASTEdit "
                              "excluded", "db_complete": ok, "checkpoints": status}}
    with C.read_session(d) as s:
        print("core layer ...")
        prof["core"] = profile_core(s)
        for vid, pretty, lbl, tp in C.STRUCT_LAYERS:
            print(f"structural layer {pretty} ...")
            r = profile_struct_layer(s, lbl, tp)
            r["pretty"] = pretty
            r["in_final_family"] = vid in C.FINAL_GRAPHS  # seq -> False
            prof[vid] = r
            print(f"  {pretty:<10} nodes={r['nodes']:>9,} delta={r['delta_total']:>9,} "
                  f"tok_types={r['n_token_types']:>4} "
                  f"deg_p50={r['delta_per_commit']['p50']:.0f} "
                  f"pl_alpha={r['delta_per_commit_powerlaw'].get('alpha', float('nan')):.2f}")
        print("cstg layer ...")
        prof["cstg"] = profile_cstg(s)
        print(f"  Term={prof['cstg']['n_Term']:,} MENTIONS={prof['cstg']['e_MENTIONS']:,} "
              f"COOCCURS={prof['cstg']['e_COOCCURS']:,} "
              f"GROUNDS_IN={prof['cstg']['e_GROUNDS_IN']:,} "
              f"Intent={prof['cstg']['n_Intent']} HAS_INTENT={prof['cstg']['e_HAS_INTENT']:,}")
        prof["store"] = store_footprint(s)
    d.close()
    p = C.save_json(prof, "kg_profile.json")
    print(f"\nsaved -> {p}")

    # sanity anchor against the cached layer stats (outputs/<project>/subgraph_layer_stats.json)
    try:
        import json
        old = json.load(open(C.OUT.parent / "subgraph_layer_stats.json"))
        a_old = old["ast"]["nodes"]; a_new = prof["ast"]["nodes"]
        print(f"anchor: AST nodes cached={a_old:,} vs live={a_new:,} "
              f"({'MATCH' if a_old == a_new else 'DIFFER'})")
    except Exception as e:
        print("anchor check skipped:", e)


if __name__ == "__main__":
    main()
