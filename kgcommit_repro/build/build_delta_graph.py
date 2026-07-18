"""
Step 2: Build delta graph edges in Neo4j.

For each commit-file pair:
1. Run GumTree parse on BEFORE and AFTER file → post-order node lists
2. Run GumTree jsondiff → matches + actions with integer node IDs
3. For each action, convert the relevant node's char offset → (line, col)
4. Match (file, pos_line, pos_col) to existing Neo4j ASTNode
5. Create REMOVES / UPDATES / MOVES / ADDS edges from Commit to ASTNode

GumTree uses post-order ID assignment: leaves first, root last.
Node IDs in jsondiff correspond exactly to the post-order traversal index
of the parse-tree JSON output.

Run:
    python build_delta_graph.py
"""

import subprocess, json, sys, os, tempfile
from pathlib import Path
from neo4j import GraphDatabase

# Legacy GumTree delta-graph prototype. In this package it is imported by
# build_online_kg.py only as a namespace (its full-AST helpers); the GumTree
# main()/loader below is NOT part of the reproduction pipeline. Constants come
# from config so nothing groovy-specific is baked in.
import _kgc_paths  # noqa: F401
from config.project_config import REPO_PATH, BASE_COMMIT, NEO4J_URI, NEO4J_AUTH
GT_LIB      = Path("tools/gumtree/gumtree-2.1.2/lib")
SAMPLE_N    = 120
GT_TIMEOUT  = 60   # seconds per GumTree call


# ── GumTree helpers ─────────────────────────────────────────────────────────

def _run_gt(*args, timeout=GT_TIMEOUT):
    lib = str(GT_LIB / "*")
    r = subprocess.run(
        ["java", "-cp", lib, "com.github.gumtreediff.client.Run", *args],
        capture_output=True, text=True, timeout=timeout
    )
    return r.stderr + r.stdout   # GumTree writes to stderr in 2.x

def _parse_json(raw):
    idx = raw.find("{")
    if idx == -1:
        return None
    try:
        return json.loads(raw[idx:])
    except json.JSONDecodeError:
        return None

def build_post_order(root):
    """Walk GumTree parse tree in post-order → list indexed by GumTree node ID."""
    nodes = []
    def walk(node):
        for child in node.get("children", []):
            walk(child)
        nodes.append(node)
    walk(root)
    return nodes

def offset_to_lc(text, offset):
    """0-indexed char offset → (line, col) 1-indexed, matching javalang convention."""
    before = text[:offset]
    lines  = before.split("\n")
    return len(lines), len(lines[-1]) + 1


# ── Neo4j helpers ────────────────────────────────────────────────────────────

# GumTree uses JavaParser names; javalang uses different names for some types.
# Lines match between both parsers; only columns differ (GumTree points to the
# start of modifiers, javalang points to the keyword/type token).
# Keys are GumTree typeLabels (from either the JavaParser or JDT generator),
# values are the matching javalang ast_type(s) used in Neo4j.
GUMTREE_TO_JL = {
    # JavaParser-style names
    "ClassOrInterfaceDeclaration": ["ClassDeclaration", "InterfaceDeclaration"],
    "ImportDeclaration":           ["Import"],
    "MethodDeclaration":           ["MethodDeclaration"],
    "ConstructorDeclaration":      ["ConstructorDeclaration"],
    "FieldDeclaration":            ["FieldDeclaration"],
    "Parameter":                   ["FormalParameter"],
    "BlockStmt":                   ["BlockStatement"],
    "IfStmt":                      ["IfStatement"],
    "ReturnStmt":                  ["ReturnStatement"],
    "ForStmt":                     ["ForStatement"],
    "WhileStmt":                   ["WhileStatement"],
    "DoStmt":                      ["DoStatement"],
    "TryStmt":                     ["TryStatement"],
    "ThrowStmt":                   ["ThrowStatement"],
    "ExpressionStmt":              ["StatementExpression"],
    "SwitchStmt":                  ["SwitchStatement"],
    "BreakStmt":                   ["BreakStatement"],
    "ContinueStmt":                ["ContinueStatement"],
    "VariableDeclarator":          ["VariableDeclarator"],
    "EnumDeclaration":             ["EnumDeclaration"],
    "AnnotationDeclaration":       ["AnnotationDeclaration"],
    # JDT-style names (used by the java-jdt fallback generator).
    # Many already coincide with javalang; map the ones that differ.
    "TypeDeclaration":             ["ClassDeclaration", "InterfaceDeclaration"],
    "SingleVariableDeclaration":   ["FormalParameter"],
    "Block":                       ["BlockStatement"],
    "VariableDeclarationStatement":["LocalVariableDeclaration"],
    "VariableDeclarationFragment": ["VariableDeclarator"],
    "ExpressionStatement":         ["StatementExpression"],
    "IfStatement":                 ["IfStatement"],
    "ReturnStatement":             ["ReturnStatement"],
    "ForStatement":                ["ForStatement"],
    "WhileStatement":              ["WhileStatement"],
    "DoStatement":                 ["DoStatement"],
    "TryStatement":                ["TryStatement"],
    "ThrowStatement":              ["ThrowStatement"],
    "SwitchStatement":             ["SwitchStatement"],
    "BreakStatement":              ["BreakStatement"],
    "ContinueStatement":           ["ContinueStatement"],
    "MethodInvocation":            ["MethodInvocation"],
}


def find_ast_node(session, file, line, col, gt_type=None, gt_label=None):
    """
    Match a GumTree node to a Neo4j ASTNode using a tiered strategy:
      1. Exact (line, col) — works when positions coincide
      2. Line + mapped javalang type — handles the GumTree col=modifier vs javalang col=type offset
      3. Line + label/value — for named nodes (methods, fields, variables)
      4. Any positioned node on the same line (last resort)
    Returns a record {id, ast_type, val} or None.
    """
    if line <= 0:
        return None

    def _q(where, **params):
        return session.run(
            f"MATCH (a:ASTNode {{file:$file}}) WHERE {where} "
            "RETURN a.id AS id, a.ast_type AS ast_type, a.value AS val LIMIT 1",
            file=file, **params
        ).single()

    # 1. Exact position
    if col > 0:
        rec = _q("a.pos_line=$line AND a.pos_col=$col", line=line, col=col)
        if rec:
            return rec

    # 2. Line + javalang type (covers GumTree col ≠ javalang col cases)
    if gt_type and gt_type in GUMTREE_TO_JL:
        for jl_type in GUMTREE_TO_JL[gt_type]:
            rec = _q("a.pos_line=$line AND a.ast_type=$atype",
                     line=line, atype=jl_type)
            if rec:
                return rec

    # 3. Line + label (method/class/field name as tie-breaker)
    if gt_label and gt_label.strip():
        rec = _q("a.pos_line=$line AND a.value=$val",
                 line=line, val=gt_label.strip())
        if rec:
            return rec

    # 4. Any positioned node on this line
    rec = _q("a.pos_line=$line AND a.pos_col > 0", line=line)
    return rec


# ── Core diff processing ─────────────────────────────────────────────────────

def process_commit_file(session, commit_id, file, before_bytes, after_bytes,
                        gen="java-javaparser"):
    """
    Runs GumTree diff and writes REMOVES/UPDATES/MOVES/ADDS edges to Neo4j.
    `gen` selects the GumTree generator; some files JavaParser 2.x rejects
    parse cleanly under the JDT generator (java-jdt) and vice-versa.
    Returns a stats dict, or None if GumTree fails (caller may retry with
    a different generator).
    """
    before_text = before_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
    after_text  = after_bytes.decode("utf-8",  errors="replace").replace("\r\n", "\n")

    # Write to temp files (no BOM — critical for both generators)
    with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
        f.write(before_text.encode("utf-8"))
        before_tmp = f.name
    with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
        f.write(after_text.encode("utf-8"))
        after_tmp = f.name

    try:
        # Parse both trees (same generator for parse + jsondiff → consistent IDs)
        src_json = _parse_json(_run_gt("parse", "-g", gen, before_tmp))
        dst_json = _parse_json(_run_gt("parse", "-g", gen, after_tmp))
        if not src_json or not dst_json:
            return None

        src_nodes = build_post_order(src_json["root"])   # indexed by src GumTree ID
        dst_nodes = build_post_order(dst_json["root"])   # indexed by dst GumTree ID

        # jsondiff: matches + actions
        jdiff = _parse_json(_run_gt("jsondiff", "-g", gen, before_tmp, after_tmp))
        if not jdiff:
            return None

        # DST_ID → SRC_ID (for finding the Neo4j parent of inserted nodes)
        dst_to_src = {m["dest"]: m["src"] for m in jdiff.get("matches", [])}

        def src_lc(node_id):
            if node_id >= len(src_nodes):
                return -1, -1
            return offset_to_lc(before_text, int(src_nodes[node_id]["pos"]))

        def dst_lc(node_id):
            if node_id >= len(dst_nodes):
                return -1, -1
            return offset_to_lc(after_text, int(dst_nodes[node_id]["pos"]))

        stats = dict(delete=0, update=0, insert=0, move=0,
                     existing_matched=0, existing_total=0,
                     insert_parented=0)

        # Track delta nodes created in THIS commit so inserted children can
        # attach to inserted parents (parent not in BASE/Neo4j).
        created_delta = {}   # dst_tree_id -> neo4j node id

        def src_node_type(nid):
            return src_nodes[nid]["typeLabel"] if nid < len(src_nodes) else None
        def src_node_label(nid):
            return src_nodes[nid].get("label","") if nid < len(src_nodes) else None

        for action in jdiff.get("actions", []):
            act     = action["action"]
            tree_id = action["tree"]

            # ── DELETE ──────────────────────────────────────────────────────
            if act == "delete":
                stats["delete"] += 1
                stats["existing_total"] += 1
                line, col = src_lc(tree_id)
                rec = find_ast_node(session, file, line, col,
                                    gt_type=src_node_type(tree_id),
                                    gt_label=src_node_label(tree_id))
                if rec:
                    stats["existing_matched"] += 1
                    session.run("""
                        MATCH (c:Commit {id:$cid}), (a:ASTNode {id:$aid})
                        MERGE (c)-[:REMOVES]->(a)
                    """, cid=commit_id, aid=rec["id"])

            # ── UPDATE ──────────────────────────────────────────────────────
            elif act == "update":
                stats["update"] += 1
                stats["existing_total"] += 1
                new_label = action.get("label", "")
                line, col = src_lc(tree_id)
                rec = find_ast_node(session, file, line, col,
                                    gt_type=src_node_type(tree_id),
                                    gt_label=src_node_label(tree_id))
                if rec:
                    stats["existing_matched"] += 1
                    session.run("""
                        MATCH (c:Commit {id:$cid}), (a:ASTNode {id:$aid})
                        MERGE (c)-[r:UPDATES]->(a)
                        ON CREATE SET r.old_value=$old, r.new_value=$new
                        ON MATCH  SET r.new_value=$new
                    """, cid=commit_id, aid=rec["id"],
                         old=rec["val"] or "", new=new_label)

            # ── MOVE ────────────────────────────────────────────────────────
            elif act == "move":
                stats["move"] += 1
                stats["existing_total"] += 1
                line, col = src_lc(tree_id)
                dst_parent_id  = action.get("parent", -1)
                new_parent_type = (dst_nodes[dst_parent_id]["typeLabel"]
                                   if 0 <= dst_parent_id < len(dst_nodes) else "")
                rec = find_ast_node(session, file, line, col,
                                    gt_type=src_node_type(tree_id),
                                    gt_label=src_node_label(tree_id))
                if rec:
                    stats["existing_matched"] += 1
                    session.run("""
                        MATCH (c:Commit {id:$cid}), (a:ASTNode {id:$aid})
                        MERGE (c)-[r:MOVES]->(a)
                        ON CREATE SET r.new_parent_type=$npt, r.new_pos=$np
                        ON MATCH  SET r.new_parent_type=$npt, r.new_pos=$np
                    """, cid=commit_id, aid=rec["id"],
                         npt=new_parent_type, np=action.get("at", -1))

            # ── INSERT ──────────────────────────────────────────────────────
            elif act == "insert":
                stats["insert"] += 1
                if tree_id >= len(dst_nodes):
                    continue

                dst_node = dst_nodes[tree_id]
                dst_line, dst_col = dst_lc(tree_id)

                # Resolve parent in Neo4j, in priority order:
                #   (a) parent is a delta node we already created this commit
                #   (b) parent is a matched (existing) node → map DST→SRC→Neo4j
                dst_parent_id   = action.get("parent", -1)
                parent_neo4j_id = None
                if dst_parent_id in created_delta:
                    parent_neo4j_id = created_delta[dst_parent_id]
                elif dst_parent_id in dst_to_src:
                    sp_id = dst_to_src[dst_parent_id]
                    p_line, p_col = src_lc(sp_id)
                    p_rec = find_ast_node(session, file, p_line, p_col,
                                          gt_type=src_node_type(sp_id),
                                          gt_label=src_node_label(sp_id))
                    if p_rec:
                        parent_neo4j_id = p_rec["id"]

                # Create new ASTNode for the inserted node
                new_id = f"{file}::DELTA::{commit_id[:8]}::{tree_id}"
                created_delta[tree_id] = new_id
                session.run("""
                    MERGE (a:ASTNode {id:$id})
                    ON CREATE SET
                      a.file      = $file,
                      a.ast_type  = $atype,
                      a.value     = $val,
                      a.pos_line  = $line,
                      a.pos_col   = $col,
                      a.is_delta  = true
                    WITH a
                    MATCH (c:Commit {id:$cid})
                    MERGE (c)-[:ADDS]->(a)
                """, id=new_id, file=file,
                     atype=dst_node["typeLabel"],
                     val=dst_node.get("label", ""),
                     line=dst_line, col=dst_col,
                     cid=commit_id)

                if parent_neo4j_id:
                    session.run("""
                        MATCH (p:ASTNode {id:$pid}), (ch:ASTNode {id:$chid})
                        MERGE (p)-[:AST_CHILD]->(ch)
                    """, pid=parent_neo4j_id, chid=new_id)
                    stats["insert_parented"] += 1

        return stats

    finally:
        os.unlink(before_tmp)
        os.unlink(after_tmp)


# ── Main ─────────────────────────────────────────────────────────────────────

def load_valid_commits(n):
    """Load pre-validated (commit, file) pairs where parent == BASE_COMMIT content."""
    import json
    pairs = json.loads(Path("outputs/valid_delta_commits.json").read_text())
    return [{"commit_id": c, "file": f} for c, f in pairs[:n]]


def main():
    import sys
    # Optional targeted mode: `python build_delta_graph.py --only NAME1,NAME2`
    # processes only files whose path contains any given substring and SKIPS
    # the cleanup, so it appends to the existing graph (e.g. to add files that
    # failed on a previous run). Without --only it does a full clean rebuild.
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with driver.session() as session:
        if only is None:
            # Full rebuild: clean any delta edges/nodes from previous runs.
            session.run("MATCH ()-[r:REMOVES|UPDATES|MOVES|ADDS]->() DELETE r")
            session.run("MATCH (a:ASTNode {is_delta:true}) DETACH DELETE a")

        samples = load_valid_commits(SAMPLE_N)
        if only:
            samples = [s for s in samples
                       if any(name in s["file"] for name in only)]
            print(f"TARGETED mode: {len(samples)} file(s) matching {only} "
                  f"(no cleanup, appending)\n")
        else:
            print(f"Processing {len(samples)} commit-file pairs...\n")

        grand = dict(delete=0, update=0, insert=0, move=0,
                     existing_matched=0, existing_total=0,
                     insert_parented=0)
        n_files = 0

        for s in samples:
            commit_id = s["commit_id"]
            file      = s["file"]

            r_before = subprocess.run(
                ["git", "-C", str(REPO_PATH), "show", f"{BASE_COMMIT}:{file}"],
                capture_output=True
            )
            r_after = subprocess.run(
                ["git", "-C", str(REPO_PATH), "show", f"{commit_id}:{file}"],
                capture_output=True
            )
            if r_before.returncode != 0:
                print(f"SKIP {commit_id[:8]}  {file}  (not in BASE)")
                continue

            # Whole-file deletion: AFTER content gone → REMOVES every ASTNode.
            if r_after.returncode != 0:
                n_removed = session.run("""
                    MATCH (c:Commit {id:$cid}), (a:ASTNode {file:$file})
                    WHERE a.is_delta IS NULL
                    MERGE (c)-[:REMOVES]->(a)
                    RETURN count(a) AS n
                """, cid=commit_id, file=file).single()["n"]
                print(f"[{commit_id[:8]}]  {file.split('/')[-1]}  "
                      f"FILE DELETED -> REMOVES {n_removed} nodes")
                grand["delete"] += n_removed
                grand["existing_total"] += n_removed
                grand["existing_matched"] += n_removed
                n_files += 1
                continue

            print(f"[{commit_id[:8]}]  {file.split('/')[-1]}")
            stats = None
            used_gen = "javaparser"
            try:
                stats = process_commit_file(
                    session, commit_id, file,
                    r_before.stdout, r_after.stdout,
                    gen="java-javaparser"
                )
            except subprocess.TimeoutExpired:
                print("  (JavaParser timeout)")
            except Exception as e:
                print(f"  (JavaParser error: {e})")

            # Fall back to the JDT generator when JavaParser can't parse the file.
            if stats is None:
                used_gen = "jdt"
                try:
                    stats = process_commit_file(
                        session, commit_id, file,
                        r_before.stdout, r_after.stdout,
                        gen="java-jdt"
                    )
                except subprocess.TimeoutExpired:
                    print("  SKIP (both generators timed out)")
                    continue
                except Exception as e:
                    print(f"  SKIP (JDT error: {e})")
                    continue

            if stats is None:
                print("  SKIP (both JavaParser and JDT failed to parse)")
                continue
            if used_gen == "jdt":
                print("  [used JDT fallback generator]")

            ex_t = stats["existing_total"]
            ex_m = stats["existing_matched"]
            ex_rate = (100 * ex_m // ex_t) if ex_t else 100
            ins_rate = (100 * stats["insert_parented"] // stats["insert"]) if stats["insert"] else 0
            print(f"  del={stats['delete']}  upd={stats['update']}  "
                  f"ins={stats['insert']}  mov={stats['move']}   "
                  f"existing-match={ex_m}/{ex_t} ({ex_rate}%)  "
                  f"ins-parented={stats['insert_parented']}/{stats['insert']} ({ins_rate}%)")

            n_files += 1
            for k in grand:
                grand[k] += stats[k]

        ex_t = grand["existing_total"]
        ex_m = grand["existing_matched"]
        ex_rate  = (100 * ex_m // ex_t) if ex_t else 100
        ins_rate = (100 * grand["insert_parented"] // grand["insert"]) if grand["insert"] else 0
        print(f"\n{'='*68}")
        print(f"Processed {n_files} files")
        print(f"TOTAL actions  del={grand['delete']} upd={grand['update']} "
              f"ins={grand['insert']} mov={grand['move']}")
        print(f"EXISTING-NODE MATCH (del/upd/move): {ex_m}/{ex_t}  ({ex_rate}%)")
        print(f"INSERT PARENT-ATTACH:               "
              f"{grand['insert_parented']}/{grand['insert']}  ({ins_rate}%)")
        print(f"{'='*68}")

    driver.close()


if __name__ == "__main__":
    main()
