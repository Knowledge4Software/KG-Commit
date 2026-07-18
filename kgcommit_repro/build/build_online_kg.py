"""
Online KG growth engine.

Processes the 8,059 ApacheJIT-labelled commits in chronological (author_ts)
order. For each commit and each .java file it changed:

  ADDED file                 -> build full AST subgraph (HAS_AST), like base.
  MODIFIED file we DO track  -> GumTree-diff its evolving state vs the commit
                                blob; write delta edges (ADDS/REMOVES/UPDATES/
                                MOVES); evolve the stored AST + re-stamp
                                positions so the next commit diffs correctly.
  MODIFIED file we DON'T track yet  -> LAZY BOOTSTRAP: build its full AST from
                                the parent blob first, then diff as above.
  DELETED file               -> REMOVES every alive node of the file.

Node identity across versions comes from GumTree's match array (mapped to graph
nodes by line/col), NOT from raw positions — so edits that shift line numbers
don't break matching. After each commit we re-stamp positions from the new
parse, keeping the invariant: alive node positions == file's current state.

Removed nodes are kept but flagged (alive=false, removed_by) so the per-commit
delta history is preserved while the live tree reflects the current state.

State (per-file current commit) is checkpointed; the run is resumable and
supports a --limit window for testing.

Run:
  python build_online_kg.py --limit 30          # test window
  python build_online_kg.py                      # full run (long; resumable)
  python build_online_kg.py --reset              # wipe online layer + restart
"""

import argparse, json, subprocess, sys, tempfile, os, time, csv
from pathlib import Path
from neo4j import GraphDatabase

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (REPO_PATH, REPO_PATHS, CSV_PATH, CKPT_PATH,
                                    NEO4J_URI, NEO4J_AUTH, base_commit,
                                    git_root_for)

# GumTree machinery (kept for full-AST bootstrap helpers); javalang-native
# differ for the MODIFIED path (exact node identity, no positional drift).
import build_delta_graph as bdg
import online_ast_diff as oad

BASE_COMMIT  = base_commit()             # loud check: online growth seeds from the base
GENERATORS   = ["java-javaparser", "java-jdt"]
PROJECT_ROOT = Path(__file__).resolve().parent   # package build/ dir (AST-worker cwd)


# ── git helpers ──────────────────────────────────────────────────────────────

def git_bytes(ref, path):
    """Return the bytes of <ref>:<path>, trying each repo in REPO_PATHS.

    Multi-repo aware: a file's previous blob (file_state[path]) and its current
    blob can live in different repositories at a spin-off/monorepo boundary. We
    probe the repo that owns `ref` first (fast, cached), then fall back to the
    full list. Single-repo projects hit REPO_PATH immediately. See
    docs/Critical_notes.tex."""
    root = git_root_for(ref.split(":")[0].split("^")[0]) if len(REPO_PATHS) > 1 else REPO_PATH
    tries = []
    if root is not None:
        tries.append(root)
    tries += [p for p in REPO_PATHS if p != root]
    for _root in tries:
        r = subprocess.run(["git", "-C", str(_root), "show", f"{ref}:{path}"],
                           capture_output=True)
        if r.returncode == 0:
            return r.stdout
    return None

def _commit_repo(commit):
    """Repo (Path) that owns `commit`; REPO_PATH for single-repo projects."""
    if len(REPO_PATHS) == 1:
        return REPO_PATH
    return git_root_for(commit) or REPO_PATH

def changed_java_files(commit):
    """Return [(status, path)] for .java files changed by commit vs its first parent."""
    repo = _commit_repo(commit)
    # Detect whether the commit has a parent.
    par = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{commit}^1"],
                         capture_output=True, text=True)
    if par.returncode != 0:
        # Root commit: everything is added.
        r = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r",
                            "--name-only", commit], capture_output=True, text=True)
        return [("A", p) for p in r.stdout.splitlines() if p.endswith(".java")]
    r = subprocess.run(["git", "-C", str(repo), "diff", "--name-status",
                        "-M", f"{commit}^1", commit],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        st = parts[0]
        if st.startswith("R"):                       # rename: retire old path, add new
            oldp, newp = parts[1], parts[2]
            # retire the old path's AST (so it is not left as an alive "ghost"),
            # then bring in the new path. Without the explicit ("D", oldp) the old
            # subtree would stay alive=true forever (the V1 rename bug).
            if oldp.endswith(".java"):
                out.append(("D", oldp))
            if newp.endswith(".java"):
                out.append(("A", newp))
        elif st in ("A", "M", "D") and parts[1].endswith(".java"):
            out.append((st, parts[1]))
    return out


# ── subprocess worker: build a full AST graph + positions for one source file ─
# Emits JSON {nodes:[{id,ast_type,group,value,is_leaf,depth,line,col}],
#             edges:[{parent,child,pos}], root}.  Node ids are "{rel}::A{i}",
# identical to the base-snapshot scheme, so everything is uniform.
_AST_WORKER = r"""
import sys, json
import javalang, javalang.tree as jlt
import networkx as nx
import build_ast_features as bf

def clean(v):
    # strip lone UTF-16 surrogates that Neo4j's driver cannot encode to UTF-8
    if isinstance(v, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in v):
        return ''.join(c for c in v if not (0xD800 <= ord(c) <= 0xDFFF))
    return v

src_path, rel = sys.argv[1], sys.argv[2]
src = open(src_path, encoding='utf-8', errors='replace').read()
try:
    tree = javalang.parse.parse(src)
except Exception as e:
    print(json.dumps({"error": str(e)})); sys.exit(0)

Ag, root = bf._build_ast_subgraph(tree, rel, jlt, nx)
depths = nx.shortest_path_length(Ag, source=root)

# positions: mirror the EXACT DFS of _build_ast_subgraph (same counter order)
positions = []
stack = [tree]
# _build_ast_subgraph assigns ids by a pre-order DFS over .children; reproduce it
def walk(node):
    pos = getattr(node, 'position', None)
    positions.append([int(pos.line), int(pos.column)] if pos else [-1, -1])
    for child in node.children:
        if child is None: continue
        items = child if isinstance(child, (list, tuple, set, frozenset)) else [child]
        for it in items:
            if isinstance(it, jlt.Node):
                walk(it)
            elif isinstance(it, str) and it.strip():
                positions.append([-1, -1])   # Identifier leaf
walk(tree)

nodes = []
for nid, d in Ag.nodes(data=True):
    i = int(nid.rsplit('::A', 1)[1])
    line, col = (positions[i] if i < len(positions) else [-1, -1])
    nodes.append({"id": nid, "ast_type": d.get("ast_type"), "group": d.get("group"),
                  "value": clean(d.get("value")), "is_leaf": bool(d.get("is_leaf")),
                  "depth": int(depths.get(nid, 0)), "line": line, "col": col})
edges = [{"parent": u, "child": v, "pos": d.get("pos")} for u, v, d in Ag.edges(data=True)]
print(json.dumps({"nodes": nodes, "edges": edges, "root": root}))
"""

def build_full_ast(src_bytes, rel_path, timeout=60):
    """Run the AST worker on source bytes; return dict or None."""
    text = src_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
    with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
        f.write(text.encode("utf-8")); tmp = f.name
    try:
        r = subprocess.run([sys.executable, "-c", _AST_WORKER, tmp, rel_path],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(PROJECT_ROOT))
        if r.returncode == 0 and r.stdout.strip():
            d = json.loads(r.stdout)
            return d if "error" not in d else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    finally:
        os.unlink(tmp)
    return None


# ── Neo4j writes ─────────────────────────────────────────────────────────────

def attach_full_ast(session, file_id, ast, commit_id, as_new_file):
    """MERGE a file's full AST into the graph (base-style nodes, alive=true)."""
    # MERGE (not CREATE): base-style ids {file}::A{i} are deterministic PER FILE (not
    # per commit), so a re-bootstrapped file (deleted then re-added, or bootstrapped
    # via two paths) legitimately re-touches the same id. MERGE is idempotent there;
    # CREATE would violate the ASTNode.id uniqueness constraint. (Only the delta
    # inserts in process_modify have commit-stamped unique ids safe for CREATE.)
    session.run("""
        MERGE (f:File {id:$file})
        WITH f
        UNWIND $nodes AS n
        MERGE (a:ASTNode {id:n.id})
        ON CREATE SET a.file=$file, a.ast_type=n.ast_type, a.group=n.group,
                      a.value=n.value, a.is_leaf=n.is_leaf, a.depth=n.depth,
                      a.pos_line=n.line, a.pos_col=n.col, a.alive=true
    """, file=file_id, nodes=ast["nodes"])
    if ast["edges"]:
        session.run("""
            UNWIND $edges AS e
            MATCH (p:ASTNode {id:e.parent}), (c:ASTNode {id:e.child})
            MERGE (p)-[r:AST_CHILD]->(c) ON CREATE SET r.pos=e.pos
        """, edges=ast["edges"])
    session.run("""
        MATCH (f:File {id:$file}), (r:ASTNode {id:$root}) MERGE (f)-[:HAS_AST]->(r)
    """, file=file_id, root=ast["root"])
    if as_new_file:
        # mark the whole subtree as added by this commit (MERGE: re-bootstrap-safe)
        session.run("""
            MATCH (c:Commit {id:$cid})
            UNWIND $ids AS nid
            MATCH (a:ASTNode {id:nid})
            MERGE (c)-[:ADDS]->(a)
        """, cid=commit_id, ids=[n["id"] for n in ast["nodes"]])


def load_alive_nodes(session, file_id):
    """All alive nodes of a file -> dict for in-memory matching."""
    rows = session.run("""
        MATCH (a:ASTNode {file:$file})
        WHERE a.alive = true
        RETURN a.id AS id, a.ast_type AS t, a.value AS v,
               a.pos_line AS line, a.pos_col AS col
    """, file=file_id).data()
    by_pos = {}      # (line,col) -> list of rows
    for r in rows:
        by_pos.setdefault((r["line"], r["col"]), []).append(r)
    return rows, by_pos


# ── core: process a MODIFIED tracked file ────────────────────────────────────

def process_modify(session, commit_id, file_id, before_bytes, after_bytes, node_map):
    """
    javalang-native diff (online_ast_diff): both versions parsed with the SAME
    scheme as the graph, so node identity is EXACT.

    node_map: {before_parse_lid -> graph_node_id} for the file's current state.
              Missing keys default to identity (base files: lid IS the graph id).
    Returns (stats, new_map) or None on parse failure. new_map is the
    {after_parse_lid -> graph_node_id} to persist as the file's next map.
    """
    d = oad.diff_sources(before_bytes, after_bytes, file_id)
    if d is None:
        return None

    def gid_of(before_lid):                       # graph id for a before node
        return node_map.get(before_lid, before_lid)

    after_by_lid = {n["lid"]: n for n in d["after_nodes"]}
    st = dict(adds=0, removes=0, updates=0, moves=0, matched=0, total=0)
    new_map = {}                                  # after_lid -> graph id

    # ---- matched: keep graph id, restamp position, value-update if changed ----
    restamp, updates = [], []
    for before_lid, after_lid, vchanged, newval in d["matched"]:
        g = gid_of(before_lid)
        new_map[after_lid] = g
        an = after_by_lid[after_lid]
        restamp.append({"id": g, "line": an["line"], "col": an["col"]})
        st["matched"] += 1
        if vchanged:
            updates.append({"id": g, "new": newval})
            st["updates"] += 1; st["total"] += 1

    # ---- deleted: mark removed + REMOVES edge ----
    removes = [gid_of(b) for b in d["deleted"]]
    st["removes"] = len(removes)
    st["total"] += len(removes)

    # ---- inserted: new graph nodes, wired to parent (matched or new) ----
    inserts = []
    for alid in d["inserted"]:
        an = after_by_lid[alid]
        gnew = f"{file_id}::D{commit_id[:8]}:{alid.rsplit('::A',1)[-1]}"
        new_map[alid] = gnew
        inserts.append({"id": gnew, "atype": an["ast_type"], "group": an["group"],
                        "val": an["value"], "is_leaf": an["is_leaf"],
                        "line": an["line"], "col": an["col"], "depth": an["depth"],
                        "after_lid": alid, "parent_lid": an["parent_lid"]})
    st["adds"] = len(inserts)
    # resolve insert parents now that new_map has all after nodes
    for row in inserts:
        row["parent"] = new_map.get(row["parent_lid"])

    # ---- moves ----
    moves = [{"id": new_map[alid], "npt": npt} for alid, npt in d["moved"]
             if alid in new_map]
    st["moves"] = len(moves)

    # ── apply to Neo4j ──
    if removes:
        session.run("""
            MATCH (c:Commit {id:$cid}) UNWIND $ids AS gid
            MATCH (a:ASTNode {id:gid})
            SET a.alive=false, a.removed_by=$cid
            MERGE (c)-[:REMOVES]->(a)
        """, cid=commit_id, ids=removes)
    if updates:
        session.run("""
            MATCH (c:Commit {id:$cid}) UNWIND $rows AS u
            MATCH (a:ASTNode {id:u.id})
            MERGE (c)-[r:UPDATES]->(a) ON CREATE SET r.old_value=a.value, r.new_value=u.new
            SET a.value=u.new
        """, cid=commit_id, rows=updates)
    if inserts:
        session.run("""
            MATCH (c:Commit {id:$cid}) UNWIND $rows AS n
            MERGE (a:ASTNode {id:n.id})
              ON CREATE SET a.file=$file, a.ast_type=n.atype, a.group=n.group,
                            a.value=n.val, a.is_leaf=n.is_leaf, a.depth=n.depth,
                            a.pos_line=n.line, a.pos_col=n.col,
                            a.is_delta=true, a.alive=true
            MERGE (c)-[:ADDS]->(a)
        """, cid=commit_id, file=file_id, rows=inserts)
        session.run("""
            UNWIND $rows AS n WITH n WHERE n.parent IS NOT NULL
            MATCH (p:ASTNode {id:n.parent}), (ch:ASTNode {id:n.id})
            MERGE (p)-[:AST_CHILD]->(ch)
        """, rows=inserts)
    if moves:
        session.run("""
            MATCH (c:Commit {id:$cid}) UNWIND $rows AS m
            MATCH (a:ASTNode {id:m.id})
            MERGE (c)-[r:MOVES]->(a) ON CREATE SET r.new_parent_type=m.npt
        """, cid=commit_id, rows=moves)
    if restamp:
        session.run("""
            UNWIND $rows AS r MATCH (a:ASTNode {id:r.id})
            SET a.pos_line=r.line, a.pos_col=r.col
        """, rows=restamp)
    return st, new_map


def delete_file(session, commit_id, file_id):
    n = session.run("""
        MATCH (c:Commit {id:$cid}), (a:ASTNode {file:$file})
        WHERE a.alive = true
        SET a.alive=false, a.removed_by=$cid
        MERGE (c)-[:REMOVES]->(a)
        RETURN count(a) AS n
    """, cid=commit_id, file=file_id).single()["n"]
    return n


# ── checkpoint ───────────────────────────────────────────────────────────────

def load_ckpt():
    if CKPT_PATH.exists():
        ck = json.loads(CKPT_PATH.read_text())
        ck.setdefault("file_maps", {})
        return ck
    # init: 120 base files reflect BASE_COMMIT; file_maps default to identity
    return {"next_index": 0, "file_state": {}, "file_maps": {}}

def save_ckpt(ck):
    CKPT_PATH.write_text(json.dumps(ck))


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N commits this run")
    ap.add_argument("--reset", action="store_true", help="wipe online layer + checkpoint, restart")
    ap.add_argument("--timing-log", default=None,
                    help="optional CSV path; append one row per commit "
                         "(idx, sha, files, delta counts, wall_ms) for the "
                         "scalability/complexity analysis. Off by default so the "
                         "normal build is unchanged. Useful when building the KG "
                         "for a NEW project to capture per-commit wall-clock.")
    args = ap.parse_args()

    tlog = None
    if args.timing_log:
        _new = not Path(args.timing_log).exists()
        tlog = open(args.timing_log, "a", newline="", encoding="utf-8")
        _tw = csv.writer(tlog)
        if _new:
            _tw.writerow(["idx", "sha", "A", "M", "D", "boot",
                          "adds", "removes", "updates", "moves", "matched",
                          "wall_ms"])

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as s:
        # ── performance: index the hot per-file liveness lookup ───────────────
        # load_current_ast / delete_file match (a:ASTNode {file:$file}) and filter
        # a.alive. Without a composite (file, alive) index this scans ALL of a
        # file's historical (mostly dead) nodes and gets slower as the graph
        # grows -- the "missing-index cliff" that stalled the hadoop-hdfs build
        # at ~5M nodes (see docs/Critical_notes.tex). Created idempotently so
        # every project (esp. large ones: camel/ignite/flink) benefits.
        s.run("CREATE INDEX astnode_file_alive IF NOT EXISTS "
              "FOR (a:ASTNode) ON (a.file, a.alive)")
        # keep the index fully covering: never leave alive=null (base nodes from
        # A2 may lack it). Backfill null->true so an alive=true seek finds them.
        s.run("MATCH (a:ASTNode) WHERE a.alive IS NULL SET a.alive=true")

        if args.reset:
            print("Resetting online layer...")
            s.run("MATCH ()-[r:ADDS|REMOVES|UPDATES|MOVES]->() DELETE r")
            s.run("MATCH (a:ASTNode {is_delta:true}) DETACH DELETE a")
            # remove full ASTs of files born/bootstrapped in a prior online run
            # (keep only the 120 base files added by BASE_COMMIT)
            s.run("""
                MATCH (f:File)-[:HAS_AST]->()
                WHERE NOT EXISTS { MATCH (:Commit {id:$base})-[:ADDED]->(f) }
                WITH f MATCH (a:ASTNode {file:f.id}) DETACH DELETE a
            """, base=BASE_COMMIT)
            s.run("MATCH (a:ASTNode) REMOVE a.alive, a.removed_by")
            if CKPT_PATH.exists(): CKPT_PATH.unlink()

        # chronological commit order
        commits = s.run("""
            MATCH (c:Commit {in_jit:true})
            RETURN c.id AS id, c.author_ts AS ts, c.buggy AS buggy
            ORDER BY c.author_ts, c.id
        """).data()
        print(f"{len(commits)} labelled commits in stream.")

        ck = load_ckpt()
        # seed base files into file_state on a fresh start
        if not ck["file_state"]:
            base_files = s.run("MATCH (f:File)-[:HAS_AST]->() RETURN f.id AS f").data()
            ck["file_state"] = {r["f"]: BASE_COMMIT for r in base_files}
            print(f"Seeded {len(ck['file_state'])} base files at BASE_COMMIT.")

        start = ck["next_index"]
        end = len(commits) if args.limit <= 0 else min(len(commits), start + args.limit)
        print(f"Processing commits [{start}:{end}]...\n")

        agg = dict(adds=0, removes=0, updates=0, moves=0, bootstrapped=0,
                   new_files=0, deleted_files=0, skipped=0, matched=0, total=0)

        for idx in range(start, end):
            commit = commits[idx]["id"]
            _t_commit = time.perf_counter()
            changes = changed_java_files(commit)
            tag = "BUG " if commits[idx]["buggy"] else "    "
            line_counts = dict(A=0, M=0, D=0, boot=0)
            cc = dict(adds=0, removes=0, updates=0, moves=0, matched=0)

            for status, path in changes:
                if status == "D":
                    if path in ck["file_state"]:
                        delete_file(s, commit, path)
                        ck["file_state"].pop(path, None)
                        ck["file_maps"].pop(path, None)
                        line_counts["D"] += 1; agg["deleted_files"] += 1
                    continue

                if status == "A" or path not in ck["file_state"]:
                    # ADDED, or MODIFIED-but-untracked -> lazy bootstrap
                    ref = commit if status == "A" else f"{commit}^1"
                    blob = git_bytes(ref, path)
                    if blob is None:
                        agg["skipped"] += 1; continue
                    ast = build_full_ast(blob, path)
                    if ast is None:
                        agg["skipped"] += 1; continue
                    attach_full_ast(s, path, ast, commit, as_new_file=(status=="A"))
                    if status == "A":
                        ck["file_state"][path] = commit
                        line_counts["A"] += 1; agg["new_files"] += 1
                        continue
                    # bootstrapped at parent state; fall through to diff this commit
                    ck["file_state"][path] = f"{commit}^1"
                    line_counts["boot"] += 1; agg["bootstrapped"] += 1

                # MODIFIED tracked file: diff stored-state -> this commit
                before = git_bytes(ck["file_state"][path], path)
                after  = git_bytes(commit, path)
                if before is None or after is None:
                    agg["skipped"] += 1; continue
                res = process_modify(s, commit, path, before, after,
                                     ck["file_maps"].get(path, {}))
                if res is None:
                    agg["skipped"] += 1; continue
                stt, new_map = res
                for k in ("adds","removes","updates","moves","matched","total"):
                    agg[k] += stt[k]
                for k in ("adds","removes","updates","moves","matched"):
                    cc[k] += stt[k]
                line_counts["M"] += 1
                ck["file_state"][path] = commit
                ck["file_maps"][path] = new_map      # thread node identity forward

            ck["next_index"] = idx + 1
            if idx % 20 == 0 or args.limit:
                save_ckpt(ck)
            if tlog is not None:
                _tw.writerow([idx, commit, line_counts['A'], line_counts['M'],
                              line_counts['D'], line_counts['boot'],
                              cc['adds'], cc['removes'], cc['updates'], cc['moves'],
                              cc['matched'],
                              round(1000.0 * (time.perf_counter() - _t_commit), 2)])
            print(f"[{idx+1}/{len(commits)}] {tag}{commit[:8]}  "
                  f"A={line_counts['A']} M={line_counts['M']} "
                  f"D={line_counts['D']} boot={line_counts['boot']}")

        if tlog is not None:
            tlog.close()
        save_ckpt(ck)
        print(f"\n{'='*60}\nDONE [{start}:{end}]")
        print(f"  new files (ADDED):  {agg['new_files']}")
        print(f"  bootstrapped files: {agg['bootstrapped']}")
        print(f"  deleted files:      {agg['deleted_files']}")
        print(f"  delta edges: adds={agg['adds']} removes={agg['removes']} "
              f"updates={agg['updates']} moves={agg['moves']}")
        print(f"  nodes carried forward (matched): {agg['matched']}")
        print(f"  skipped (parse/git): {agg['skipped']}")
        print(f"  files tracked now:   {len(ck['file_state'])}")
        print("  -> run validate_online_kg.py to check evolved AST vs ground truth")
    driver.close()


if __name__ == "__main__":
    main()
