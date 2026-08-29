"""
Online KG growth engine for the ALTERNATIVE subgraphs (CFG / DFG / PDG / SEQ).

This is the parametrized twin of build_online_kg.py (the AST engine). It streams
the same 8,059 ApacheJIT-labelled Groovy commits in chronological (author_ts)
order and, per commit and per changed .java file:

  ADDED file                 -> build the full subgraph (HAS_<X>), mark ADDS.
  MODIFIED tracked file      -> build before/after subgraphs, run the SHARED
                                uniform differ (subgraph_diff.diff_graphs), write
                                delta edges ADDS/REMOVES/UPDATES/MOVES, evolve the
                                stored graph + re-stamp lines, thread node identity
                                forward via a per-file {parse_lid -> graph_id} map.
  MODIFIED untracked file    -> LAZY BOOTSTRAP from the parent blob, then diff.
  DELETED file               -> REMOVES every alive node of the file.

Nodes carry the label from subgraph_spec (:CFGNode / :DFGNode / :PDGNode /
:SEQNode) and the token type in property `atype`, so they coexist with the AST
layer in the same DB and never collide. Delta edges reuse the AST vocabulary
(ADDS/REMOVES/UPDATES/MOVES) typed only by the node label they point at.

Unlike the AST engine we do NOT seed base files: file_state starts empty and the
lazy bootstrap builds a file's subgraph the first time it is touched -- which is
exactly the set of files that ever contribute delta tokens.

Checkpoint is per kind (outputs/online_kg_checkpoint_<kind>.json); resumable;
--limit windows for testing; --reset wipes only THIS kind's layer.

Run:
  python build_subgraph_online_kg.py --kind cfg --limit 30
  python build_subgraph_online_kg.py --kind pdg
  python build_subgraph_online_kg.py --kind dfg --reset
"""
import argparse, json, subprocess, sys, tempfile, os, threading, queue, time, csv
from pathlib import Path
from neo4j import GraphDatabase

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import (REPO_PATH, REPO_PATHS, NEO4J_URI, NEO4J_AUTH,
                                    base_commit, ckpt_path as _cfg_ckpt_path,
                                    git_root_for)
import subgraph_spec as spec
import subgraph_diff as sdiff

# the subgraph builder worker lives beside this file (package build/ dir)
_BUILD_DIR   = Path(__file__).resolve().parent
BASE_COMMIT  = base_commit()             # loud check: online growth seeds from the base
BUILDER      = str(_BUILD_DIR / "subgraph_builders.py")


# ── git helpers (identical semantics to the AST engine; multi-repo aware) ─────
# See docs/Critical_notes.tex: some projects (hadoop-mapreduce) span two repos.
# git_bytes tries each repo in REPO_PATHS; changed_java_files uses the repo that
# owns the commit. Single-repo projects behave exactly as before.

def git_bytes(ref, path):
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
    if len(REPO_PATHS) == 1:
        return REPO_PATH
    return git_root_for(commit) or REPO_PATH

def changed_java_files(commit):
    repo = _commit_repo(commit)
    par = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{commit}^1"],
                         capture_output=True, text=True)
    if par.returncode != 0:
        r = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r",
                            "--name-only", commit], capture_output=True, text=True)
        return [("A", p) for p in r.stdout.splitlines() if p.endswith(".java")]
    r = subprocess.run(["git", "-C", str(repo), "diff", "--name-status",
                        "-M", f"{commit}^1", commit], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        st = parts[0]
        if st.startswith("R"):
            oldp, newp = parts[1], parts[2]
            if oldp.endswith(".java"): out.append(("D", oldp))
            if newp.endswith(".java"): out.append(("A", newp))
        elif st in ("A", "M", "D") and parts[1].endswith(".java"):
            out.append((st, parts[1]))
    return out


# ── persistent builder worker (amortizes import cost; watchdog restarts it) ──

class BuilderServer:
    """One long-lived `subgraph_builders.py --serve` process. Requests go over
    stdin, responses over stdout, read on a background thread so a hung/patho-
    logical file trips a per-request timeout -> we kill and respawn the worker
    (matching the old subprocess-per-file safety, without paying startup each
    call)."""
    def __init__(self, timeout=90):
        self.timeout = timeout
        self.proc = None; self.q = None; self.reader = None
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            [sys.executable, BUILDER, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            cwd=str(_BUILD_DIR), text=True, bufsize=1)
        self.q = queue.Queue()
        self.reader = threading.Thread(target=self._pump, daemon=True)
        self.reader.start()

    def _pump(self):
        for line in self.proc.stdout:
            self.q.put(line)

    def _restart(self):
        try: self.proc.kill()
        except Exception: pass
        self._start()

    def build(self, kind, src_bytes, rel_path):
        text = src_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
            f.write(text.encode("utf-8")); tmp = f.name
        try:
            self.proc.stdin.write(json.dumps({"kind": kind, "path": tmp, "rel": rel_path}) + "\n")
            self.proc.stdin.flush()
            line = self.q.get(timeout=self.timeout)
            d = json.loads(line)
            return d if "error" not in d else None
        except (queue.Empty, BrokenPipeError, ValueError, OSError):
            self._restart(); return None
        finally:
            try: os.unlink(tmp)
            except OSError: pass

    def close(self):
        try: self.proc.stdin.close(); self.proc.kill()
        except Exception: pass


# ── Neo4j writes (label parametrized from the spec) ──────────────────────────

def attach_full(session, sp, file_id, graph, commit_id, as_new_file):
    L, AR = sp.node_label, sp.attach_rel
    # Every write here is chunked into its OWN transaction (execute_write).
    # A bare session.run keeps all statements in one auto-commit scope, so a
    # large file accumulates until dbms.memory.transaction.total.max is hit --
    # the failure that stopped B5 twice on hive. See build_online_kg.py.
    B = 2000

    def _chunked(q, key, items, **fixed):
        for i in range(0, len(items), B):
            b = items[i:i + B]
            session.execute_write(
                lambda tx, b=b: tx.run(q, **{key: b}, **fixed).consume())

    session.run(f"MERGE (f:File {{id:$file}})", file=file_id)
    _chunked(f"""
        UNWIND $nodes AS n
        MERGE (a:{L} {{id:n.id}})
        ON CREATE SET a.file=$file, a.atype=n.type, a.group=n.group,
                      a.method=n.method, a.value=n.value,
                      a.pos_line=n.line, a.pos_col=n.col, a.alive=true
    """, "nodes", graph["nodes"], file=file_id)
    if graph["edges"]:
        _chunked(f"""
            UNWIND $edges AS e
            MATCH (p:{L} {{id:e.src}}), (c:{L} {{id:e.dst}})
            MERGE (p)-[r:SUB_EDGE {{kind:$kind}}]->(c) ON CREATE SET r.etype=e.etype
        """, "edges", graph["edges"], kind=sp.kind)
    if graph["roots"]:
        _chunked(f"""
            MATCH (f:File {{id:$file}})
            UNWIND $roots AS rid
            MATCH (r:{L} {{id:rid}}) MERGE (f)-[:{AR}]->(r)
        """, "roots", graph["roots"], file=file_id)
    if as_new_file:
        _chunked(f"""
            MATCH (c:Commit {{id:$cid}})
            UNWIND $ids AS nid
            MATCH (a:{L} {{id:nid}})
            MERGE (c)-[:ADDS]->(a)
        """, "ids", [n["id"] for n in graph["nodes"]], cid=commit_id)


def delete_file(session, sp, commit_id, file_id):
    return session.run(f"""
        MATCH (c:Commit {{id:$cid}}), (a:{sp.node_label} {{file:$file}})
        WHERE a.alive = true
        SET a.alive=false, a.removed_by=$cid
        MERGE (c)-[:REMOVES]->(a)
        RETURN count(a) AS n
    """, cid=commit_id, file=file_id).single()["n"]


# ── core: process a MODIFIED tracked file via the shared differ ──────────────

def process_modify(session, bs, graph_cache, sp, kind, use_multiset, commit_id,
                   file_id, before_bytes, after_bytes, node_map):
    # reuse last commit's after-graph as this commit's before-graph (halves builds)
    bg = graph_cache.get(file_id) or bs.build(kind, before_bytes, file_id)
    ag = bs.build(kind, after_bytes, file_id)
    if bg is None or ag is None:
        return None
    d = (sdiff.diff_multiset if use_multiset else sdiff.diff_graphs)(bg, ag)

    L = sp.node_label
    gid_of = lambda blid: node_map.get(blid, blid)
    after_by_lid = {n["id"]: n for n in d["after_nodes"]}
    st = dict(adds=0, removes=0, updates=0, moves=0, matched=0, total=0)
    new_map = {}

    # NB: we deliberately do NOT restamp positions of matched nodes each commit.
    # The differ matches by ordinal computed from the parse (not from stored
    # positions), and nothing downstream (tokens / validation) reads pos_line;
    # restamping every alive node was an O(file-size) write per commit -- the
    # dominant cost -- so it is dropped. Per-commit cost is now O(change).
    updates = []
    for before_lid, after_lid, vchanged, newval in d["matched"]:
        g = gid_of(before_lid); new_map[after_lid] = g
        st["matched"] += 1
        if vchanged:
            updates.append({"id": g, "new": newval}); st["updates"] += 1; st["total"] += 1

    removes = [gid_of(b) for b in d["deleted"]]
    st["removes"] = len(removes); st["total"] += len(removes)

    inserts = []
    for alid in d["inserted"]:
        an = after_by_lid[alid]
        suffix = alid.rsplit("::", 1)[-1]
        gnew = f"{file_id}::D{commit_id[:8]}:{suffix}"
        new_map[alid] = gnew
        inserts.append({"id": gnew, "atype": an["type"], "group": an["group"],
                        "method": an["method"], "val": an["value"],
                        "line": an["line"], "col": an["col"]})
    st["adds"] = len(inserts)

    moves = [{"id": new_map[alid], "npt": npt} for alid, npt in d["moved"]
             if alid in new_map]
    st["moves"] = len(moves)

    # edges touching an inserted node (matched-matched edges already persist)
    ins_lids = set(d["inserted"])
    edge_rows = []
    for e in d["after_edges"]:
        if e["src"] in ins_lids or e["dst"] in ins_lids:
            s, t = new_map.get(e["src"]), new_map.get(e["dst"])
            if s and t:
                edge_rows.append({"src": s, "dst": t, "etype": e["etype"]})

    # ── apply ──
    if removes:
        session.run(f"""
            MATCH (c:Commit {{id:$cid}}) UNWIND $ids AS gid
            MATCH (a:{L} {{id:gid}})
            SET a.alive=false, a.removed_by=$cid
            MERGE (c)-[:REMOVES]->(a)
        """, cid=commit_id, ids=removes)
    if updates:
        session.run(f"""
            MATCH (c:Commit {{id:$cid}}) UNWIND $rows AS u
            MATCH (a:{L} {{id:u.id}})
            MERGE (c)-[r:UPDATES]->(a) ON CREATE SET r.old_value=a.value, r.new_value=u.new
            SET a.value=u.new
        """, cid=commit_id, rows=updates)
    if inserts:
        session.run(f"""
            MATCH (c:Commit {{id:$cid}}) UNWIND $rows AS n
            MERGE (a:{L} {{id:n.id}})
              ON CREATE SET a.file=$file, a.atype=n.atype, a.group=n.group,
                            a.method=n.method, a.value=n.val,
                            a.pos_line=n.line, a.pos_col=n.col,
                            a.is_delta=true, a.alive=true
            MERGE (c)-[:ADDS]->(a)
        """, cid=commit_id, file=file_id, rows=inserts)
    if edge_rows:
        session.run(f"""
            UNWIND $rows AS e
            MATCH (p:{L} {{id:e.src}}), (c:{L} {{id:e.dst}})
            MERGE (p)-[r:SUB_EDGE {{kind:$kind}}]->(c) ON CREATE SET r.etype=e.etype
        """, rows=edge_rows, kind=kind)
    if moves:
        session.run(f"""
            MATCH (c:Commit {{id:$cid}}) UNWIND $rows AS m
            MATCH (a:{L} {{id:m.id}})
            MERGE (c)-[r:MOVES]->(a) ON CREATE SET r.new_parent_type=m.npt
        """, cid=commit_id, rows=moves)
    return st, new_map, ag


# ── checkpoint (per kind) ────────────────────────────────────────────────────

def ckpt_path(kind):
    return _cfg_ckpt_path(kind)          # outputs/<project>/online_kg_checkpoint_<kind>.json

def load_ckpt(kind):
    p = ckpt_path(kind)
    if p.exists():
        ck = json.loads(p.read_text()); ck.setdefault("file_maps", {}); return ck
    return {"next_index": 0, "file_state": {}, "file_maps": {}}

def save_ckpt(kind, ck):
    ckpt_path(kind).write_text(json.dumps(ck))


def reset_layer(session, sp):
    print(f"Resetting {sp.node_label} layer...")
    while True:
        n = session.run(f"""
            MATCH (a:{sp.node_label}) WITH a LIMIT 20000 DETACH DELETE a
            RETURN count(a) AS n
        """).single()["n"]
        if not n:
            break
        print(f"  deleted {n} nodes...")


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(spec.REGISTRY))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--multiset", action="store_true",
                    help="use the lightweight typed-multiset fallback differ")
    ap.add_argument("--timing-log", default=None,
                    help="optional CSV path; append one row per commit "
                         "(idx, sha, files, delta counts, wall_ms) for the "
                         "scalability/complexity analysis. Off by default so the "
                         "normal build is unchanged.")
    args = ap.parse_args()
    sp = spec.get(args.kind)

    tlog = _tw = None
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
        # CRITICAL: index this layer's id/file (like :ASTNode(id)) -- without it
        # every MATCH/MERGE (a:<Label> {id:..}) is a full label scan, making the
        # whole build O(N) per lookup and quadratic overall.
        L = sp.node_label
        s.run(f"CREATE INDEX {L.lower()}_id IF NOT EXISTS FOR (a:{L}) ON (a.id)")
        s.run(f"CREATE INDEX {L.lower()}_file IF NOT EXISTS FOR (a:{L}) ON (a.file)")
        s.run("CREATE INDEX file_id IF NOT EXISTS FOR (f:File) ON (f.id)")
        # composite (file, alive): the per-file liveness lookup (delete_file /
        # load_current) filters a.alive after matching {file:$file}; without this
        # it scans a file's dead-node history and slows as the graph grows (the
        # missing-index cliff -- see docs/Critical_notes.tex). Backfill null->true
        # so the seek is fully covering.
        s.run(f"CREATE INDEX {L.lower()}_file_alive IF NOT EXISTS FOR (a:{L}) ON (a.file, a.alive)")
        s.run(f"MATCH (a:{L}) WHERE a.alive IS NULL SET a.alive=true")

        if args.reset:
            reset_layer(s, sp)
            if ckpt_path(args.kind).exists(): ckpt_path(args.kind).unlink()

        commits = s.run("""
            MATCH (c:Commit {in_jit:true})
            RETURN c.id AS id, c.author_ts AS ts, c.buggy AS buggy
            ORDER BY c.author_ts, c.id
        """).data()
        print(f"[{args.kind}] {len(commits)} labelled commits in stream.")

        ck = load_ckpt(args.kind)
        start = ck["next_index"]
        end = len(commits) if args.limit <= 0 else min(len(commits), start + args.limit)
        print(f"Processing commits [{start}:{end}] "
              f"(differ={'multiset' if args.multiset else 'shared'})...\n")

        bs = BuilderServer()
        graph_cache = {}     # file_id -> last after-graph JSON (next before-graph)
        agg = dict(adds=0, removes=0, updates=0, moves=0, bootstrapped=0,
                   new_files=0, deleted_files=0, skipped=0, matched=0)

        for idx in range(start, end):
            commit = commits[idx]["id"]
            _t_commit = time.perf_counter()
            changes = changed_java_files(commit)
            tag = "BUG " if commits[idx]["buggy"] else "    "
            lc = dict(A=0, M=0, D=0, boot=0)
            cc = dict(adds=0, removes=0, updates=0, moves=0, matched=0)

            for status, path in changes:
                if status == "D":
                    if path in ck["file_state"]:
                        delete_file(s, sp, commit, path)
                        ck["file_state"].pop(path, None); ck["file_maps"].pop(path, None)
                        graph_cache.pop(path, None)
                        lc["D"] += 1; agg["deleted_files"] += 1
                    continue

                if status == "A" or path not in ck["file_state"]:
                    ref = commit if status == "A" else f"{commit}^1"
                    blob = git_bytes(ref, path)
                    if blob is None: agg["skipped"] += 1; continue
                    graph = bs.build(args.kind, blob, path)
                    if graph is None: agg["skipped"] += 1; continue
                    attach_full(s, sp, path, graph, commit, as_new_file=(status == "A"))
                    graph_cache[path] = graph
                    if status == "A":
                        ck["file_state"][path] = commit
                        lc["A"] += 1; agg["new_files"] += 1; continue
                    ck["file_state"][path] = f"{commit}^1"
                    lc["boot"] += 1; agg["bootstrapped"] += 1

                # before-bytes only needed on a cold cache miss (else reuse graph)
                before = b"" if path in graph_cache else git_bytes(ck["file_state"][path], path)
                after  = git_bytes(commit, path)
                if before is None or after is None: agg["skipped"] += 1; continue
                res = process_modify(s, bs, graph_cache, sp, args.kind, args.multiset,
                                     commit, path, before, after,
                                     ck["file_maps"].get(path, {}))
                if res is None: agg["skipped"] += 1; continue
                stt, new_map, ag = res
                for k in ("adds", "removes", "updates", "moves", "matched"):
                    agg[k] += stt[k]; cc[k] += stt[k]
                lc["M"] += 1
                ck["file_state"][path] = commit
                ck["file_maps"][path] = new_map
                graph_cache[path] = ag

            ck["next_index"] = idx + 1
            if idx % 20 == 0 or args.limit:
                save_ckpt(args.kind, ck)
            if _tw is not None:
                _tw.writerow([idx, commit, lc['A'], lc['M'], lc['D'], lc['boot'],
                              cc['adds'], cc['removes'], cc['updates'], cc['moves'],
                              cc['matched'],
                              round(1000.0 * (time.perf_counter() - _t_commit), 2)])
            print(f"[{idx+1}/{len(commits)}] {tag}{commit[:8]}  "
                  f"A={lc['A']} M={lc['M']} D={lc['D']} boot={lc['boot']}")

        if tlog is not None:
            tlog.close()
        save_ckpt(args.kind, ck)
        print(f"\n{'='*60}\nDONE [{args.kind}] [{start}:{end}]")
        print(f"  new files={agg['new_files']} bootstrapped={agg['bootstrapped']} "
              f"deleted={agg['deleted_files']}")
        print(f"  delta edges: adds={agg['adds']} removes={agg['removes']} "
              f"updates={agg['updates']} moves={agg['moves']}")
        print(f"  matched carried forward={agg['matched']} skipped={agg['skipped']}")
        print(f"  files tracked now={len(ck['file_state'])}")
        bs.close()
    driver.close()


if __name__ == "__main__":
    main()
