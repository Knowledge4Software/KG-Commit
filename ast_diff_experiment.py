#!/usr/bin/env python3
"""
AST Diff Experiment — GumTree-based structural differencing on sample commits.

For 10 sample commits from apache/groovy that modified base-snapshot files,
this script:
  1. Extracts before/after .java file pairs using git
  2. Runs GumTree 2.1.2 (Java 8 compatible) to produce a JSON edit script
  3. Parses the edit script (INSERT / DELETE / UPDATE / MOVE actions)
  4. Stores results in Neo4j as ASTDiff + ASTEdit nodes
  5. Exports one interactive HTML visualisation per commit via pyvis

Neo4j schema added:
  (:ASTDiff  {id, commit_id, file, n_insert, n_delete, n_update, n_move, total})
  (:ASTEdit  {id, diff_id, action, node_type, node_label, parent_type, pos_start, pos_end})
  (:Commit)-[:HAS_AST_DIFF]->(:ASTDiff)
  (:ASTDiff)-[:EDIT]->(:ASTEdit)
  (:File)-[:DIFFED_AT]->(:ASTDiff)

Run:
  python ast_diff_experiment.py
  python ast_diff_experiment.py --samples 5     # fewer samples
  python ast_diff_experiment.py --skip-neo4j    # parse only, skip DB write
"""

import sys, json, os, re, subprocess, tempfile, argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_PATH    = PROJECT_ROOT / "repos" / "apache" / "groovy"
GT_LIB       = PROJECT_ROOT / "tools" / "gumtree" / "gumtree-2.1.2" / "lib"
OUT_DIR      = PROJECT_ROOT / "outputs" / "ast_diff"

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password1234"
BASE_COMMIT    = "408b29851d7bbe4d343340832297e4be7e0c5578"

# ── GumTree invocation ────────────────────────────────────────────────────────
def run_gumtree_diff(before: Path, after: Path, timeout: int = 60) -> list[str] | None:
    """Run GumTree 2.1.2 'diff' (text format) and return action lines, or None."""
    cp = str(GT_LIB / "*")
    cmd = [
        "java", "-cp", cp,
        "com.github.gumtreediff.client.Run",
        "diff", "-g", "java-javaparser",
        str(before), str(after),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    # GumTree writes everything to stderr; exit 255 is normal for this version
    raw = r.stderr or r.stdout
    if not raw:
        return None
    # Keep only action lines (Insert / Delete / Update / Move); skip Match lines
    lines = [ln for ln in raw.splitlines()
             if ln.startswith(("Insert ", "Delete ", "Update ", "Move "))]
    return lines if lines else None

# ── Parse GumTree text-diff action lines ─────────────────────────────────────
# Text diff format (GumTree 2.x):
#   Insert NodeType(id) into ParentType(id) at pos
#   Insert NodeType: label(id) into ParentType(id) at pos
#   Delete NodeType(id)
#   Delete NodeType: label(id)
#   Update NodeType: label(id) to new_label
#   Move   NodeType(id) into ParentType(id) at pos
_NODE = r'(\w+)(?:: (.+?))?'                       # NodeType [: label]
_INSERT_RE = re.compile(rf'^(Insert|Move) {_NODE}\(\d+\) into {_NODE}\(\d+\) at (\d+)$')
_DELETE_RE = re.compile(rf'^(Delete) {_NODE}\(\d+\)$')
_UPDATE_RE = re.compile(rf'^(Update) {_NODE}\(\d+\) to (.+)$')

def parse_action_line(line: str) -> dict | None:
    line = line.strip()
    m = _INSERT_RE.match(line)
    if m:
        action, nt, nl, pt, pl, pos = m.groups()
        return {"action": action.lower(), "node_type": nt, "node_label": nl or "",
                "parent_type": pt, "position": int(pos), "new_label": ""}
    m = _DELETE_RE.match(line)
    if m:
        action, nt, nl = m.groups()
        return {"action": "delete", "node_type": nt, "node_label": nl or "",
                "parent_type": "", "position": -1, "new_label": ""}
    m = _UPDATE_RE.match(line)
    if m:
        action, nt, nl, new_label = m.groups()
        return {"action": "update", "node_type": nt, "node_label": nl or "",
                "parent_type": "", "position": -1, "new_label": new_label.strip()}
    return None

def parse_actions(lines: list[str]) -> list[dict]:
    edits = []
    for i, line in enumerate(lines):
        parsed = parse_action_line(line)
        if parsed:
            parsed["idx"] = i
            edits.append(parsed)
    return edits

# ── Git helpers ───────────────────────────────────────────────────────────────
_ENC = __import__("sys").getfilesystemencoding()

def git_show_file(commit: str, rel_path: str) -> str | None:
    """Return file content at a given commit, or None if absent."""
    r = subprocess.run(
        ["git", "-C", str(REPO_PATH), "show", f"{commit}:{rel_path}"],
        capture_output=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace")

def git_parent(commit: str) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(REPO_PATH), "rev-parse", f"{commit}^"],
        capture_output=True, text=True, timeout=15,
    )
    return r.stdout.strip() if r.returncode == 0 else None

def write_no_bom(path: Path, content: str):
    path.write_bytes(content.encode("utf-8"))

# ── Neo4j ingestion ───────────────────────────────────────────────────────────
def setup_constraints(session):
    session.execute_write(lambda tx: tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:ASTDiff) REQUIRE d.id IS UNIQUE"
    ).consume())
    session.execute_write(lambda tx: tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:ASTEdit) REQUIRE e.id IS UNIQUE"
    ).consume())

def ingest_diff(session, commit_id: str, file_path: str, edits: list[dict]):
    counts = Counter(e["action"] for e in edits)
    diff_id = f"{commit_id[:8]}::{file_path}"

    def _write(tx):
        # ASTDiff node
        tx.run("""
            MERGE (d:ASTDiff {id: $did})
            ON CREATE SET d.commit_id = $cid, d.file = $file,
                          d.n_insert = $ni, d.n_delete = $nd,
                          d.n_update = $nu, d.n_move = $nm,
                          d.total = $tot
        """, did=diff_id, cid=commit_id, file=file_path,
             ni=counts.get("insert", 0), nd=counts.get("delete", 0),
             nu=counts.get("update", 0), nm=counts.get("move", 0),
             tot=len(edits))

        # Link Commit -> ASTDiff
        tx.run("""
            MATCH (c:Commit {id: $cid})
            MATCH (d:ASTDiff {id: $did})
            MERGE (c)-[:HAS_AST_DIFF]->(d)
        """, cid=commit_id, did=diff_id)

        # Link File -> ASTDiff
        tx.run("""
            MATCH (f:File {id: $fid})
            MATCH (d:ASTDiff {id: $did})
            MERGE (f)-[:DIFFED_AT]->(d)
        """, fid=file_path, did=diff_id)

        # ASTEdit nodes
        edit_nodes = [
            {"id": f"{diff_id}::{e['idx']}", "diff_id": diff_id,
             "action": e["action"], "node_type": e["node_type"],
             "node_label": e["node_label"][:120],
             "new_label": e.get("new_label", "")[:120],
             "parent_type": e["parent_type"],
             "position": e["position"]}
            for e in edits
        ]
        tx.run("""
            UNWIND $edits AS e
            MERGE (ae:ASTEdit {id: e.id})
            ON CREATE SET ae.diff_id    = e.diff_id,
                          ae.action     = e.action,
                          ae.node_type  = e.node_type,
                          ae.node_label = e.node_label,
                          ae.new_label  = e.new_label,
                          ae.parent_type= e.parent_type,
                          ae.position   = e.position
        """, edits=edit_nodes)
        tx.run("""
            MATCH (d:ASTDiff {id: $did})
            WITH d
            MATCH (ae:ASTEdit) WHERE ae.diff_id = $did
            MERGE (d)-[:EDIT]->(ae)
        """, did=diff_id)

    session.execute_write(_write)

# ── pyvis HTML export for one diff ───────────────────────────────────────────
ACTION_COLORS = {
    "insert": "#27ae60",   # green
    "delete": "#e74c3c",   # red
    "update": "#e67e22",   # orange
    "move":   "#3498db",   # blue
}

def export_diff_html(commit_id: str, file_path: str,
                     edits: list[dict], out_path: Path):
    from pyvis.network import Network
    net = Network(height="850px", width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="white", notebook=False)
    net.set_options("""{
      "physics": {"solver": "forceAtlas2Based",
        "forceAtlas2Based": {"gravitationalConstant": -80, "springLength": 100},
        "stabilization": {"iterations": 200}},
      "interaction": {"hover": true, "navigationButtons": true}
    }""")

    # Central commit node
    counts = Counter(e["action"] for e in edits)
    commit_label = commit_id[:8]
    net.add_node("commit", label=commit_label, color="#9b59b6", size=40,
                 title=(f"Commit: {commit_id}\nFile: {file_path}\n"
                        f"insert={counts['insert']}  delete={counts['delete']}  "
                        f"update={counts['update']}  move={counts['move']}  "
                        f"total={len(edits)}"))

    # One node per unique (action, node_type) — sized by frequency
    # Collect parent_types per (action, node_type) for the tooltip
    type_counts: Counter = Counter()
    parent_map: dict[tuple, set] = {}
    for e in edits:
        nt  = e["node_type"]  or "Unknown"
        pt  = e["parent_type"] or "—"
        key = (e["action"], nt)
        type_counts[key] += 1
        parent_map.setdefault(key, set()).add(pt)

    # Add one node per (action, node_type) + one edge commit -> node
    for (action, nt), cnt in type_counts.items():
        nid    = f"{action}::{nt}"
        parents = ", ".join(sorted(parent_map[(action, nt)]))
        net.add_node(nid,
                     label=f"{nt}\n({action})",
                     color=ACTION_COLORS.get(action, "#95a5a6"),
                     size=max(14, min(45, 12 + cnt * 2)),
                     title=f"Action: {action}\nType: {nt}\nCount: {cnt}\nParent(s): {parents}")
        net.add_edge("commit", nid,
                     color=ACTION_COLORS.get(action, "#888"),
                     width=max(1, min(6, cnt // 5 + 1)),
                     arrows="to",
                     title=f"{cnt} {action} edits")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path))

# ── Main experiment loop ──────────────────────────────────────────────────────
def select_sample_commits(session, n: int) -> list[dict]:
    """Pick n diverse commit-file pairs: one commit per unique base-snapshot file."""
    result = session.run("""
        MATCH (c:Commit)-[:MODIFIED]->(f:File)-[:HAS_AST]->(:ASTNode)
        WITH f.id AS file, collect(c.id)[0] AS commit_id
        RETURN commit_id, file
        LIMIT $n
    """, n=n)
    return [{"commit_id": r["commit_id"], "file": r["file"]} for r in result]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples",    type=int, default=10)
    ap.add_argument("--skip-neo4j", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ast_diff_"))

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            if not args.skip_neo4j:
                setup_constraints(session)

            print(f"Selecting {args.samples} sample commits...")
            samples = select_sample_commits(session, args.samples)
            print(f"Found {len(samples)} candidates.\n")

            summary_rows = []

            for i, s in enumerate(samples, 1):
                commit_id = s["commit_id"]
                file_path = s["file"]
                short     = commit_id[:8]

                print(f"[{i}/{len(samples)}] {short}  {file_path}")

                parent_id = git_parent(commit_id)
                if not parent_id:
                    print(f"  SKIP: no parent commit found")
                    continue

                before_src = git_show_file(parent_id, file_path)
                after_src  = git_show_file(commit_id, file_path)

                if before_src is None or after_src is None:
                    print(f"  SKIP: file not found in one of the commits")
                    continue

                bf = tmp / f"{short}_before.java"
                af = tmp / f"{short}_after.java"
                write_no_bom(bf, before_src)
                write_no_bom(af, after_src)

                lines = run_gumtree_diff(bf, af)
                if lines is None:
                    print(f"  SKIP: GumTree failed or timed out")
                    continue

                edits = parse_actions(lines)
                counts = Counter(e["action"] for e in edits)
                print(f"  {len(edits)} edits — "
                      f"insert={counts['insert']} delete={counts['delete']} "
                      f"update={counts['update']} move={counts['move']}")

                if not args.skip_neo4j:
                    ingest_diff(session, commit_id, file_path, edits)
                    print(f"  -> stored in Neo4j as ASTDiff {short}::{file_path}")

                html_out = OUT_DIR / f"{short}_{Path(file_path).stem}_diff.html"
                export_diff_html(commit_id, file_path, edits, html_out)
                print(f"  -> HTML: {html_out.name}")

                summary_rows.append({
                    "commit":   short,
                    "file":     file_path,
                    "insert":   counts["insert"],
                    "delete":   counts["delete"],
                    "update":   counts["update"],
                    "move":     counts["move"],
                    "total":    len(edits),
                })

            print("\n" + "=" * 70)
            print(f"{'COMMIT':8}  {'INSERT':7} {'DELETE':7} {'UPDATE':7} "
                  f"{'MOVE':6} {'TOTAL':6}  FILE")
            print("-" * 70)
            for r in summary_rows:
                fn = Path(r["file"]).name
                print(f"{r['commit']:8}  {r['insert']:7} {r['delete']:7} "
                      f"{r['update']:7} {r['move']:6} {r['total']:6}  {fn}")
            print("=" * 70)

            if not args.skip_neo4j:
                counts_q = session.run("""
                    MATCH (d:ASTDiff) RETURN count(d) AS diffs
                """).single()
                edits_q = session.run("""
                    MATCH (e:ASTEdit) RETURN count(e) AS edits
                """).single()
                print(f"\nNeo4j: {counts_q['diffs']} ASTDiff nodes, "
                      f"{edits_q['edits']} ASTEdit nodes")

    finally:
        driver.close()
        import shutil; shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nHTML files in: {OUT_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()
