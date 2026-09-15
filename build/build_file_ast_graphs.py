#!/usr/bin/env python3
"""
File-level (whole compilation-unit) AST extractor for the KG-Commit base
snapshot, scoped to apache/groovy at BASE_COMMIT.

Unlike build_ast_features.py (per-METHOD AST stats for Family 2), this script
builds ONE full AST graph per .java file, rooted at the file itself, so it can
be attached under each File node in the Neo4j knowledge graph
(see ingest_base_kg_and_ast.py). Reuses the proven javalang traversal helpers
from build_ast_features.py unchanged (same subprocess-per-file + timeout
pattern that fixed the historical javalang hang on Groovyc.java).

Outputs (under outputs/):
  file_ast_subgraph_stats.csv   one row per file (deliverable)
  file_ast_aggregates.json      group_totals/type_totals + ok/skipped files
  file_ast_graphs/*.pkl         (G, root_nid) full AST graph per file

Run:
  python build_file_ast_graphs.py > outputs/file_ast_build_log.txt
"""

import sys, csv, json, pickle, subprocess, tempfile, os
from pathlib import Path
from collections import Counter

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
import build_ast_features as bf
from config.project_config import base_commit

PROJECT_ROOT = bf.PROJECT_ROOT
REPO_PATH    = bf.REPO_PATH
BASE_COMMIT  = base_commit()             # loud check: base snapshot needs a base commit
OUT_DIR      = bf.OUT_DIR
GRAPH_DIR    = OUT_DIR / "file_ast_graphs"

PARSE_TIMEOUT_S = bf.PARSE_TIMEOUT_S
MAX_SRC_BYTES   = bf.MAX_SRC_BYTES

git = bf.git


# ═════════════════════════════════════════════════════════════════════════
#  WORKER MODE (run as a child via subprocess; killed on timeout by parent)
# ═════════════════════════════════════════════════════════════════════════
def _worker_wholefile(abs_path, rel_path, out_path):
    import javalang, javalang.tree as jlt
    import networkx as nx

    src  = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    tree = javalang.parse.parse(src)   # tree IS the CompilationUnit root

    # File-specific counts: ONE linear pass over the whole subtree. This is
    # the SAFE top-level usage of javalang's `for x in tree` (walk_tree once),
    # not the nested-stack trap (see memory: javalang-node-iteration-trap).
    has_pkg = False
    n_classes = n_methods = n_fields = n_imports = 0
    for _, node in tree:
        if isinstance(node, jlt.PackageDeclaration):
            has_pkg = True
        elif isinstance(node, (jlt.ClassDeclaration, jlt.InterfaceDeclaration, jlt.EnumDeclaration)):
            n_classes += 1
        elif isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            n_methods += 1
        elif isinstance(node, jlt.FieldDeclaration):
            n_fields += 1
        elif isinstance(node, jlt.Import):
            n_imports += 1

    tc, gc, n, n_e, max_d, avg_d, avg_br, leaf_r = bf.traverse_method_stats(tree, jlt)

    def pct(g):
        return round(100 * gc.get(g, 0) / max(1, n), 1)

    row = {
        'file': rel_path,
        'n_ast_nodes': n, 'n_edges': n_e,
        'max_depth': max_d, 'avg_depth': avg_d, 'avg_branching': avg_br,
        'n_node_types': len(tc), 'n_groups': len(gc),
        'type_entropy':  round(bf.shannon_entropy(list(tc.values())), 4),
        'group_entropy': round(bf.shannon_entropy(list(gc.values())), 4),
        'leaf_ratio': leaf_r,
        'pct_declaration': pct('declaration'), 'pct_statement': pct('statement'),
        'pct_expression': pct('expression'),   'pct_literal': pct('literal'),
        'pct_leaf': pct('leaf'),
        'n_classes': n_classes, 'n_methods': n_methods,
        'n_fields': n_fields, 'n_imports': n_imports,
        'has_package': has_pkg,
        'file_size_bytes': len(src.encode('utf-8')),
    }

    # Full AST graph, reusing the existing (proven, trap-free) builder as-is.
    Ag, root_nid = bf._build_ast_subgraph(tree, rel_path, jlt, nx)
    depths = nx.shortest_path_length(Ag, source=root_nid)   # tree -> exact depth
    nx.set_node_attributes(Ag, depths, name='depth')
    Ag.graph['file'] = rel_path   # self-describing pickle, no manifest needed

    safe = "".join(c if c.isalnum() else "_" for c in rel_path)
    pf = GRAPH_DIR / f"{safe}.pkl"
    pf.write_bytes(pickle.dumps((Ag, root_nid), protocol=4))

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'row': row, 'gtot': dict(gc), 'ttot': dict(tc)}, f)


# ── Parent: run the worker in a killable subprocess (re-invokes THIS file) ──
def run_worker(args, timeout):
    """Returns parsed JSON dict on success, or a status string on failure."""
    fd, out_path = tempfile.mkstemp(suffix='.json'); os.close(fd)
    cmd = [sys.executable, os.path.abspath(__file__), '--worker-wholefile', *args, out_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        try: os.unlink(out_path)
        except OSError: pass
        return 'TIMEOUT'
    if proc.returncode != 0:
        try: os.unlink(out_path)
        except OSError: pass
        err = (proc.stderr or '').strip().splitlines()
        return 'ERR:' + (err[-1] if err else f'exit {proc.returncode}')
    try:
        with open(out_path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return f'ERR:no-output ({e})'
    finally:
        try: os.unlink(out_path)
        except OSError: pass
    return data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    for old in GRAPH_DIR.glob('*.pkl'):
        old.unlink()

    head = git("rev-parse HEAD")
    print(f"Current HEAD : {head[:12]}", flush=True)
    git(f"checkout {BASE_COMMIT}")
    print(f"Checked out  : {git('rev-parse HEAD')[:12]}\n", flush=True)

    java_files = sorted(REPO_PATH.rglob("*.java"))
    n_files = len(java_files)
    print(f"Found {n_files} .java files under {REPO_PATH}\n", flush=True)

    all_rows = []
    group_totals, type_totals = Counter(), Counter()
    ok_files, skipped_files = [], []

    for i, fp in enumerate(java_files, 1):
        rel = fp.relative_to(REPO_PATH).as_posix()
        tag = f"[{i}/{n_files}] {rel}"
        try:
            size = fp.stat().st_size
        except OSError:
            size = 0
        if size > MAX_SRC_BYTES:
            print(f"{tag}  SKIP (too large, {size//1024} KB)", flush=True)
            skipped_files.append(rel); continue

        res = run_worker([str(fp), rel], PARSE_TIMEOUT_S)
        if isinstance(res, dict):
            all_rows.append(res['row'])
            group_totals += Counter(res['gtot'])
            type_totals  += Counter(res['ttot'])
            ok_files.append(rel)
            print(f"{tag}  ok ({res['row']['n_ast_nodes']} AST nodes)", flush=True)
        elif res == 'TIMEOUT':
            print(f"{tag}  KILLED (parse > {PARSE_TIMEOUT_S}s)", flush=True)
            skipped_files.append(rel)
        else:
            print(f"{tag}  {res}", flush=True)
            skipped_files.append(rel)

    print(f"\nParsed OK : {len(ok_files)} files", flush=True)
    print(f"Skipped   : {len(skipped_files)} files", flush=True)

    if not all_rows:
        print("No files extracted - aborting."); git(f"checkout {head}"); return

    cols = list(all_rows[0].keys())
    with open(OUT_DIR / "file_ast_subgraph_stats.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)
    with open(OUT_DIR / "file_ast_aggregates.json", 'w', encoding='utf-8') as f:
        json.dump({'group_totals': dict(group_totals), 'type_totals': dict(type_totals),
                   'n_files': len(all_rows), 'ok_files': len(ok_files),
                   'skipped_files': skipped_files}, f)

    n_vals = [r['n_ast_nodes'] for r in all_rows]
    d_vals = [r['max_depth']   for r in all_rows]
    print(f"\nAST size  : min={min(n_vals)} med={sorted(n_vals)[len(n_vals)//2]} "
          f"mean={sum(n_vals)/len(n_vals):.0f} max={max(n_vals)}")
    print(f"Max depth : min={min(d_vals)} med={sorted(d_vals)[len(d_vals)//2]} "
          f"mean={sum(d_vals)/len(d_vals):.1f} max={max(d_vals)}")
    print(f"Top groups: {dict(group_totals.most_common(6))}")
    print(f"\nGraphs saved : {len(ok_files)} -> {GRAPH_DIR}")

    git(f"checkout {head}")
    print(f"\nRestored HEAD to: {head[:12]}")
    print("DONE.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker-wholefile':
        _worker_wholefile(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()