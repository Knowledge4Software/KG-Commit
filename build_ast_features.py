#!/usr/bin/env python3
"""
Standalone AST feature extractor for the KG-Commit base snapshot (Family 2).

Why standalone + subprocess-per-file?
  javalang has catastrophic (exponential) blow-up on a few Java files
  (e.g. Groovyc.java). A pure-Python parse cannot be interrupted from inside
  the same process. The ONLY reliable way to bound it is to parse each file in
  a separate OS process and kill that process on timeout.

  We use subprocess.run(..., timeout=N), which on Windows reliably terminates
  the worker (and its tree) when the deadline passes. A hung file is killed
  and skipped; everything else proceeds.

Outputs (under outputs/):
  ast_subgraph_stats.csv     master per-method feature table (deliverable)
  ast_rows.json              same rows, typed, for the notebook to load
  ast_aggregates.json        group_totals + type_totals (for Viz 1)
  ast_example_graphs/*.pkl   ~15 small example AST graphs for visualization

Run:
  python build_ast_features.py
"""

import sys, csv, json, math, pickle, subprocess, tempfile, os
from pathlib import Path
from collections import defaultdict, Counter

# ── Configuration (mirrors the notebook) ────────────────────────────────────
PROJECT_ROOT = Path(r"c:\Users\sinab\Documents\GitHub\KG-Commit")
REPO_PATH    = PROJECT_ROOT / "repos" / "apache" / "groovy"
BASE_COMMIT  = "408b29851d7bbe4d343340832297e4be7e0c5578"

OUT_DIR      = PROJECT_ROOT / "outputs"
EX_GRAPH_DIR = OUT_DIR / "ast_example_graphs"

PARSE_TIMEOUT_S = 45        # per-file hard timeout
MAX_SRC_BYTES   = 600_000   # skip absurdly large files outright

# ── AST grouping (must match notebook cell a200b002) ────────────────────────
_AST_GROUP_MAP = {
    'MethodDeclaration': 'declaration', 'ConstructorDeclaration': 'declaration',
    'ClassDeclaration': 'declaration',  'InterfaceDeclaration': 'declaration',
    'EnumDeclaration': 'declaration',   'FieldDeclaration': 'declaration',
    'VariableDeclarator': 'declaration','FormalParameter': 'declaration',
    'LocalVariableDeclaration': 'declaration', 'EnumConstantDeclaration': 'declaration',
    'IfStatement': 'statement',         'ForStatement': 'statement',
    'EnhancedForControl': 'statement',  'ForControl': 'statement',
    'WhileStatement': 'statement',      'DoStatement': 'statement',
    'TryStatement': 'statement',        'CatchClause': 'statement',
    'SwitchStatement': 'statement',     'SwitchStatementCase': 'statement',
    'ReturnStatement': 'statement',     'ThrowStatement': 'statement',
    'BreakStatement': 'statement',      'ContinueStatement': 'statement',
    'AssertStatement': 'statement',     'StatementExpression': 'statement',
    'BlockStatement': 'statement',      'SynchronizedStatement': 'statement',
    'BinaryOperation': 'expression',    'MethodInvocation': 'expression',
    'SuperMethodInvocation': 'expression', 'Assignment': 'expression',
    'Cast': 'expression',               'TernaryExpression': 'expression',
    'ArrayCreator': 'expression',       'ClassCreator': 'expression',
    'ArrayAccess': 'expression',        'MemberReference': 'expression',
    'SuperMemberReference': 'expression','ArraySelector': 'expression',
    'Literal': 'literal',
    'BasicType': 'type',  'ReferenceType': 'type',
    'TypeArgument': 'type','TypeParameter': 'type',
    'Annotation': 'annotation','ElementValuePair': 'annotation',
    'ElementArrayValue': 'annotation',
    'Identifier': 'leaf',
}
AST_COLORS = {
    'declaration': '#2980B9', 'statement': '#E67E22', 'expression': '#27AE60',
    'literal': '#8E44AD', 'type': '#16A085', 'annotation': '#C0392B',
    'leaf': '#BDC3C7', 'other': '#95A5A6',
}

def ast_group(name):  return _AST_GROUP_MAP.get(name, 'other')
def ast_color(name):  return AST_COLORS.get(ast_group(name), AST_COLORS['other'])

def shannon_entropy(values):
    total = sum(values)
    if total == 0: return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs)


# ── AST traversal stats (no NetworkX) ───────────────────────────────────────
def traverse_method_stats(method_node, jlt):
    type_cnts, group_cnts = Counter(), Counter()
    depths, n_leaves = [], 0
    stack = [(method_node, 0)]
    while stack:
        item, depth = stack.pop()
        depths.append(depth)
        if isinstance(item, jlt.Node):
            t = type(item).__name__
            type_cnts[t] += 1
            group_cnts[ast_group(t)] += 1
            children = []
            for child in item.children:   # DIRECT children only
                if child is None: continue
                if isinstance(child, jlt.Node):
                    children.append((child, depth + 1))
                elif isinstance(child, (list, frozenset, set)):
                    for sub in child:
                        if isinstance(sub, jlt.Node):
                            children.append((sub, depth + 1))
                        elif isinstance(sub, str) and sub.strip():
                            children.append((sub, depth + 1))
                elif isinstance(child, str) and child.strip():
                    children.append((child, depth + 1))
            if children:
                stack.extend(reversed(children))
            else:
                n_leaves += 1
        else:
            type_cnts['Identifier'] += 1
            group_cnts['leaf'] += 1
            n_leaves += 1
    n      = len(depths)
    max_d  = max(depths, default=0)
    avg_d  = sum(depths) / max(1, n)
    n_int  = max(1, n - n_leaves)
    avg_br = (n - 1) / n_int
    leaf_r = n_leaves / max(1, n)
    return type_cnts, group_cnts, n, n - 1, max_d, round(avg_d, 2), round(avg_br, 2), round(leaf_r, 3)


def _enclosing_class(path_nodes, jlt):
    for anc in reversed(path_nodes):
        if isinstance(anc, (jlt.ClassDeclaration, jlt.InterfaceDeclaration,
                            jlt.EnumDeclaration, jlt.AnnotationDeclaration)):
            return anc.name
    return '<unknown>'


def _build_ast_subgraph(method_node, method_id, jlt, nx):
    Ag = nx.DiGraph(); ctr = [0]
    def make_nid():
        v = f"{method_id}::A{ctr[0]}"; ctr[0] += 1; return v
    root_nid = None
    stack = [(method_node, None, 0)]
    while stack:
        jval, parent_nid, child_pos = stack.pop()
        nid = make_nid()
        if parent_nid is None: root_nid = nid
        if isinstance(jval, jlt.Node):
            t = type(jval).__name__; vs = ''
            if getattr(jval, 'name', None):                 vs = str(jval.name)[:24]
            elif getattr(jval, 'value', None) is not None:  vs = str(jval.value)[:24]
            elif getattr(jval, 'operator', None):           vs = str(jval.operator)[:24]
            Ag.add_node(nid, ast_type=t, group=ast_group(t),
                        color=ast_color(t), value=vs, is_leaf=False)
            if parent_nid is not None:
                Ag.add_edge(parent_nid, nid, rel='AST_CHILD', pos=child_pos)
            children = []; cp = 0
            for child in jval.children:   # DIRECT children only
                if child is None: continue
                if isinstance(child, jlt.Node):
                    children.append((child, nid, cp)); cp += 1
                elif isinstance(child, (list, frozenset, set)):
                    for it in child:
                        if isinstance(it, jlt.Node):
                            children.append((it, nid, cp)); cp += 1
                        elif isinstance(it, str) and it.strip():
                            children.append((it, nid, cp)); cp += 1
                elif isinstance(child, str) and child.strip():
                    children.append((child, nid, cp)); cp += 1
            stack.extend(reversed(children))
        else:
            val = jval[:24] if isinstance(jval, str) else str(jval)[:24]
            Ag.add_node(nid, ast_type='Identifier', group='leaf',
                        color=AST_COLORS['leaf'], value=val, is_leaf=True)
            if parent_nid is not None:
                Ag.add_edge(parent_nid, nid, rel='AST_CHILD', pos=child_pos)
    return Ag, root_nid


# ═════════════════════════════════════════════════════════════════════════
#  WORKER MODES (run as a child via subprocess; killed on timeout by parent)
# ═════════════════════════════════════════════════════════════════════════
def _worker_file(abs_path, rel_path, out_path):
    import javalang, javalang.tree as jlt
    src  = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    tree = javalang.parse.parse(src)
    has_pkg = any(isinstance(n, jlt.PackageDeclaration) for _, n in tree)

    rows = []
    gtot, ttot = Counter(), Counter()
    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class(path_nodes, jlt)
        kind = 'constructor' if isinstance(node, jlt.ConstructorDeclaration) else 'method'
        mid  = f"{rel_path}::{cls}::{node.name}@{node.position.line}"
        tc, gc2, n, n_e, max_d, avg_d, avg_br, leaf_r = traverse_method_stats(node, jlt)
        gtot += gc2; ttot += tc
        def pct(g): return round(100 * gc2.get(g, 0) / max(1, n), 1)
        rows.append({
            'method_id': mid, 'file': rel_path,
            'class': cls, 'method': node.name, 'kind': kind,
            'n_ast_nodes': n, 'n_edges': n_e,
            'max_depth': max_d, 'avg_depth': avg_d, 'avg_branching': avg_br,
            'n_node_types': len(tc), 'n_groups': len(gc2),
            'type_entropy':  round(shannon_entropy(list(tc.values())), 4),
            'group_entropy': round(shannon_entropy(list(gc2.values())), 4),
            'leaf_ratio': leaf_r,
            'pct_declaration': pct('declaration'), 'pct_statement': pct('statement'),
            'pct_expression': pct('expression'),   'pct_literal': pct('literal'),
            'pct_leaf': pct('leaf'),
            'has_method_link': True,
            'has_class_ancestor': cls != '<unknown>',
            'has_file_ancestor': True,
            'has_pkg_ancestor': bool(has_pkg),
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'gtot': dict(gtot), 'ttot': dict(ttot)}, f)


def _worker_graph(abs_path, rel_path, wanted_path, out_path):
    import javalang, javalang.tree as jlt, networkx as nx
    wanted = json.load(open(wanted_path, encoding='utf-8'))  # list of [cls, method, kind, mid]
    src  = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    tree = javalang.parse.parse(src)

    parsed = defaultdict(list)
    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class(path_nodes, jlt)
        kind = 'constructor' if isinstance(node, jlt.ConstructorDeclaration) else 'method'
        parsed[(cls, node.name, kind)].append((node.position.line, node))
    for k in parsed: parsed[k].sort(key=lambda t: t[0])

    wanted_by_key = defaultdict(list)
    for cls, meth, kind, mid in wanted:
        wanted_by_key[(cls, meth, kind)].append(mid)

    saved = []
    for key, mids in wanted_by_key.items():
        cands = parsed.get(key, [])
        for i, mid in enumerate(mids):
            if i >= len(cands): continue
            _, jnode = cands[i]
            Ag, root = _build_ast_subgraph(jnode, mid, jlt, nx)
            safe = "".join(c if c.isalnum() else "_" for c in mid)
            pf = EX_GRAPH_DIR / f"{safe}.pkl"
            pf.write_bytes(pickle.dumps((Ag, root), protocol=4))
            saved.append(mid)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'saved': saved}, f)


# ── Parent: run a worker mode in a killable subprocess ──────────────────────
def run_worker(mode, args, timeout):
    """Returns parsed JSON dict on success, or a status string on failure."""
    fd, out_path = tempfile.mkstemp(suffix='.json'); os.close(fd)
    cmd = [sys.executable, os.path.abspath(__file__), mode, *args, out_path]
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


def git(cmd):
    return subprocess.run(["git"] + cmd.split(), cwd=str(REPO_PATH),
                          capture_output=True, text=True).stdout.strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EX_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    for old in EX_GRAPH_DIR.glob('*.pkl'):
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

        res = run_worker('--worker-file', [str(fp), rel], PARSE_TIMEOUT_S)
        if isinstance(res, dict):
            all_rows.extend(res['rows'])
            group_totals += Counter(res['gtot'])
            type_totals  += Counter(res['ttot'])
            ok_files.append(rel)
            print(f"{tag}  ok ({len(res['rows'])} methods)", flush=True)
        elif res == 'TIMEOUT':
            print(f"{tag}  KILLED (parse > {PARSE_TIMEOUT_S}s)", flush=True)
            skipped_files.append(rel)
        else:
            print(f"{tag}  {res}", flush=True)
            skipped_files.append(rel)

    print(f"\nParsed OK : {len(ok_files)} files,  {len(all_rows)} methods", flush=True)
    print(f"Skipped   : {len(skipped_files)} files", flush=True)

    if not all_rows:
        print("No methods extracted — aborting."); git(f"checkout {head}"); return

    cols = list(all_rows[0].keys())
    with open(OUT_DIR / "ast_subgraph_stats.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)
    with open(OUT_DIR / "ast_rows.json", 'w', encoding='utf-8') as f:
        json.dump(all_rows, f)
    with open(OUT_DIR / "ast_aggregates.json", 'w', encoding='utf-8') as f:
        json.dump({'group_totals': dict(group_totals), 'type_totals': dict(type_totals),
                   'n_methods': len(all_rows), 'ok_files': len(ok_files),
                   'skipped_files': skipped_files}, f)

    n_vals = [r['n_ast_nodes'] for r in all_rows]
    d_vals = [r['max_depth']   for r in all_rows]
    print(f"\nAST size  : min={min(n_vals)} med={sorted(n_vals)[len(n_vals)//2]} "
          f"mean={sum(n_vals)/len(n_vals):.0f} max={max(n_vals)}")
    print(f"Max depth : min={min(d_vals)} med={sorted(d_vals)[len(d_vals)//2]} "
          f"mean={sum(d_vals)/len(d_vals):.1f} max={max(d_vals)}")
    print(f"Top groups: {dict(group_totals.most_common(6))}")

    # ── Phase 2: example graphs from files that parsed OK ───────────────────
    ok_set = set(ok_files)
    viz_ids = []
    for r in sorted([r for r in all_rows
                     if 8 <= r['n_ast_nodes'] <= 55 and r['file'] in ok_set],
                    key=lambda r: -r['type_entropy'])[:6]:
        viz_ids.append(r['method_id'])
    cc = Counter(r['class'] for r in all_rows
                 if 6 <= r['n_ast_nodes'] <= 150 and r['file'] in ok_set)
    if cc:
        tcls = cc.most_common(1)[0][0]
        for r in [r for r in all_rows if r['class'] == tcls
                  and 6 <= r['n_ast_nodes'] <= 150 and r['file'] in ok_set][:9]:
            if r['method_id'] not in viz_ids:
                viz_ids.append(r['method_id'])

    by_mid = {r['method_id']: r for r in all_rows}
    file_to_wanted = defaultdict(list)
    for mid in viz_ids:
        r = by_mid[mid]
        file_to_wanted[r['file']].append([r['class'], r['method'], r['kind'], mid])

    print(f"\nBuilding {len(viz_ids)} example graphs from {len(file_to_wanted)} files...", flush=True)
    built = 0
    for rel, wanted in file_to_wanted.items():
        fd, wp = tempfile.mkstemp(suffix='.json'); os.close(fd)
        with open(wp, 'w', encoding='utf-8') as f: json.dump(wanted, f)
        res = run_worker('--worker-graph', [str(REPO_PATH / rel), rel, wp], PARSE_TIMEOUT_S)
        try: os.unlink(wp)
        except OSError: pass
        if isinstance(res, dict):
            built += len(res['saved'])
        else:
            print(f"  graph build failed for {rel}: {res}", flush=True)
    print(f"Example graphs saved : {built}  -> {EX_GRAPH_DIR}", flush=True)

    git(f"checkout {head}")
    print(f"\nRestored HEAD to: {head[:12]}")
    print("DONE.")


if __name__ == '__main__':
    # Worker dispatch: parent re-invokes this script with a --worker-* mode.
    if len(sys.argv) > 1 and sys.argv[1] == '--worker-file':
        _worker_file(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 1 and sys.argv[1] == '--worker-graph':
        _worker_graph(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
