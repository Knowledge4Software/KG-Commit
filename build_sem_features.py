#!/usr/bin/env python3
"""
Standalone Semantic-analysis feature extractor for KG-Commit base snapshot
— Family 6 (Formal and Semantic Analysis).

Family 6 spans abstract interpretation, alias/points-to, taint, type systems,
symbolic execution, and formal contracts. The production tools for the heavy
variants (Infer, Soot/WALA alias, SMT symbolic execution) are NOT run here —
instead we compute the cheap, source-level SUBSET the design doc itself
recommends for KG-Commit ("a restricted set of cheap domains: nullness + taint"
+ contract mining), entirely from the javalang AST:

  * NULLNESS abstract-interpretation proxy: null literals/assignments/returns,
    null-checks, dereferences, and UNGUARDED dereferences (NPE-risk) of
    nullable/parameter variables  -> per-method nullness graph (the visualised
    artifact: VAR / DEREF / CHECK / GUARD nodes, MAY_BE_NULL risk edges).
  * FORMAL CONTRACTS mining: requireNonNull / Preconditions.checkX / assert /
    if-null-throw precondition guards.
  * TYPE SYSTEM: @Nullable / @NonNull annotation usage.
  * TAINT (lightweight heuristic): source-like / sink-like call-site counts.
  * EFFECT TYPES: field-write / IO / purity flags.
  * SYMBOLIC-EXECUTION proxy: path-predicate count + estimated path count.

javalang `for x in node` walks the whole subtree (walk_tree); a single top-level
walk is O(n) and safe — we use it. Use node.children only for manual sub-walks.

Outputs (under outputs/):
  sem_subgraph_stats.csv     master per-method semantic feature table
  sem_rows.json              typed rows for the notebook
  sem_aggregates.json        signal totals + node-kind totals
  sem_example_graphs/*.pkl   ~12 example nullness graphs for visualization

Run:  python build_sem_features.py
"""

import sys, csv, json, math, pickle, subprocess, tempfile, os
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(r"c:\Users\sinab\Documents\GitHub\KG-Commit")
REPO_PATH    = PROJECT_ROOT / "repos" / "apache" / "groovy"
BASE_COMMIT  = "408b29851d7bbe4d343340832297e4be7e0c5578"
OUT_DIR      = PROJECT_ROOT / "outputs"
EX_GRAPH_DIR = OUT_DIR / "sem_example_graphs"
PARSE_TIMEOUT_S = 45
MAX_SRC_BYTES   = 600_000

SEM_COLORS = {
    'var':      '#2980B9',   # tracked variable
    'var_risk': '#C0392B',   # nullable + unguarded variable (NPE risk)
    'var_safe': '#27AE60',   # guarded / checked variable
    'deref':    '#7F8C8D',   # dereference site
    'check':    '#E67E22',   # null-check
    'guard':    '#16A085',   # precondition guard
    'other':    '#95A5A6',
}
GUARD_CALLS = {'requireNonNull', 'checkNotNull', 'checkArgument', 'checkState',
               'notNull', 'requireNonNullElse', 'isTrue', 'noNullElements',
               'checkPositionIndex', 'checkElementIndex'}
SOURCE_PREFIX = ('get', 'read', 'load', 'parse', 'fetch', 'receive', 'next', 'poll')
SOURCE_CONTAINS = ('Parameter', 'Property', 'Argument', 'Request', 'Input', 'Text')
SINK_PREFIX = ('set', 'write', 'exec', 'print', 'append', 'put', 'send', 'store', 'insert', 'update')
SINK_CONTAINS = ('execute', 'query', 'print', 'write', 'append', 'output')
IO_NAMES = {'print', 'println', 'write', 'read', 'flush', 'close', 'append',
            'readLine', 'newLine', 'getBytes'}

def shannon_entropy(values):
    total = sum(values)
    if total == 0: return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs)


# ═════════════════════════════════════════════════════════════════════════
#  SEMANTIC (NULLNESS + CONTRACT + EFFECT) ANALYZER
# ═════════════════════════════════════════════════════════════════════════
def _is_null_literal(node, jlt):
    return isinstance(node, jlt.Literal) and node.value == 'null'

def _local_var_of(ref, jlt, locals_):
    """If ref is a bare local-variable MemberReference, return its name."""
    if isinstance(ref, jlt.MemberReference) and not ref.qualifier and ref.member in locals_:
        return ref.member
    return None


def analyze(method_node, nx, jlt, build_graph=False):
    # ---- pass 1: local variable names (params + declared locals) ----
    params = [p.name for p in (method_node.parameters or []) if getattr(p, 'name', None)]
    locals_ = set(params)
    for _, n in method_node:
        if isinstance(n, jlt.VariableDeclarator) and n.name:
            locals_.add(n.name)

    # annotations on parameters
    n_nullable = n_nonnull = 0
    for p in (method_node.parameters or []):
        for a in (p.annotations or []):
            nm = (a.name or '').lower()
            if 'nullable' in nm: n_nullable += 1
            elif 'nonnull' in nm or 'notnull' in nm: n_nonnull += 1

    checked_vars, guarded_vars, nullable_vars = set(), set(), set()
    deref_sites = []           # (var, line)
    check_sites = []           # (var, line)
    guard_sites = []           # (var, line, gtype)
    n_null_literals = n_null_returns = n_null_assigns = 0
    n_taint_src = n_taint_sink = 0
    writes_field = does_io = False
    n_asserts = 0
    n_path_pred = 0

    def line_of(n):
        p = getattr(n, 'position', None)
        return p.line if p is not None else -1

    # ---- single O(n) walk over the method ----
    for path, node in method_node:
        parent = path[-1] if path else None

        if isinstance(node, jlt.Literal) and node.value == 'null':
            n_null_literals += 1
            if isinstance(parent, jlt.ReturnStatement):
                n_null_returns += 1

        elif isinstance(node, jlt.Assignment):
            if _is_null_literal(node.value, jlt):
                v = _local_var_of(node.expressionl, jlt, locals_)
                if v: nullable_vars.add(v); n_null_assigns += 1
            lhs = node.expressionl
            # field write: this.field (qualifier None) or bare name not a local
            if isinstance(lhs, jlt.MemberReference):
                if lhs.qualifier is None or (not lhs.qualifier and lhs.member not in locals_):
                    writes_field = True

        elif isinstance(node, jlt.VariableDeclarator):
            if _is_null_literal(getattr(node, 'initializer', None), jlt):
                nullable_vars.add(node.name); n_null_assigns += 1

        elif isinstance(node, jlt.BinaryOperation):
            if node.operator in ('==', '!='):
                for a, b in ((node.operandl, node.operandr), (node.operandr, node.operandl)):
                    if _is_null_literal(b, jlt):
                        v = _local_var_of(a, jlt, locals_)
                        if v:
                            checked_vars.add(v); check_sites.append((v, line_of(node)))
            if node.operator in ('&&', '||'):
                n_path_pred += 1

        elif isinstance(node, jlt.MethodInvocation):
            # dereference: x.foo()
            if isinstance(node.qualifier, str) and node.qualifier in locals_:
                deref_sites.append((node.qualifier, line_of(node)))
            # guard call: requireNonNull(x) / checkNotNull(x)
            if node.member in GUARD_CALLS:
                for arg in (node.arguments or []):
                    v = _local_var_of(arg, jlt, locals_)
                    if v:
                        guarded_vars.add(v); guard_sites.append((v, line_of(node), node.member))
            # taint heuristic
            m = node.member or ''
            if m.startswith(SOURCE_PREFIX) or any(c in m for c in SOURCE_CONTAINS):
                n_taint_src += 1
            if m.startswith(SINK_PREFIX) or any(c in m for c in SINK_CONTAINS):
                n_taint_sink += 1
            if m in IO_NAMES:
                does_io = True

        elif isinstance(node, jlt.MemberReference):
            # field dereference: x.field
            if isinstance(node.qualifier, str) and node.qualifier in locals_:
                deref_sites.append((node.qualifier, line_of(node)))

        elif isinstance(node, jlt.AssertStatement):
            n_asserts += 1

        elif isinstance(node, (jlt.IfStatement, jlt.WhileStatement, jlt.ForStatement,
                               jlt.DoStatement, jlt.TernaryExpression)):
            n_path_pred += 1

        elif isinstance(node, jlt.SwitchStatementCase):
            n_path_pred += 1

    # if-null-throw / if-null-return precondition guards
    for _, node in method_node:
        if isinstance(node, jlt.IfStatement):
            cond = node.condition
            v = None
            if isinstance(cond, jlt.BinaryOperation) and cond.operator in ('==', '!='):
                for a, b in ((cond.operandl, cond.operandr), (cond.operandr, cond.operandl)):
                    if _is_null_literal(b, jlt):
                        v = _local_var_of(a, jlt, locals_) or v
            if v:
                then = node.then_statement
                stmts = (then.statements if isinstance(then, jlt.BlockStatement) else [then]) if then else []
                if any(isinstance(s, (jlt.ThrowStatement, jlt.ReturnStatement)) for s in stmts):
                    guarded_vars.add(v); guard_sites.append((v, line_of(node), 'if-guard'))

    # NPE-risk: deref of a (param or null-assigned) var that is never checked/guarded
    safe_vars = checked_vars | guarded_vars
    risk_vars = set()
    n_unguarded = 0
    deref_vars = set(v for v, _ in deref_sites)
    for v, ln in deref_sites:
        if (v in params or v in nullable_vars) and v not in safe_vars:
            risk_vars.add(v); n_unguarded += 1

    n_preconditions = len([g for g in guard_sites])
    is_pure = (not writes_field) and (not does_io)
    est_paths = min(2 ** min(n_path_pred, 20), 1_048_576)

    stats = {
        'n_params': len(params), 'n_locals': len(locals_),
        'n_null_literals': n_null_literals, 'n_null_returns': n_null_returns,
        'n_null_assigns': n_null_assigns,
        'n_null_checks': len(check_sites), 'n_checked_vars': len(checked_vars),
        'n_guards': len(guard_sites), 'n_guarded_vars': len(guarded_vars),
        'n_preconditions': n_preconditions, 'n_asserts': n_asserts,
        'n_dereferences': len(deref_sites), 'n_deref_vars': len(deref_vars),
        'n_unguarded_derefs': n_unguarded, 'n_risk_vars': len(risk_vars),
        'n_nullable_annot': n_nullable, 'n_nonnull_annot': n_nonnull,
        'n_taint_sources': n_taint_src, 'n_taint_sinks': n_taint_sink,
        'writes_field': writes_field, 'does_io': does_io, 'is_pure': is_pure,
        'n_path_predicates': n_path_pred, 'est_paths': est_paths,
    }

    G = None
    if build_graph:
        G = nx.DiGraph()
        part_vars = deref_vars | checked_vars | guarded_vars | nullable_vars
        def vcolor(v):
            if v in risk_vars: return SEM_COLORS['var_risk']
            if v in safe_vars: return SEM_COLORS['var_safe']
            return SEM_COLORS['var']
        for v in part_vars:
            G.add_node(f"V::{v}", kind='VAR', group='var', label=v, color=vcolor(v),
                       is_param=(v in params), risk=(v in risk_vars), safe=(v in safe_vars))
        for i, (v, ln) in enumerate(deref_sites):
            nid = f"DR{i}"
            G.add_node(nid, kind='DEREF', group='deref', label='deref',
                       color=SEM_COLORS['deref'], line=ln)
            if f"V::{v}" in G:
                G.add_edge(f"V::{v}", nid, rel='DEREF_AT')
                if v in risk_vars:
                    G.add_edge(f"V::{v}", nid, rel='MAY_BE_NULL')
        for i, (v, ln) in enumerate(check_sites):
            nid = f"CK{i}"
            G.add_node(nid, kind='CHECK', group='check', label='==null',
                       color=SEM_COLORS['check'], line=ln)
            if f"V::{v}" in G: G.add_edge(nid, f"V::{v}", rel='CHECKS')
        for i, (v, ln, gt) in enumerate(guard_sites):
            nid = f"GD{i}"
            G.add_node(nid, kind='GUARD', group='guard', label=gt[:10],
                       color=SEM_COLORS['guard'], line=ln)
            if f"V::{v}" in G: G.add_edge(nid, f"V::{v}", rel='GUARDS')
        stats['n_sem_nodes'] = G.number_of_nodes()
        stats['n_sem_edges'] = G.number_of_edges()
    return stats, G


def _enclosing_class(path_nodes, jlt):
    for anc in reversed(path_nodes):
        if isinstance(anc, (jlt.ClassDeclaration, jlt.InterfaceDeclaration,
                            jlt.EnumDeclaration, jlt.AnnotationDeclaration)):
            return anc.name
    return '<unknown>'


# ═════════════════════════════════════════════════════════════════════════
#  WORKER MODES
# ═════════════════════════════════════════════════════════════════════════
def _worker_file(abs_path, rel_path, out_path):
    import javalang, javalang.tree as jlt, networkx as nx
    src  = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    tree = javalang.parse.parse(src)
    has_pkg = any(isinstance(n, jlt.PackageDeclaration) for _, n in tree)

    rows = []
    sig_totals, guard_types = Counter(), Counter()
    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class(path_nodes, jlt)
        kind = 'constructor' if isinstance(node, jlt.ConstructorDeclaration) else 'method'
        mid  = f"{rel_path}::{cls}::{node.name}@{node.position.line}"
        st, _ = analyze(node, nx, jlt, build_graph=False)
        for key in ('n_null_checks', 'n_guards', 'n_dereferences', 'n_unguarded_derefs',
                    'n_null_returns', 'n_taint_sources', 'n_taint_sinks', 'n_asserts'):
            sig_totals[key] += st[key]
        rows.append({
            'method_id': mid, 'file': rel_path,
            'class': cls, 'method': node.name, 'kind': kind, **st,
            'has_method_link': True, 'has_class_ancestor': cls != '<unknown>',
            'has_file_ancestor': True, 'has_pkg_ancestor': bool(has_pkg),
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'sig_totals': dict(sig_totals)}, f)


def _worker_graph(abs_path, rel_path, wanted_path, out_path):
    import javalang, javalang.tree as jlt, networkx as nx
    wanted = json.load(open(wanted_path, encoding='utf-8'))
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
            _, G = analyze(jnode, nx, jlt, build_graph=True)
            safe = "".join(c if c.isalnum() else "_" for c in mid)
            (EX_GRAPH_DIR / f"{safe}.pkl").write_bytes(pickle.dumps(G, protocol=4))
            saved.append(mid)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'saved': saved}, f)


def run_worker(mode, args, timeout):
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
        with open(out_path, encoding='utf-8') as f: data = json.load(f)
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
    for old in EX_GRAPH_DIR.glob('*.pkl'): old.unlink()

    head = git("rev-parse HEAD")
    print(f"Current HEAD : {head[:12]}", flush=True)
    git(f"checkout {BASE_COMMIT}")
    print(f"Checked out  : {git('rev-parse HEAD')[:12]}\n", flush=True)

    java_files = sorted(REPO_PATH.rglob("*.java"))
    n_files = len(java_files)
    print(f"Found {n_files} .java files\n", flush=True)

    all_rows = []
    sig_totals = Counter()
    ok_files, skipped_files = [], []
    for i, fp in enumerate(java_files, 1):
        rel = fp.relative_to(REPO_PATH).as_posix()
        tag = f"[{i}/{n_files}] {rel}"
        try: size = fp.stat().st_size
        except OSError: size = 0
        if size > MAX_SRC_BYTES:
            print(f"{tag}  SKIP (too large)", flush=True); skipped_files.append(rel); continue
        res = run_worker('--worker-file', [str(fp), rel], PARSE_TIMEOUT_S)
        if isinstance(res, dict):
            all_rows.extend(res['rows'])
            sig_totals += Counter(res['sig_totals'])
            ok_files.append(rel)
            print(f"{tag}  ok ({len(res['rows'])} methods)", flush=True)
        elif res == 'TIMEOUT':
            print(f"{tag}  KILLED (>{PARSE_TIMEOUT_S}s)", flush=True); skipped_files.append(rel)
        else:
            print(f"{tag}  {res}", flush=True); skipped_files.append(rel)

    print(f"\nParsed OK : {len(ok_files)} files,  {len(all_rows)} methods", flush=True)
    print(f"Skipped   : {len(skipped_files)} files", flush=True)
    if not all_rows:
        print("No methods — aborting."); git(f"checkout {head}"); return

    col_order = ['method_id','file','class','method','kind',
                 'n_params','n_locals','n_dereferences','n_deref_vars',
                 'n_null_literals','n_null_returns','n_null_assigns',
                 'n_null_checks','n_checked_vars','n_guards','n_guarded_vars',
                 'n_preconditions','n_asserts','n_unguarded_derefs','n_risk_vars',
                 'n_nullable_annot','n_nonnull_annot','n_taint_sources','n_taint_sinks',
                 'writes_field','does_io','is_pure','n_path_predicates','est_paths',
                 'has_method_link','has_class_ancestor','has_file_ancestor','has_pkg_ancestor']
    with open(OUT_DIR / "sem_subgraph_stats.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=col_order); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k, '') for k in col_order})
    with open(OUT_DIR / "sem_rows.json", 'w', encoding='utf-8') as f:
        json.dump(all_rows, f)
    with open(OUT_DIR / "sem_aggregates.json", 'w', encoding='utf-8') as f:
        json.dump({'sig_totals': dict(sig_totals),
                   'n_methods': len(all_rows), 'ok_files': len(ok_files),
                   'skipped_files': skipped_files}, f)

    npe = sum(r['n_unguarded_derefs'] for r in all_rows)
    pure = sum(1 for r in all_rows if r['is_pure'])
    print(f"\nDereferences      : {sum(r['n_dereferences'] for r in all_rows):,}")
    print(f"Null checks       : {sum(r['n_null_checks'] for r in all_rows):,}")
    print(f"Precondition guards: {sum(r['n_guards'] for r in all_rows):,}")
    print(f"Unguarded derefs (NPE-risk): {npe:,}")
    print(f"Null returns      : {sum(r['n_null_returns'] for r in all_rows):,}")
    print(f"Pure methods      : {pure:,} ({pure/len(all_rows)*100:.1f}%)")
    print(f"@Nullable/@NonNull annots: {sum(r['n_nullable_annot'] for r in all_rows)}/"
          f"{sum(r['n_nonnull_annot'] for r in all_rows)}")
    print(f"Taint src/sink    : {sum(r['n_taint_sources'] for r in all_rows):,}/"
          f"{sum(r['n_taint_sinks'] for r in all_rows):,}")

    # ── Phase 2: example nullness graphs (rich nullness structure) ──────────
    ok_set = set(ok_files)
    def score(r): return r['n_dereferences'] + 2 * r['n_unguarded_derefs'] + r['n_null_checks'] + r['n_guards']
    viz_ids = []
    for r in sorted([r for r in all_rows
                     if r['file'] in ok_set and (r['n_dereferences'] + r['n_null_checks'] + r['n_guards']) >= 3
                     and r['n_deref_vars'] >= 1],
                    key=lambda r: -score(r))[:6]:
        viz_ids.append(r['method_id'])
    cc = Counter(r['class'] for r in all_rows
                 if r['file'] in ok_set and (r['n_dereferences'] + r['n_null_checks']) >= 2)
    if cc:
        tcls = cc.most_common(1)[0][0]
        for r in sorted([r for r in all_rows if r['class'] == tcls and r['file'] in ok_set
                         and (r['n_dereferences'] + r['n_null_checks'] + r['n_guards']) >= 2],
                        key=lambda r: -score(r))[:6]:
            if r['method_id'] not in viz_ids: viz_ids.append(r['method_id'])

    by_mid = {r['method_id']: r for r in all_rows}
    file_to_wanted = defaultdict(list)
    for mid in viz_ids:
        r = by_mid[mid]
        file_to_wanted[r['file']].append([r['class'], r['method'], r['kind'], mid])

    print(f"\nBuilding {len(viz_ids)} example nullness graphs from {len(file_to_wanted)} files...", flush=True)
    built = 0
    for rel, wanted in file_to_wanted.items():
        fd, wp = tempfile.mkstemp(suffix='.json'); os.close(fd)
        with open(wp, 'w', encoding='utf-8') as f: json.dump(wanted, f)
        res = run_worker('--worker-graph', [str(REPO_PATH / rel), rel, wp], PARSE_TIMEOUT_S)
        try: os.unlink(wp)
        except OSError: pass
        if isinstance(res, dict): built += len(res['saved'])
        else: print(f"  graph build failed for {rel}: {res}", flush=True)
    print(f"Example nullness graphs saved : {built}  -> {EX_GRAPH_DIR}", flush=True)

    git(f"checkout {head}")
    print(f"\nRestored HEAD to: {head[:12]}")
    print("DONE.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker-file':
        _worker_file(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 1 and sys.argv[1] == '--worker-graph':
        _worker_graph(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
