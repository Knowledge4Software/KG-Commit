#!/usr/bin/env python3
"""
Standalone Data-Flow-Graph (DFG) feature extractor for KG-Commit base snapshot
— Family 4 (Data-Flow Representations).

Same offline, killable-subprocess architecture as build_ast/cfg_features.py.
For every concrete method we build an intra-procedural def-use graph over LOCAL
variables (parameters + locals): DEF nodes (definitions) and USE nodes (reads),
connected by DDG_REACHES edges (a definition reaches a use). We use a simple
reaching-definitions walk (gen/kill, branch-union; loops single-pass — a
source-level approximation, per the design doc).

IMPORTANT: javalang `for x in node` walks the WHOLE subtree; use node.children
for direct children.

Outputs (under outputs/):
  dfg_subgraph_stats.csv     master per-method DFG feature table (deliverable)
  dfg_rows.json              typed rows for the notebook
  dfg_aggregates.json        node-group / def-type totals + var stats
  dfg_example_graphs/*.pkl   ~12 example def-use graphs for visualization

Run:  python build_dfg_features.py
"""

import sys, csv, json, math, pickle, subprocess, tempfile, os
from pathlib import Path
from collections import defaultdict, Counter

# per-project config; in this package only DefUseBuilder is used (via subgraph_builders.py)
import _kgc_paths  # noqa: F401
from config.project_config import PROJECT_ROOT, REPO_PATH, BASE_COMMIT, OUT
OUT_DIR      = OUT
EX_GRAPH_DIR = OUT_DIR / "dfg_example_graphs"
PARSE_TIMEOUT_S = 45
MAX_SRC_BYTES   = 600_000

# def-type colours / groups (for visualization)
DFG_COLORS = {'def': '#C0392B', 'use': '#2980B9', 'other': '#95A5A6'}
DEFTYPE_COLORS = {
    'param':   '#16A085', 'decl': '#2980B9', 'assign': '#E67E22',
    'update':  '#8E44AD', 'forvar': '#27AE60', 'catchvar': '#C0392B',
}

def shannon_entropy(values):
    total = sum(values)
    if total == 0: return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs)


# ═════════════════════════════════════════════════════════════════════════
#  DEF-USE (DATA-FLOW) BUILDER
# ═════════════════════════════════════════════════════════════════════════
class DefUseBuilder:
    def __init__(self, nx_mod, jlt):
        self.nx = nx_mod
        self.jlt = jlt
        self.G = nx_mod.DiGraph()
        self.ctr = 0
        self.defined = set()     # locally-defined variable names (pass 1)

    # ---- pass 1: collect names that are ever defined locally ----
    def collect_defined(self, method_node):
        jlt = self.jlt
        for p in (method_node.parameters or []):
            if getattr(p, 'name', None):
                self.defined.add(p.name)
        stack = [method_node]
        while stack:
            node = stack.pop()
            if isinstance(node, jlt.VariableDeclarator) and node.name:
                self.defined.add(node.name)
            elif isinstance(node, jlt.Assignment):
                lhs = node.expressionl
                if isinstance(lhs, jlt.MemberReference) and not lhs.qualifier and lhs.member:
                    self.defined.add(lhs.member)
            elif isinstance(node, jlt.MemberReference):
                if (not node.qualifier and node.member and
                        (node.prefix_operators or node.postfix_operators)):
                    self.defined.add(node.member)
            for child in node.children:
                if child is None: continue
                if isinstance(child, jlt.Node):
                    stack.append(child)
                elif isinstance(child, (list, set, frozenset)):
                    for it in child:
                        if isinstance(it, jlt.Node):
                            stack.append(it)

    # ---- node helpers ----
    def _line(self, node, fallback=None):
        p = getattr(node, 'position', None)
        return p.line if p is not None else fallback

    def make_def(self, var, def_type, line):
        nid = f"D{self.ctr}"; self.ctr += 1
        self.G.add_node(nid, kind='DEF', group='def', var=var,
                        def_type=def_type, line=line if line is not None else -1,
                        color=DFG_COLORS['def'])
        return nid

    def make_use(self, var, line, reaching):
        nid = f"U{self.ctr}"; self.ctr += 1
        self.G.add_node(nid, kind='USE', group='use', var=var,
                        def_type='', line=line if line is not None else -1,
                        color=DFG_COLORS['use'])
        for d in reaching.get(var, ()):       # connect all reaching defs
            self.G.add_edge(d, nid, rel='DDG_REACHES', var=var)
        return nid

    # ---- extract variable reads from an expression subtree ----
    def uses_in(self, expr, line):
        jlt = self.jlt
        if not isinstance(expr, jlt.Node): return []
        out = []
        stack = [expr]
        while stack:
            node = stack.pop()
            if not isinstance(node, jlt.Node):
                continue
            if isinstance(node, jlt.MemberReference):
                if node.member in self.defined and not node.qualifier:
                    out.append((node.member, self._line(node, line)))
                if isinstance(node.qualifier, str) and node.qualifier in self.defined:
                    out.append((node.qualifier, self._line(node, line)))
            for child in node.children:
                if child is None: continue
                if isinstance(child, jlt.Node):
                    stack.append(child)
                elif isinstance(child, (list, set, frozenset)):
                    for it in child:
                        if isinstance(it, jlt.Node):
                            stack.append(it)
        return out

    def emit_uses(self, expr, line, reaching):
        for var, ln in self.uses_in(expr, line):
            self.make_use(var, ln, reaching)

    @staticmethod
    def _copy(reaching):
        return {k: set(v) for k, v in reaching.items()}

    @staticmethod
    def _merge(a, b):
        out = {k: set(v) for k, v in a.items()}
        for k, v in b.items():
            out.setdefault(k, set()).update(v)
        return out

    # ---- pass 2: reaching-definitions walk ----
    def process_seq(self, stmts, reaching):
        for st in self._stmts(stmts):
            reaching = self.process_stmt(st, reaching)
        return reaching

    def _stmts(self, s):
        if s is None: return []
        if isinstance(s, self.jlt.BlockStatement): return s.statements or []
        if isinstance(s, list): return s
        return [s]

    def _assign(self, asg, reaching, line):
        """Handle an Assignment node; returns updated reaching."""
        jlt = self.jlt
        compound = asg.type not in ('=', None)
        self.emit_uses(asg.value, line, reaching)          # RHS reads
        lhs = asg.expressionl
        if isinstance(lhs, jlt.MemberReference) and not lhs.qualifier and lhs.member in self.defined:
            if compound:
                self.make_use(lhs.member, self._line(lhs, line), reaching)
            d = self.make_def(lhs.member, 'assign', self._line(lhs, line))
            reaching = self._copy(reaching); reaching[lhs.member] = {d}
        else:
            self.emit_uses(lhs, line, reaching)            # e.g. arr[i] = ...
        return reaching

    def process_stmt(self, st, reaching):
        jlt = self.jlt
        line = self._line(st)

        if isinstance(st, jlt.LocalVariableDeclaration):
            for d in st.declarators:
                self.emit_uses(d.initializer, line, reaching)
                nid = self.make_def(d.name, 'decl', self._line(d, line))
                reaching = self._copy(reaching); reaching[d.name] = {nid}
            return reaching

        if isinstance(st, jlt.StatementExpression):
            e = st.expression
            if isinstance(e, jlt.Assignment):
                return self._assign(e, reaching, line)
            # ++/-- as a statement: use + def
            if isinstance(e, jlt.MemberReference) and not e.qualifier and \
               e.member in self.defined and (e.prefix_operators or e.postfix_operators):
                self.make_use(e.member, line, reaching)
                nid = self.make_def(e.member, 'update', line)
                reaching = self._copy(reaching); reaching[e.member] = {nid}
                return reaching
            self.emit_uses(e, line, reaching)
            return reaching

        if isinstance(st, jlt.IfStatement):
            self.emit_uses(st.condition, line, reaching)
            r_then = self.process_seq(st.then_statement, self._copy(reaching))
            if st.else_statement is not None:
                r_else = self.process_seq(st.else_statement, self._copy(reaching))
                return self._merge(r_then, r_else)
            return self._merge(reaching, r_then)

        if isinstance(st, (jlt.WhileStatement,)):
            self.emit_uses(st.condition, line, reaching)
            r_body = self.process_seq(st.body, self._copy(reaching))
            return self._merge(reaching, r_body)

        if isinstance(st, jlt.DoStatement):
            r_body = self.process_seq(st.body, self._copy(reaching))
            self.emit_uses(st.condition, line, r_body)
            return self._merge(reaching, r_body)

        if isinstance(st, jlt.ForStatement):
            ctrl = st.control
            if isinstance(ctrl, jlt.EnhancedForControl):
                self.emit_uses(ctrl.iterable, line, reaching)
                var = getattr(ctrl, 'var', None)
                if var is not None:
                    for d in getattr(var, 'declarators', []) or []:
                        nid = self.make_def(d.name, 'forvar', line)
                        reaching = self._copy(reaching); reaching[d.name] = {nid}
                r_body = self.process_seq(st.body, self._copy(reaching))
                return self._merge(reaching, r_body)
            else:  # ForControl: init ; condition ; update
                if ctrl is not None and ctrl.init:
                    init = ctrl.init
                    if isinstance(init, list):
                        for x in init:
                            reaching = self.process_stmt(x, reaching) \
                                if isinstance(x, jlt.Statement) else \
                                self._for_init_expr(x, reaching, line)
                    elif isinstance(init, jlt.Statement):
                        reaching = self.process_stmt(init, reaching)
                    else:
                        reaching = self._for_init_expr(init, reaching, line)
                if ctrl is not None and ctrl.condition is not None:
                    self.emit_uses(ctrl.condition, line, reaching)
                r_body = self.process_seq(st.body, self._copy(reaching))
                if ctrl is not None and ctrl.update:
                    for u in ctrl.update:
                        r_body = self._for_update_expr(u, r_body, line)
                return self._merge(reaching, r_body)

        if isinstance(st, jlt.SwitchStatement):
            self.emit_uses(st.expression, line, reaching)
            for case in (st.cases or []):
                reaching = self.process_seq(case.statements, reaching)
            return reaching

        if isinstance(st, jlt.TryStatement):
            for res in (getattr(st, 'resources', None) or []):
                reaching = self.process_stmt(res, reaching) \
                    if isinstance(res, jlt.Statement) else reaching
            r = self.process_seq(st.block, self._copy(reaching))
            outs = [r]
            for catch in (st.catches or []):
                cr = self._copy(reaching)
                param = getattr(catch, 'parameter', None)
                if param is not None and getattr(param, 'name', None):
                    nid = self.make_def(param.name, 'catchvar', self._line(catch, line))
                    cr[param.name] = {nid}
                outs.append(self.process_seq(catch.block, cr))
            merged = outs[0]
            for o in outs[1:]:
                merged = self._merge(merged, o)
            fin = getattr(st, 'finally_block', None)
            if fin:
                merged = self.process_seq(fin, merged)
            return merged

        if isinstance(st, jlt.SynchronizedStatement):
            self.emit_uses(st.lock, line, reaching)
            return self.process_seq(st.block, reaching)

        if isinstance(st, jlt.BlockStatement):
            return self.process_seq(st.statements, reaching)

        if isinstance(st, (jlt.ReturnStatement, jlt.ThrowStatement)):
            self.emit_uses(getattr(st, 'expression', None), line, reaching)
            return reaching

        # other statements: best-effort use extraction
        for attr in getattr(st, 'attrs', []):
            self.emit_uses(getattr(st, attr, None), line, reaching)
        return reaching

    def _for_init_expr(self, expr, reaching, line):
        jlt = self.jlt
        if isinstance(expr, jlt.Assignment):
            return self._assign(expr, reaching, line)
        self.emit_uses(expr, line, reaching)
        return reaching

    def _for_update_expr(self, expr, reaching, line):
        jlt = self.jlt
        if isinstance(expr, jlt.Assignment):
            return self._assign(expr, reaching, line)
        if isinstance(expr, jlt.MemberReference) and not expr.qualifier and \
           expr.member in self.defined:
            self.make_use(expr.member, line, reaching)
            nid = self.make_def(expr.member, 'update', line)
            reaching = self._copy(reaching); reaching[expr.member] = {nid}
            return reaching
        self.emit_uses(expr, line, reaching)
        return reaching

    def build(self, method_node):
        self.collect_defined(method_node)
        reaching = {}
        for p in (method_node.parameters or []):
            if getattr(p, 'name', None):
                nid = self.make_def(p.name, 'param', self._line(p))
                reaching[p.name] = {nid}
        self.process_seq(method_node.body, reaching)
        return self.G


def _enclosing_class(path_nodes, jlt):
    for anc in reversed(path_nodes):
        if isinstance(anc, (jlt.ClassDeclaration, jlt.InterfaceDeclaration,
                            jlt.EnumDeclaration, jlt.AnnotationDeclaration)):
            return anc.name
    return '<unknown>'


def _dfg_stats(G):
    defs = [n for n, d in G.nodes(data=True) if d['kind'] == 'DEF']
    uses = [n for n, d in G.nodes(data=True) if d['kind'] == 'USE']
    n_def, n_use = len(defs), len(uses)
    edges = G.number_of_edges()
    variables = set(d['var'] for _, d in G.nodes(data=True))
    deftypes = Counter(G.nodes[n]['def_type'] for n in defs)
    upd = [G.out_degree(d) for d in defs]
    dpu = [G.in_degree(u) for u in uses]
    unused = sum(1 for d in defs if G.out_degree(d) == 0)
    undef  = sum(1 for u in uses if G.in_degree(u) == 0)
    var_use_counts = Counter()
    for u in uses:
        var_use_counts[G.nodes[u]['var']] += 1
    return {
        'n_def_nodes': n_def, 'n_use_nodes': n_use,
        'n_dfg_nodes': n_def + n_use, 'n_du_edges': edges,
        'n_variables': len(variables),
        'n_params': deftypes.get('param', 0),
        'n_unused_defs': unused, 'n_undef_uses': undef,
        'max_uses_per_def': max(upd, default=0),
        'avg_uses_per_def': round(sum(upd) / n_def, 2) if n_def else 0.0,
        'max_defs_per_use': max(dpu, default=0),
        'du_density': round(edges / max(1, n_def * n_use), 4) if (n_def and n_use) else 0.0,
        'var_entropy': round(shannon_entropy(list(var_use_counts.values())), 4),
        'deftypes': dict(deftypes),
    }


# ═════════════════════════════════════════════════════════════════════════
#  WORKER MODES
# ═════════════════════════════════════════════════════════════════════════
def _worker_file(abs_path, rel_path, out_path):
    import javalang, javalang.tree as jlt, networkx as nx
    src  = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    tree = javalang.parse.parse(src)
    has_pkg = any(isinstance(n, jlt.PackageDeclaration) for _, n in tree)

    rows = []
    group_totals, deftype_totals = Counter(), Counter()
    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class(path_nodes, jlt)
        kind = 'constructor' if isinstance(node, jlt.ConstructorDeclaration) else 'method'
        mid  = f"{rel_path}::{cls}::{node.name}@{node.position.line}"

        b = DefUseBuilder(nx, jlt)
        G = b.build(node)
        st = _dfg_stats(G)
        group_totals['def'] += st['n_def_nodes']
        group_totals['use'] += st['n_use_nodes']
        deftype_totals += Counter(st.pop('deftypes'))

        rows.append({
            'method_id': mid, 'file': rel_path,
            'class': cls, 'method': node.name, 'kind': kind,
            **st,
            'has_method_link': True,
            'has_class_ancestor': cls != '<unknown>',
            'has_file_ancestor': True,
            'has_pkg_ancestor': bool(has_pkg),
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'group_totals': dict(group_totals),
                   'deftype_totals': dict(deftype_totals)}, f)


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
            b = DefUseBuilder(nx, jlt)
            G = b.build(jnode)
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
    for old in EX_GRAPH_DIR.glob('*.pkl'): old.unlink()

    head = git("rev-parse HEAD")
    print(f"Current HEAD : {head[:12]}", flush=True)
    git(f"checkout {BASE_COMMIT}")
    print(f"Checked out  : {git('rev-parse HEAD')[:12]}\n", flush=True)

    java_files = sorted(REPO_PATH.rglob("*.java"))
    n_files = len(java_files)
    print(f"Found {n_files} .java files\n", flush=True)

    all_rows = []
    group_totals, deftype_totals = Counter(), Counter()
    ok_files, skipped_files = [], []

    for i, fp in enumerate(java_files, 1):
        rel = fp.relative_to(REPO_PATH).as_posix()
        tag = f"[{i}/{n_files}] {rel}"
        try: size = fp.stat().st_size
        except OSError: size = 0
        if size > MAX_SRC_BYTES:
            print(f"{tag}  SKIP (too large)", flush=True)
            skipped_files.append(rel); continue
        res = run_worker('--worker-file', [str(fp), rel], PARSE_TIMEOUT_S)
        if isinstance(res, dict):
            all_rows.extend(res['rows'])
            group_totals += Counter(res['group_totals'])
            deftype_totals += Counter(res['deftype_totals'])
            ok_files.append(rel)
            print(f"{tag}  ok ({len(res['rows'])} methods)", flush=True)
        elif res == 'TIMEOUT':
            print(f"{tag}  KILLED (>{PARSE_TIMEOUT_S}s)", flush=True)
            skipped_files.append(rel)
        else:
            print(f"{tag}  {res}", flush=True)
            skipped_files.append(rel)

    print(f"\nParsed OK : {len(ok_files)} files,  {len(all_rows)} methods", flush=True)
    print(f"Skipped   : {len(skipped_files)} files", flush=True)
    if not all_rows:
        print("No methods — aborting."); git(f"checkout {head}"); return

    col_order = ['method_id','file','class','method','kind',
                 'n_dfg_nodes','n_def_nodes','n_use_nodes','n_du_edges',
                 'n_variables','n_params','n_unused_defs','n_undef_uses',
                 'max_uses_per_def','avg_uses_per_def','max_defs_per_use',
                 'du_density','var_entropy',
                 'has_method_link','has_class_ancestor','has_file_ancestor','has_pkg_ancestor']
    with open(OUT_DIR / "dfg_subgraph_stats.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=col_order); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k, '') for k in col_order})
    with open(OUT_DIR / "dfg_rows.json", 'w', encoding='utf-8') as f:
        json.dump(all_rows, f)
    with open(OUT_DIR / "dfg_aggregates.json", 'w', encoding='utf-8') as f:
        json.dump({'group_totals': dict(group_totals),
                   'deftype_totals': dict(deftype_totals),
                   'n_methods': len(all_rows), 'ok_files': len(ok_files),
                   'skipped_files': skipped_files}, f)

    edg = [r['n_du_edges'] for r in all_rows]
    nv  = [r['n_variables'] for r in all_rows]
    print(f"\nDef-use edges : min={min(edg)} med={sorted(edg)[len(edg)//2]} "
          f"mean={sum(edg)/len(edg):.1f} max={max(edg)}")
    print(f"Variables/meth: min={min(nv)} med={sorted(nv)[len(nv)//2]} "
          f"mean={sum(nv)/len(nv):.1f} max={max(nv)}")
    print(f"Node groups   : {dict(group_totals)}")
    print(f"Def types     : {dict(deftype_totals.most_common())}")
    print(f"Total unused defs : {sum(r['n_unused_defs'] for r in all_rows)}")

    # ── Phase 2: example def-use graphs ─────────────────────────────────────
    ok_set = set(ok_files)
    viz_ids = []
    for r in sorted([r for r in all_rows
                     if 6 <= r['n_dfg_nodes'] <= 40 and r['file'] in ok_set
                     and r['n_du_edges'] >= 3 and r['n_variables'] >= 2],
                    key=lambda r: -r['n_du_edges'])[:6]:
        viz_ids.append(r['method_id'])
    cc = Counter(r['class'] for r in all_rows
                 if 5 <= r['n_dfg_nodes'] <= 45 and r['file'] in ok_set
                 and r['n_du_edges'] >= 2)
    if cc:
        tcls = cc.most_common(1)[0][0]
        for r in [r for r in all_rows if r['class'] == tcls
                  and 5 <= r['n_dfg_nodes'] <= 45 and r['file'] in ok_set
                  and r['n_du_edges'] >= 2][:6]:
            if r['method_id'] not in viz_ids:
                viz_ids.append(r['method_id'])

    by_mid = {r['method_id']: r for r in all_rows}
    file_to_wanted = defaultdict(list)
    for mid in viz_ids:
        r = by_mid[mid]
        file_to_wanted[r['file']].append([r['class'], r['method'], r['kind'], mid])

    print(f"\nBuilding {len(viz_ids)} example DFGs from {len(file_to_wanted)} files...", flush=True)
    built = 0
    for rel, wanted in file_to_wanted.items():
        fd, wp = tempfile.mkstemp(suffix='.json'); os.close(fd)
        with open(wp, 'w', encoding='utf-8') as f: json.dump(wanted, f)
        res = run_worker('--worker-graph', [str(REPO_PATH / rel), rel, wp], PARSE_TIMEOUT_S)
        try: os.unlink(wp)
        except OSError: pass
        if isinstance(res, dict): built += len(res['saved'])
        else: print(f"  graph build failed for {rel}: {res}", flush=True)
    print(f"Example DFGs saved : {built}  -> {EX_GRAPH_DIR}", flush=True)

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
