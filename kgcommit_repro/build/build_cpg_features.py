#!/usr/bin/env python3
"""
Standalone PDG/CPG feature extractor for KG-Commit base snapshot
— Family 5 (Unified Dependence Representations: PDG / SDG / CPG).

A per-method Program Dependence Graph that UNIFIES the previous families:
  * statement nodes + control-flow edges      (CFG  — Family 3)
  * control-dependence edges  CDG_CONTROLS     (structural: predicate->stmt)
  * data-dependence edges     DDG_REACHES      (reaching-definitions — Family 4,
                                                here flow-sensitive over the CFG,
                                                so loop-carried deps ARE captured)
PDG = CDG + DDG over the statement nodes; the full CPG additionally embeds the
AST subtree (Family 2) under each statement and the call edges (Family 3) — we
keep this graph at statement+dependence granularity for tractable analysis and
report the layer breakdown.

Same offline killable-subprocess architecture. javalang `for x in node` walks
the whole subtree; use node.children for direct children.

Outputs (under outputs/):
  cpg_subgraph_stats.csv     master per-method PDG feature table (deliverable)
  cpg_rows.json              typed rows for the notebook
  cpg_aggregates.json        edge-layer / node-kind totals
  cpg_example_graphs/*.pkl   ~12 example PDGs for visualization

Run:  python build_cpg_features.py
"""

import sys, csv, json, math, pickle, subprocess, tempfile, os
from pathlib import Path
from collections import defaultdict, Counter, deque

# per-project config; in this package only CPGBuilder is used (via subgraph_builders.py)
import _kgc_paths  # noqa: F401
from config.project_config import PROJECT_ROOT, REPO_PATH, BASE_COMMIT, OUT
OUT_DIR      = OUT
EX_GRAPH_DIR = OUT_DIR / "cpg_example_graphs"
PARSE_TIMEOUT_S = 45
MAX_SRC_BYTES   = 600_000

# node-kind groups / colours (statement granularity, mirrors CFG)
CPG_COLORS = {
    'entryexit': '#34495E', 'branch': '#E67E22', 'loop': '#C0392B',
    'switch': '#8E44AD', 'jump': '#16A085', 'exception': '#D35400',
    'simple': '#2980B9', 'other': '#95A5A6',
}
_KIND_GROUP = {
    'ENTRY': 'entryexit', 'EXIT': 'entryexit', 'IF': 'branch',
    'WHILE': 'loop', 'FOR': 'loop', 'DO_WHILE': 'loop',
    'SWITCH': 'switch', 'CASE': 'switch',
    'RETURN': 'jump', 'BREAK': 'jump', 'CONTINUE': 'jump', 'THROW': 'jump',
    'TRY': 'exception', 'CATCH': 'exception', 'SYNC': 'exception',
    'EXPR': 'simple', 'DECL': 'simple', 'ASSERT': 'simple',
    'STMT': 'simple', 'NOP': 'simple',
}
def cpg_group(kind): return _KIND_GROUP.get(kind, 'other')
def cpg_color(kind): return CPG_COLORS.get(cpg_group(kind), CPG_COLORS['other'])

# edge layers
_CFG_RELS = {'NEXT','TRUE','FALSE','LOOP_BACK','CASE','FALL','THROW','EXCEPTION','RETURN'}
def edge_layer(rel):
    if rel == 'CDG_CONTROLS': return 'cdg'
    if rel == 'DDG_REACHES':  return 'ddg'
    return 'cfg'

def shannon_entropy(values):
    total = sum(values)
    if total == 0: return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs)


# ═════════════════════════════════════════════════════════════════════════
#  PDG / CPG BUILDER
# ═════════════════════════════════════════════════════════════════════════
class CPGBuilder:
    def __init__(self, nx_mod, jlt):
        self.nx = nx_mod
        self.jlt = jlt
        self.G = nx_mod.MultiDiGraph()   # multi-layer: CFG + CDG + DDG edges coexist
        self.ctr = 0
        self.loop_stack = []            # {'break':[], 'continue':node}
        self.returns, self.throws = [], []
        self.cdg = []                   # (controller, node) structural control-dep
        self.defined = set()            # local var names (pass 1)
        self.def_vars = {}              # node -> set(vars defined)
        self.use_vars = {}              # node -> set(vars used)
        self.entry = self.exit = None
        self.depth = 0; self.max_nesting = 0

    # ---- pass 1: locally-defined variable names ----
    def collect_defined(self, m):
        jlt = self.jlt
        for p in (m.parameters or []):
            if getattr(p, 'name', None): self.defined.add(p.name)
        stack = [m]
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
            for ch in node.children:
                if ch is None: continue
                if isinstance(ch, jlt.Node): stack.append(ch)
                elif isinstance(ch, (list, set, frozenset)):
                    for it in ch:
                        if isinstance(it, jlt.Node): stack.append(it)

    def _uses(self, expr):
        """set of local-variable names read anywhere in expr."""
        jlt = self.jlt
        out = set()
        if not isinstance(expr, jlt.Node):
            if isinstance(expr, (list, set, frozenset)):
                for it in expr:
                    out |= self._uses(it)
            return out
        stack = [expr]
        while stack:
            node = stack.pop()
            if not isinstance(node, jlt.Node): continue
            if isinstance(node, jlt.MemberReference):
                if node.member in self.defined and not node.qualifier:
                    out.add(node.member)
                if isinstance(node.qualifier, str) and node.qualifier in self.defined:
                    out.add(node.qualifier)
            for ch in node.children:
                if ch is None: continue
                if isinstance(ch, jlt.Node): stack.append(ch)
                elif isinstance(ch, (list, set, frozenset)):
                    for it in ch:
                        if isinstance(it, jlt.Node): stack.append(it)
        return out

    def _line(self, node, fb=None):
        p = getattr(node, 'position', None)
        return p.line if p is not None else fb

    def _new(self, kind, label='', line=None, ctrl=None, defs=(), uses=()):
        nid = f"N{self.ctr}"; self.ctr += 1
        self.G.add_node(nid, kind=kind, group=cpg_group(kind), color=cpg_color(kind),
                        label=str(label)[:26], line=line if line is not None else -1)
        self.def_vars[nid] = set(defs)
        self.use_vars[nid] = set(uses)
        if ctrl is not None:
            self.cdg.append((ctrl, nid))
        return nid

    def _edge(self, u, v, rel='NEXT'):
        if u is not None and v is not None:
            self.G.add_edge(u, v, rel=rel, layer=edge_layer(rel))

    def _stmts(self, s):
        if s is None: return []
        if isinstance(s, self.jlt.BlockStatement): return s.statements or []
        if isinstance(s, list): return s
        return [s]

    def _enter(self): self.depth += 1; self.max_nesting = max(self.max_nesting, self.depth)
    def _leave(self): self.depth -= 1

    # ---- def/use for a single statement's own node ----
    def _assign_defuse(self, asg):
        jlt = self.jlt
        compound = asg.type not in ('=', None)
        uses = set(self._uses(asg.value))
        defs = set()
        lhs = asg.expressionl
        if isinstance(lhs, jlt.MemberReference) and not lhs.qualifier and lhs.member in self.defined:
            if compound: uses.add(lhs.member)
            defs.add(lhs.member)
        else:
            uses |= self._uses(lhs)
        return defs, uses

    # ---- pass 2: structured CFG walk (also records CDG + def/use) ----
    def process_seq(self, stmts, ctrl):
        entry = None; prev = []
        for st in self._stmts(stmts):
            s_e, s_x = self.process_stmt(st, ctrl)
            if entry is None: entry = s_e
            else:
                for pe in prev: self._edge(pe, s_e, 'NEXT')
            prev = s_x
            if not s_x: break
        if entry is None:                      # empty block
            n = self._new('NOP', '{}', ctrl=ctrl)
            return n, [n]
        return entry, prev

    def process_stmt(self, st, ctrl):
        jlt = self.jlt; line = self._line(st)

        if isinstance(st, jlt.IfStatement):
            n = self._new('IF', 'if', line, ctrl=ctrl, uses=self._uses(st.condition))
            self._enter()
            te, tx = self.process_seq(st.then_statement, n)
            self._edge(n, te, 'TRUE'); exits = list(tx)
            if st.else_statement is not None:
                ee, ex = self.process_seq(st.else_statement, n)
                self._edge(n, ee, 'FALSE'); exits += ex
            else:
                exits.append(n)
            self._leave(); return n, exits

        if isinstance(st, (jlt.WhileStatement, jlt.ForStatement)) or isinstance(st, jlt.DoStatement):
            is_do = isinstance(st, jlt.DoStatement)
            if isinstance(st, jlt.WhileStatement):
                kind, uses, defs = 'WHILE', self._uses(st.condition), set()
            elif isinstance(st, jlt.DoStatement):
                kind, uses, defs = 'DO_WHILE', self._uses(st.condition), set()
            else:
                kind, uses, defs = 'FOR', set(), set()
                ctrl_node = st.control
                if isinstance(ctrl_node, jlt.EnhancedForControl):
                    uses |= self._uses(getattr(ctrl_node, 'iterable', None))
                    var = getattr(ctrl_node, 'var', None)
                    for d in (getattr(var, 'declarators', []) or []):
                        defs.add(d.name)
                elif ctrl_node is not None:
                    if ctrl_node.condition is not None:
                        uses |= self._uses(ctrl_node.condition)
                    for part in ('init', 'update'):
                        seq = getattr(ctrl_node, part, None) or []
                        seq = seq if isinstance(seq, list) else [seq]
                        for x in seq:
                            if isinstance(x, jlt.LocalVariableDeclaration):
                                for d in x.declarators:
                                    defs.add(d.name); uses |= self._uses(d.initializer)
                            elif isinstance(x, jlt.Assignment):
                                dd, uu = self._assign_defuse(x); defs |= dd; uses |= uu
                            else:
                                uses |= self._uses(x)
            n = self._new(kind, kind.lower(), line, ctrl=ctrl, defs=defs, uses=uses)
            self.loop_stack.append({'break': [], 'continue': n})
            self._enter()
            be, bx = self.process_seq(st.body, n)
            self._leave()
            if is_do:
                for e in bx: self._edge(e, n, 'NEXT')
                self._edge(n, be, 'LOOP_BACK')
                cx = self.loop_stack.pop()
                return be, [n] + cx['break']
            else:
                self._edge(n, be, 'TRUE')
                for e in bx: self._edge(e, n, 'LOOP_BACK')
                cx = self.loop_stack.pop()
                return n, [n] + cx['break']

        if isinstance(st, jlt.SwitchStatement):
            n = self._new('SWITCH', 'switch', line, ctrl=ctrl, uses=self._uses(st.expression))
            self.loop_stack.append({'break': [], 'continue': None})
            self._enter(); prev_fall = []
            for case in (st.cases or []):
                ce, cx = self.process_seq(case.statements, n)
                self._edge(n, ce, 'CASE')
                for pf in prev_fall: self._edge(pf, ce, 'FALL')
                prev_fall = cx
            self._leave(); cxn = self.loop_stack.pop()
            return n, list(prev_fall) + cxn['break'] + [n]

        if isinstance(st, jlt.TryStatement):
            self._enter()
            te, tx = self.process_seq(st.block, ctrl); exits = list(tx)
            for catch in (st.catches or []):
                param = getattr(catch, 'parameter', None)
                cvar = {param.name} if param is not None and getattr(param, 'name', None) else set()
                cn = self._new('CATCH', 'catch', self._line(catch, line), ctrl=ctrl, defs=cvar)
                self._edge(te, cn, 'EXCEPTION')
                ce, cx = self.process_seq(catch.block, cn)
                self._edge(cn, ce, 'NEXT'); exits += cx
            fin = getattr(st, 'finally_block', None)
            if fin:
                fe, fx = self.process_seq(fin, ctrl)
                for e in exits: self._edge(e, fe, 'NEXT')
                exits = fx
            self._leave(); return te, exits

        if isinstance(st, jlt.SynchronizedStatement):
            n = self._new('SYNC', 'synchronized', line, ctrl=ctrl, uses=self._uses(st.lock))
            self._enter(); be, bx = self.process_seq(st.block, n); self._leave()
            self._edge(n, be, 'NEXT'); return n, bx

        if isinstance(st, jlt.BlockStatement):
            return self.process_seq(st.statements, ctrl)

        if isinstance(st, jlt.LocalVariableDeclaration):
            defs, uses = set(), set()
            for d in st.declarators:
                defs.add(d.name); uses |= self._uses(d.initializer)
            n = self._new('DECL', 'decl', line, ctrl=ctrl, defs=defs, uses=uses)
            return n, [n]

        if isinstance(st, jlt.StatementExpression):
            e = st.expression
            if isinstance(e, jlt.Assignment):
                defs, uses = self._assign_defuse(e)
                n = self._new('EXPR', 'assign', line, ctrl=ctrl, defs=defs, uses=uses)
                return n, [n]
            if isinstance(e, jlt.MemberReference) and not e.qualifier and \
               e.member in self.defined and (e.prefix_operators or e.postfix_operators):
                n = self._new('EXPR', 'update', line, ctrl=ctrl,
                              defs={e.member}, uses={e.member})
                return n, [n]
            n = self._new('EXPR', 'expr', line, ctrl=ctrl, uses=self._uses(e))
            return n, [n]

        if isinstance(st, jlt.ReturnStatement):
            n = self._new('RETURN', 'return', line, ctrl=ctrl,
                          uses=self._uses(getattr(st, 'expression', None)))
            self.returns.append(n); return n, []

        if isinstance(st, jlt.ThrowStatement):
            n = self._new('THROW', 'throw', line, ctrl=ctrl,
                          uses=self._uses(getattr(st, 'expression', None)))
            self.throws.append(n); return n, []

        if isinstance(st, jlt.BreakStatement):
            n = self._new('BREAK', 'break', line, ctrl=ctrl)
            if self.loop_stack: self.loop_stack[-1]['break'].append(n)
            return n, []

        if isinstance(st, jlt.ContinueStatement):
            n = self._new('CONTINUE', 'continue', line, ctrl=ctrl)
            if self.loop_stack and self.loop_stack[-1]['continue'] is not None:
                self._edge(n, self.loop_stack[-1]['continue'], 'LOOP_BACK')
            return n, []

        # other / assert
        T = type(st).__name__
        kind = 'ASSERT' if T == 'AssertStatement' else 'STMT'
        uses = set()
        for attr in getattr(st, 'attrs', []):
            uses |= self._uses(getattr(st, attr, None))
        n = self._new(kind, T.replace('Statement', '') or T, line, ctrl=ctrl, uses=uses)
        return n, [n]

    # ---- reaching-definitions over the CFG -> DDG edges ----
    def _reaching_defs_ddg(self):
        cfg_succ = defaultdict(list); cfg_pred = defaultdict(list)
        for u, v, d in self.G.edges(data=True):
            if d['layer'] == 'cfg':
                cfg_succ[u].append(v); cfg_pred[v].append(u)
        # gen/kill
        defs_of = defaultdict(set)   # var -> set(nodes defining it)
        for n, dv in self.def_vars.items():
            for v in dv: defs_of[v].add(n)
        gen = {n: {(v, n) for v in self.def_vars[n]} for n in self.G.nodes()}
        IN = {n: set() for n in self.G.nodes()}
        OUT = {n: set(gen[n]) for n in self.G.nodes()}
        wl = deque(self.G.nodes())
        while wl:
            n = wl.popleft()
            newin = set()
            for p in cfg_pred[n]: newin |= OUT[p]
            killvars = self.def_vars[n]
            newout = gen[n] | {(v, m) for (v, m) in newin if v not in killvars}
            if newin != IN[n] or newout != OUT[n]:
                IN[n] = newin; OUT[n] = newout
                wl.extend(cfg_succ[n])
        ddg = 0
        for n in self.G.nodes():
            uv = self.use_vars[n]
            if not uv: continue
            for (v, m) in IN[n]:
                if v in uv and m != n:
                    self._edge(m, n, 'DDG_REACHES'); ddg += 1
        return ddg

    def build(self, method_node):
        self.collect_defined(method_node)
        params = [p.name for p in (method_node.parameters or []) if getattr(p, 'name', None)]
        self.entry = self._new('ENTRY', 'entry', defs=set(params))
        self.exit = self._new('EXIT', 'exit')
        be, bx = self.process_seq(method_node.body, self.entry)
        self._edge(self.entry, be, 'NEXT')
        for e in bx: self._edge(e, self.exit, 'NEXT')
        for r in self.returns: self._edge(r, self.exit, 'RETURN')
        for t in self.throws: self._edge(t, self.exit, 'THROW')
        # data dependence (before adding CDG edges, so CFG-only reaching analysis)
        self._reaching_defs_ddg()
        # control dependence edges
        for c, n in self.cdg:
            self._edge(c, n, 'CDG_CONTROLS')
        return self.G, self.entry


def _enclosing_class(path_nodes, jlt):
    for anc in reversed(path_nodes):
        if isinstance(anc, (jlt.ClassDeclaration, jlt.InterfaceDeclaration,
                            jlt.EnumDeclaration, jlt.AnnotationDeclaration)):
            return anc.name
    return '<unknown>'


def _cpg_stats(G, builder):
    layers = Counter(d['layer'] for _, _, d in G.edges(data=True))
    kinds = Counter(d['kind'] for _, d in G.nodes(data=True))
    n = G.number_of_nodes()
    n_cfg = layers.get('cfg', 0)
    n_cdg = layers.get('cdg', 0)
    n_ddg = layers.get('ddg', 0)
    e_tot = n_cfg + n_cdg + n_ddg
    n_pred = kinds.get('IF', 0) + kinds.get('WHILE', 0) + kinds.get('FOR', 0) \
        + kinds.get('DO_WHILE', 0) + kinds.get('SWITCH', 0)
    cyclo = max(n_cfg - n + 2, 1)
    return {
        'n_cpg_nodes': n, 'n_cfg_edges': n_cfg, 'n_cdg_edges': n_cdg,
        'n_ddg_edges': n_ddg, 'n_total_edges': e_tot,
        'cyclomatic': cyclo, 'n_predicates': n_pred,
        'max_nesting': builder.max_nesting,
        'edges_per_node': round(e_tot / max(1, n), 2),
        'pdg_edges': n_cdg + n_ddg,
        'n_node_kinds': len(kinds),
        'kind_entropy': round(shannon_entropy(list(kinds.values())), 4),
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
    layer_totals, kind_totals = Counter(), Counter()
    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class(path_nodes, jlt)
        kind = 'constructor' if isinstance(node, jlt.ConstructorDeclaration) else 'method'
        mid  = f"{rel_path}::{cls}::{node.name}@{node.position.line}"

        b = CPGBuilder(nx, jlt)
        G, _entry = b.build(node)
        layer_totals += Counter(d['layer'] for _, _, d in G.edges(data=True))
        kind_totals += Counter(d['kind'] for _, d in G.nodes(data=True))
        st = _cpg_stats(G, b)
        rows.append({
            'method_id': mid, 'file': rel_path,
            'class': cls, 'method': node.name, 'kind': kind, **st,
            'has_method_link': True, 'has_class_ancestor': cls != '<unknown>',
            'has_file_ancestor': True, 'has_pkg_ancestor': bool(has_pkg),
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'layer_totals': dict(layer_totals),
                   'kind_totals': dict(kind_totals)}, f)


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
            b = CPGBuilder(nx, jlt)
            G, entry = b.build(jnode)
            safe = "".join(c if c.isalnum() else "_" for c in mid)
            (EX_GRAPH_DIR / f"{safe}.pkl").write_bytes(pickle.dumps((G, entry), protocol=4))
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
    layer_totals, kind_totals = Counter(), Counter()
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
            layer_totals += Counter(res['layer_totals'])
            kind_totals += Counter(res['kind_totals'])
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
                 'n_cpg_nodes','n_cfg_edges','n_cdg_edges','n_ddg_edges',
                 'n_total_edges','pdg_edges','edges_per_node','cyclomatic',
                 'n_predicates','max_nesting','n_node_kinds','kind_entropy',
                 'has_method_link','has_class_ancestor','has_file_ancestor','has_pkg_ancestor']
    with open(OUT_DIR / "cpg_subgraph_stats.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=col_order); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k, '') for k in col_order})
    with open(OUT_DIR / "cpg_rows.json", 'w', encoding='utf-8') as f:
        json.dump(all_rows, f)
    with open(OUT_DIR / "cpg_aggregates.json", 'w', encoding='utf-8') as f:
        json.dump({'layer_totals': dict(layer_totals), 'kind_totals': dict(kind_totals),
                   'n_methods': len(all_rows), 'ok_files': len(ok_files),
                   'skipped_files': skipped_files}, f)

    et = [r['n_total_edges'] for r in all_rows]
    print(f"\nCPG edges/method : min={min(et)} med={sorted(et)[len(et)//2]} "
          f"mean={sum(et)/len(et):.1f} max={max(et)}")
    print(f"Edge layers      : CFG={layer_totals['cfg']:,} "
          f"CDG={layer_totals['cdg']:,} DDG={layer_totals['ddg']:,}")
    print(f"Total PDG edges  : {layer_totals['cdg']+layer_totals['ddg']:,}")

    # ── Phase 2: example PDGs (multi-layer, branchy + data flow) ────────────
    ok_set = set(ok_files)
    viz_ids = []
    for r in sorted([r for r in all_rows
                     if 7 <= r['n_cpg_nodes'] <= 40 and r['file'] in ok_set
                     and r['n_ddg_edges'] >= 2 and r['n_cdg_edges'] >= 2],
                    key=lambda r: -(r['n_cdg_edges'] + r['n_ddg_edges']))[:6]:
        viz_ids.append(r['method_id'])
    cc = Counter(r['class'] for r in all_rows
                 if 6 <= r['n_cpg_nodes'] <= 45 and r['file'] in ok_set
                 and r['pdg_edges'] >= 2)
    if cc:
        tcls = cc.most_common(1)[0][0]
        for r in [r for r in all_rows if r['class'] == tcls
                  and 6 <= r['n_cpg_nodes'] <= 45 and r['file'] in ok_set
                  and r['pdg_edges'] >= 2][:6]:
            if r['method_id'] not in viz_ids: viz_ids.append(r['method_id'])

    by_mid = {r['method_id']: r for r in all_rows}
    file_to_wanted = defaultdict(list)
    for mid in viz_ids:
        r = by_mid[mid]
        file_to_wanted[r['file']].append([r['class'], r['method'], r['kind'], mid])

    print(f"\nBuilding {len(viz_ids)} example PDGs from {len(file_to_wanted)} files...", flush=True)
    built = 0
    for rel, wanted in file_to_wanted.items():
        fd, wp = tempfile.mkstemp(suffix='.json'); os.close(fd)
        with open(wp, 'w', encoding='utf-8') as f: json.dump(wanted, f)
        res = run_worker('--worker-graph', [str(REPO_PATH / rel), rel, wp], PARSE_TIMEOUT_S)
        try: os.unlink(wp)
        except OSError: pass
        if isinstance(res, dict): built += len(res['saved'])
        else: print(f"  graph build failed for {rel}: {res}", flush=True)
    print(f"Example PDGs saved : {built}  -> {EX_GRAPH_DIR}", flush=True)

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
