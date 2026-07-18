#!/usr/bin/env python3
"""
Standalone Control-Flow-Graph (CFG) feature extractor for KG-Commit
base snapshot — Family 3 (Control-Flow Representations).

Same offline architecture as build_ast_features.py: each Java file is parsed in
a killable subprocess (subprocess.run(..., timeout=N)) so a pathological file is
terminated and skipped instead of hanging. For every concrete method we build a
statement-level CFG from the javalang AST and compute control-flow metrics
(cyclomatic complexity, branch/loop counts, nesting depth, exception handling,
call sites). We also assemble a project-level call graph.

IMPORTANT: javalang Node iteration `for x in node` walks the ENTIRE subtree
(walk_tree). To get DIRECT children use `node.children`. (See the long debugging
saga for Family 2.)

Outputs (under outputs/):
  cfg_subgraph_stats.csv     master per-method CFG feature table (deliverable)
  cfg_rows.json              same rows, typed, for the notebook to load
  cfg_aggregates.json        edge-type / node-type totals + complexity hist
  cfg_call_graph.json        project-level method->method call graph
  cfg_example_graphs/*.pkl   ~12 example CFGs for visualization

Run:
  python build_cfg_features.py
"""

import sys, csv, json, math, pickle, subprocess, tempfile, os
from pathlib import Path
from collections import defaultdict, Counter

# ── Configuration (per-project). In this package this module is used ONLY for its
#    CFGBuilder class (imported by subgraph_builders.py); the standalone main()/
#    base-snapshot path is not part of the reproduction pipeline. Constants derive
#    from config so no absolute/groovy path is baked in. ────────────────────────
import _kgc_paths  # noqa: F401
from config.project_config import PROJECT_ROOT, REPO_PATH, BASE_COMMIT, OUT
OUT_DIR      = OUT
EX_GRAPH_DIR = OUT_DIR / "cfg_example_graphs"

PARSE_TIMEOUT_S = 45
MAX_SRC_BYTES   = 600_000

# ── CFG node-kind groups and colours (for visualization) ────────────────────
CFG_COLORS = {
    'entryexit':  '#34495E',   # dark slate  (ENTRY/EXIT)
    'branch':     '#E67E22',   # orange      (IF)
    'loop':       '#C0392B',   # red         (WHILE/FOR/DO)
    'switch':     '#8E44AD',   # purple      (SWITCH/CASE)
    'jump':       '#16A085',   # teal        (RETURN/BREAK/CONTINUE/THROW)
    'exception':  '#D35400',   # dark orange (CATCH/TRY)
    'simple':     '#2980B9',   # blue        (EXPR/DECL/...)
    'other':      '#95A5A6',
}
_KIND_GROUP = {
    'ENTRY': 'entryexit', 'EXIT': 'entryexit',
    'IF': 'branch',
    'WHILE': 'loop', 'FOR': 'loop', 'DO_WHILE': 'loop',
    'SWITCH': 'switch', 'CASE': 'switch',
    'RETURN': 'jump', 'BREAK': 'jump', 'CONTINUE': 'jump', 'THROW': 'jump',
    'TRY': 'exception', 'CATCH': 'exception', 'SYNC': 'exception',
    'EXPR': 'simple', 'DECL': 'simple', 'ASSERT': 'simple',
    'STMT': 'simple', 'NOP': 'simple',
}
def cfg_group(kind): return _KIND_GROUP.get(kind, 'other')
def cfg_color(kind): return CFG_COLORS.get(cfg_group(kind), CFG_COLORS['other'])

def shannon_entropy(values):
    total = sum(values)
    if total == 0: return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p) for p in probs)


# ═════════════════════════════════════════════════════════════════════════
#  CFG BUILDER  (structured, statement-level, over the javalang AST)
# ═════════════════════════════════════════════════════════════════════════
class CFGBuilder:
    """Builds a statement-level control-flow graph for one method body.

    process_seq / process_stmt return (entry_node, exit_nodes):
      entry_node  = first node of the construct
      exit_nodes  = list of dangling nodes whose control falls through to
                    whatever statement comes next.
    Terminal statements (return/throw/break/continue) yield exit_nodes = [].
    """
    def __init__(self, nx_mod, jlt):
        self.nx = nx_mod
        self.jlt = jlt
        self.G = nx_mod.DiGraph()
        self.ctr = 0
        self.loop_stack = []     # [{'break': [...], 'continue': node_or_None}]
        self.returns = []
        self.throws = []
        self.depth = 0
        self.max_nesting = 0

    def _new(self, kind, label='', line=None):
        nid = f"N{self.ctr}"; self.ctr += 1
        self.G.add_node(nid, kind=kind, group=cfg_group(kind),
                        color=cfg_color(kind), label=str(label)[:28],
                        line=line if line is not None else -1)
        return nid

    def _edge(self, u, v, rel='NEXT'):
        if u is not None and v is not None:
            self.G.add_edge(u, v, rel=rel)

    def _stmts_of(self, s):
        if s is None: return []
        if isinstance(s, self.jlt.BlockStatement):
            return s.statements or []
        if isinstance(s, list): return s
        return [s]

    def _line(self, st):
        p = getattr(st, 'position', None)
        return p.line if p is not None else None

    def build(self, method_node):
        entry = self._new('ENTRY', 'entry')
        exit_ = self._new('EXIT', 'exit')
        body_entry, body_exits = self.process_seq(method_node.body)
        self._edge(entry, body_entry, 'NEXT')
        for e in body_exits:
            self._edge(e, exit_, 'NEXT')
        for r in self.returns:
            self._edge(r, exit_, 'RETURN')
        for t in self.throws:
            self._edge(t, exit_, 'THROW')
        return self.G, entry

    def process_seq(self, stmts):
        stmts = self._stmts_of(stmts)
        if not stmts:
            n = self._new('NOP', '{}')
            return n, [n]
        entry = None
        prev_exits = []
        for st in stmts:
            s_entry, s_exits = self.process_stmt(st)
            if entry is None:
                entry = s_entry
            else:
                for pe in prev_exits:
                    self._edge(pe, s_entry, 'NEXT')
            prev_exits = s_exits
            if not s_exits:
                break    # unreachable tail after return/throw/break/continue
        return entry, prev_exits

    def _enter(self):
        self.depth += 1
        self.max_nesting = max(self.max_nesting, self.depth)
    def _leave(self):
        self.depth -= 1

    def process_stmt(self, st):
        jlt = self.jlt
        line = self._line(st)

        if isinstance(st, jlt.IfStatement):
            cond = self._new('IF', 'if', line)
            self._enter()
            then_e, then_x = self.process_seq(st.then_statement)
            self._edge(cond, then_e, 'TRUE')
            exits = list(then_x)
            if st.else_statement is not None:
                else_e, else_x = self.process_seq(st.else_statement)
                self._edge(cond, else_e, 'FALSE')
                exits += else_x
            else:
                exits.append(cond)            # false falls through
            self._leave()
            return cond, exits

        if isinstance(st, jlt.WhileStatement):
            cond = self._new('WHILE', 'while', line)
            self.loop_stack.append({'break': [], 'continue': cond})
            self._enter()
            body_e, body_x = self.process_seq(st.body)
            self._leave()
            self._edge(cond, body_e, 'TRUE')
            for be in body_x:
                self._edge(be, cond, 'LOOP_BACK')
            ctx = self.loop_stack.pop()
            return cond, [cond] + ctx['break']

        if isinstance(st, jlt.ForStatement):
            cond = self._new('FOR', 'for', line)
            self.loop_stack.append({'break': [], 'continue': cond})
            self._enter()
            body_e, body_x = self.process_seq(st.body)
            self._leave()
            self._edge(cond, body_e, 'TRUE')
            for be in body_x:
                self._edge(be, cond, 'LOOP_BACK')
            ctx = self.loop_stack.pop()
            return cond, [cond] + ctx['break']

        if isinstance(st, jlt.DoStatement):
            cond = self._new('DO_WHILE', 'do-while', line)
            self.loop_stack.append({'break': [], 'continue': cond})
            self._enter()
            body_e, body_x = self.process_seq(st.body)
            self._leave()
            for be in body_x:
                self._edge(be, cond, 'NEXT')
            self._edge(cond, body_e, 'LOOP_BACK')    # true -> repeat body
            ctx = self.loop_stack.pop()
            return body_e, [cond] + ctx['break']

        if isinstance(st, jlt.SwitchStatement):
            sw = self._new('SWITCH', 'switch', line)
            self.loop_stack.append({'break': [], 'continue': None})
            self._enter()
            prev_fall = []
            for case in (st.cases or []):
                c_entry, c_exits = self.process_seq(case.statements)
                self._edge(sw, c_entry, 'CASE')
                for pf in prev_fall:
                    self._edge(pf, c_entry, 'FALL')
                prev_fall = c_exits
            self._leave()
            ctx = self.loop_stack.pop()
            exits = list(prev_fall) + ctx['break'] + [sw]
            return sw, exits

        if isinstance(st, jlt.TryStatement):
            self._enter()
            try_e, try_x = self.process_seq(st.block)
            exits = list(try_x)
            for catch in (st.catches or []):
                cn = self._new('CATCH', 'catch', self._line(catch))
                self._edge(try_e, cn, 'EXCEPTION')
                cb_e, cb_x = self.process_seq(catch.block)
                self._edge(cn, cb_e, 'NEXT')
                exits += cb_x
            fin = getattr(st, 'finally_block', None)
            if fin:
                fin_e, fin_x = self.process_seq(fin)
                for e in exits:
                    self._edge(e, fin_e, 'NEXT')
                exits = fin_x
            self._leave()
            return try_e, exits

        if isinstance(st, jlt.SynchronizedStatement):
            n = self._new('SYNC', 'synchronized', line)
            self._enter()
            body_e, body_x = self.process_seq(st.block)
            self._leave()
            self._edge(n, body_e, 'NEXT')
            return n, body_x

        if isinstance(st, jlt.BlockStatement):
            return self.process_seq(st.statements)

        if isinstance(st, jlt.ReturnStatement):
            n = self._new('RETURN', 'return', line); self.returns.append(n)
            return n, []

        if isinstance(st, jlt.ThrowStatement):
            n = self._new('THROW', 'throw', line); self.throws.append(n)
            return n, []

        if isinstance(st, jlt.BreakStatement):
            n = self._new('BREAK', 'break', line)
            if self.loop_stack:
                self.loop_stack[-1]['break'].append(n)
            return n, []

        if isinstance(st, jlt.ContinueStatement):
            n = self._new('CONTINUE', 'continue', line)
            if self.loop_stack and self.loop_stack[-1]['continue'] is not None:
                self._edge(n, self.loop_stack[-1]['continue'], 'LOOP_BACK')
            return n, []

        # default: simple sequential statement
        T = type(st).__name__
        kind = {'StatementExpression': 'EXPR',
                'LocalVariableDeclaration': 'DECL',
                'AssertStatement': 'ASSERT'}.get(T, 'STMT')
        n = self._new(kind, T.replace('Statement', '') or T, line)
        return n, [n]


def _count_calls_and_types(method_node, jlt):
    """Walk method AST (via .children!) to count call sites and collect
    callee method names + instantiated/used type names."""
    n_calls = 0
    callees = []
    stack = [method_node]
    while stack:
        node = stack.pop()
        if isinstance(node, jlt.MethodInvocation):
            n_calls += 1
            if node.member:
                callees.append(node.member)
        for child in node.children:    # DIRECT children only
            if child is None: continue
            if isinstance(child, jlt.Node):
                stack.append(child)
            elif isinstance(child, (list, set, frozenset)):
                for it in child:
                    if isinstance(it, jlt.Node):
                        stack.append(it)
    return n_calls, callees


def _enclosing_class(path_nodes, jlt):
    for anc in reversed(path_nodes):
        if isinstance(anc, (jlt.ClassDeclaration, jlt.InterfaceDeclaration,
                            jlt.EnumDeclaration, jlt.AnnotationDeclaration)):
            return anc.name
    return '<unknown>'


def _cfg_stats(G, kinds_counter):
    n = G.number_of_nodes(); e = G.number_of_edges()
    edge_types = Counter(d.get('rel', 'NEXT') for _, _, d in G.edges(data=True))
    n_branches = kinds_counter.get('IF', 0)
    n_loops    = (kinds_counter.get('WHILE', 0) + kinds_counter.get('FOR', 0)
                  + kinds_counter.get('DO_WHILE', 0))
    n_switch   = kinds_counter.get('SWITCH', 0)
    n_catch    = kinds_counter.get('CATCH', 0)
    # decision points for McCabe: if + loops + cases + catch
    n_cases    = sum(1 for _ in ())  # placeholder; cases counted via FALL/CASE edges
    n_decisions = n_branches + n_loops + n_switch + n_catch
    cyclo_edges = e - n + 2          # McCabe: E - N + 2P, P=1
    cyclo_dec   = n_decisions + 1
    return {
        'n_cfg_nodes': n, 'n_cfg_edges': e,
        'cyclomatic': max(cyclo_edges, 1),
        'cyclomatic_decisions': cyclo_dec,
        'n_decision_points': n_decisions,
        'n_branches': n_branches, 'n_loops': n_loops, 'n_switch': n_switch,
        'n_returns': kinds_counter.get('RETURN', 0),
        'n_throws':  kinds_counter.get('THROW', 0),
        'n_breaks':  kinds_counter.get('BREAK', 0),
        'n_continues': kinds_counter.get('CONTINUE', 0),
        'n_catch': n_catch, 'has_try': n_catch > 0 or kinds_counter.get('SYNC',0) > 0,
        'n_back_edges': edge_types.get('LOOP_BACK', 0),
        'edge_types': dict(edge_types),
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
    edge_totals, node_totals = Counter(), Counter()
    call_edges = []     # (caller_mid, callee_name)
    method_defs = []    # (mid, simple_method_name)

    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class(path_nodes, jlt)
        kind = 'constructor' if isinstance(node, jlt.ConstructorDeclaration) else 'method'
        mid  = f"{rel_path}::{cls}::{node.name}@{node.position.line}"

        b = CFGBuilder(nx, jlt)
        G, _entry = b.build(node)
        kinds = Counter(d['kind'] for _, d in G.nodes(data=True))
        node_totals += kinds
        st = _cfg_stats(G, kinds)
        edge_totals += Counter(st.pop('edge_types'))

        n_calls, callees = _count_calls_and_types(node, jlt)
        declared_throws = list(node.throws) if getattr(node, 'throws', None) else []

        method_defs.append((mid, node.name))
        for c in callees:
            call_edges.append((mid, c))

        type_ent = shannon_entropy(list(kinds.values()))
        rows.append({
            'method_id': mid, 'file': rel_path,
            'class': cls, 'method': node.name, 'kind': kind,
            **st,
            'max_nesting': b.max_nesting,
            'n_call_sites': n_calls,
            'n_throws_declared': len(declared_throws),
            'throws_declared': ';'.join(str(t) for t in declared_throws),
            'n_node_kinds': len(kinds),
            'kind_entropy': round(type_ent, 4),
            'has_method_link': True,
            'has_class_ancestor': cls != '<unknown>',
            'has_file_ancestor': True,
            'has_pkg_ancestor': bool(has_pkg),
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'edge_totals': dict(edge_totals),
                   'node_totals': dict(node_totals),
                   'call_edges': call_edges, 'method_defs': method_defs}, f)


def _worker_graph(abs_path, rel_path, wanted_path, out_path):
    import javalang, javalang.tree as jlt, networkx as nx
    wanted = json.load(open(wanted_path, encoding='utf-8'))   # [[cls,method,kind,mid],...]
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
            b = CFGBuilder(nx, jlt)
            G, entry = b.build(jnode)
            safe = "".join(c if c.isalnum() else "_" for c in mid)
            (EX_GRAPH_DIR / f"{safe}.pkl").write_bytes(
                pickle.dumps((G, entry), protocol=4))
            saved.append(mid)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'saved': saved}, f)


# ── Parent: killable subprocess runner ──────────────────────────────────────
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
    for old in EX_GRAPH_DIR.glob('*.pkl'):
        old.unlink()

    head = git("rev-parse HEAD")
    print(f"Current HEAD : {head[:12]}", flush=True)
    git(f"checkout {BASE_COMMIT}")
    print(f"Checked out  : {git('rev-parse HEAD')[:12]}\n", flush=True)

    java_files = sorted(REPO_PATH.rglob("*.java"))
    n_files = len(java_files)
    print(f"Found {n_files} .java files\n", flush=True)

    all_rows = []
    edge_totals, node_totals = Counter(), Counter()
    all_call_edges, all_method_defs = [], []
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
            edge_totals += Counter(res['edge_totals'])
            node_totals += Counter(res['node_totals'])
            all_call_edges.extend(res['call_edges'])
            all_method_defs.extend(res['method_defs'])
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

    # ── CSV + typed JSON + aggregates ───────────────────────────────────────
    col_order = ['method_id','file','class','method','kind',
                 'n_cfg_nodes','n_cfg_edges','cyclomatic','cyclomatic_decisions',
                 'n_decision_points','n_branches','n_loops','n_switch',
                 'n_returns','n_throws','n_breaks','n_continues','n_catch',
                 'has_try','n_back_edges','max_nesting','n_call_sites',
                 'n_throws_declared','throws_declared','n_node_kinds','kind_entropy',
                 'has_method_link','has_class_ancestor','has_file_ancestor','has_pkg_ancestor']
    with open(OUT_DIR / "cfg_subgraph_stats.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=col_order); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k, '') for k in col_order})
    with open(OUT_DIR / "cfg_rows.json", 'w', encoding='utf-8') as f:
        json.dump(all_rows, f)

    # ── Project-level call graph (method name resolution) ───────────────────
    name_to_mids = defaultdict(list)
    for mid, mname in all_method_defs:
        name_to_mids[mname].append(mid)
    internal_edges = Counter()   # (caller_mid, callee_mid)
    external_calls = Counter()   # callee_name not defined in project
    for caller_mid, callee_name in all_call_edges:
        targets = name_to_mids.get(callee_name, [])
        if targets:
            for t in targets:
                internal_edges[(caller_mid, t)] += 1
        else:
            external_calls[callee_name] += 1
    call_graph = {
        'nodes': [{'method_id': mid, 'name': nm} for mid, nm in all_method_defs],
        'internal_edges': [{'caller': a, 'callee': b, 'count': c}
                           for (a, b), c in internal_edges.items()],
        'n_internal_edges': len(internal_edges),
        'n_external_call_names': len(external_calls),
        'top_external': external_calls.most_common(25),
    }
    with open(OUT_DIR / "cfg_call_graph.json", 'w', encoding='utf-8') as f:
        json.dump(call_graph, f)

    with open(OUT_DIR / "cfg_aggregates.json", 'w', encoding='utf-8') as f:
        json.dump({'edge_totals': dict(edge_totals), 'node_totals': dict(node_totals),
                   'n_methods': len(all_rows), 'ok_files': len(ok_files),
                   'skipped_files': skipped_files,
                   'n_internal_call_edges': len(internal_edges),
                   'n_external_call_names': len(external_calls)}, f)

    cyc = [r['cyclomatic'] for r in all_rows]
    nn  = [r['n_cfg_nodes'] for r in all_rows]
    print(f"\nCFG nodes  : min={min(nn)} med={sorted(nn)[len(nn)//2]} "
          f"mean={sum(nn)/len(nn):.0f} max={max(nn)}")
    print(f"Cyclomatic : min={min(cyc)} med={sorted(cyc)[len(cyc)//2]} "
          f"mean={sum(cyc)/len(cyc):.1f} max={max(cyc)}")
    print(f"Edge types : {dict(edge_totals.most_common())}")
    print(f"Call graph : {len(internal_edges)} internal edges, "
          f"{len(external_calls)} distinct external callees")

    # ── Phase 2: example CFGs (small-to-medium, branchy = interesting) ──────
    ok_set = set(ok_files)
    viz_ids = []
    for r in sorted([r for r in all_rows
                     if 6 <= r['n_cfg_nodes'] <= 45 and r['file'] in ok_set
                     and r['n_decision_points'] >= 1],
                    key=lambda r: -r['cyclomatic'])[:6]:
        viz_ids.append(r['method_id'])
    cc = Counter(r['class'] for r in all_rows
                 if 5 <= r['n_cfg_nodes'] <= 60 and r['file'] in ok_set)
    if cc:
        tcls = cc.most_common(1)[0][0]
        for r in [r for r in all_rows if r['class'] == tcls
                  and 5 <= r['n_cfg_nodes'] <= 60 and r['file'] in ok_set][:6]:
            if r['method_id'] not in viz_ids:
                viz_ids.append(r['method_id'])

    by_mid = {r['method_id']: r for r in all_rows}
    file_to_wanted = defaultdict(list)
    for mid in viz_ids:
        r = by_mid[mid]
        file_to_wanted[r['file']].append([r['class'], r['method'], r['kind'], mid])

    print(f"\nBuilding {len(viz_ids)} example CFGs from {len(file_to_wanted)} files...", flush=True)
    built = 0
    for rel, wanted in file_to_wanted.items():
        fd, wp = tempfile.mkstemp(suffix='.json'); os.close(fd)
        with open(wp, 'w', encoding='utf-8') as f: json.dump(wanted, f)
        res = run_worker('--worker-graph', [str(REPO_PATH / rel), rel, wp], PARSE_TIMEOUT_S)
        try: os.unlink(wp)
        except OSError: pass
        if isinstance(res, dict): built += len(res['saved'])
        else: print(f"  graph build failed for {rel}: {res}", flush=True)
    print(f"Example CFGs saved : {built}  -> {EX_GRAPH_DIR}", flush=True)

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
