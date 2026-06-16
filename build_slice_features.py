#!/usr/bin/env python3
"""
Standalone Program-Slicing feature extractor for KG-Commit base snapshot
— Family 7 (Program Slicing).

Slices are a *by-product of the PDG* (Family 5). We reuse the CPGBuilder from
build_cpg_features.py to build each method's Program Dependence Graph, then
compute slices by reachability over the DEPENDENCE edges (CDG ∪ DDG, ignoring
pure control-flow edges):

  backward slice(c) = all statements that may influence c
                     = ancestors of c following dependence in-edges
  forward  slice(c) = all statements c may influence
                     = descendants of c following dependence out-edges

The key predictive feature (per the design doc) is BACKWARD SLICE SIZE:
"how much code does this statement depend on?" — stored as per-statement /
per-method scalars rather than explicit slice-membership edges.

Same offline killable-subprocess architecture.

Outputs (under outputs/):
  slice_subgraph_stats.csv     master per-method slice feature table
  slice_rows.json              typed rows for the notebook
  slice_aggregates.json        slice-size distribution + node-kind totals
  slice_example_graphs/*.pkl   ~12 examples: (PDG, entry, criterion, bslice set)

Run:  python build_slice_features.py
"""

import sys, os, csv, json, math, pickle, subprocess, tempfile
from pathlib import Path
from collections import defaultdict, Counter, deque

# make sibling extractor importable in worker subprocesses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(r"c:\Users\sinab\Documents\GitHub\KG-Commit")
REPO_PATH    = PROJECT_ROOT / "repos" / "apache" / "groovy"
BASE_COMMIT  = "408b29851d7bbe4d343340832297e4be7e0c5578"
OUT_DIR      = PROJECT_ROOT / "outputs"
EX_GRAPH_DIR = OUT_DIR / "slice_example_graphs"
PARSE_TIMEOUT_S = 45
MAX_SRC_BYTES   = 600_000


def _dep_adjacency(G):
    """Dependence-only adjacency (CDG ∪ DDG): succ, pred."""
    succ, pred = defaultdict(set), defaultdict(set)
    for u, v, d in G.edges(data=True):
        if d.get('layer') in ('cdg', 'ddg'):
            succ[u].add(v); pred[v].add(u)
    return succ, pred


def _reach(adj, start):
    seen = {start}; q = deque([start])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in seen:
                seen.add(y); q.append(y)
    return seen


def compute_slices(G, entry):
    """Return (stats_dict, criterion_node, backward_slice_set) for one PDG."""
    succ, pred = _dep_adjacency(G)
    # statement criteria = all nodes except synthetic ENTRY/EXIT
    crit_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') not in ('ENTRY', 'EXIT')]
    n_nodes = max(1, len(crit_nodes))
    n_dep = sum(1 for _, _, d in G.edges(data=True) if d.get('layer') in ('cdg', 'ddg'))

    bsizes, fsizes = {}, {}
    for c in crit_nodes:
        bsizes[c] = len(_reach(pred, c))
        fsizes[c] = len(_reach(succ, c))

    bvals = list(bsizes.values()) or [0]
    fvals = list(fsizes.values()) or [0]

    # output-relevant slice: largest backward slice among return/throw stmts
    out_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') in ('RETURN', 'THROW')]
    return_slice = max((bsizes[n] for n in out_nodes if n in bsizes), default=0)

    # criterion for visualization = statement with the largest backward slice
    criterion = max(crit_nodes, key=lambda n: bsizes.get(n, 0)) if crit_nodes else entry
    bslice_set = _reach(pred, criterion) if criterion in pred or criterion in succ or True else {criterion}
    bslice_set = _reach(pred, criterion)

    max_b = max(bvals); mean_b = sum(bvals) / len(bvals)
    stats = {
        'n_pdg_nodes': G.number_of_nodes(), 'n_stmt_nodes': len(crit_nodes),
        'n_dep_edges': n_dep,
        'max_backward_slice': max_b,
        'mean_backward_slice': round(mean_b, 2),
        'max_forward_slice': max(fvals),
        'mean_forward_slice': round(sum(fvals) / len(fvals), 2),
        'return_slice_size': return_slice,
        'slice_coverage': round(max_b / n_nodes, 3),       # fraction of method in biggest slice
        'mean_slice_coverage': round(mean_b / n_nodes, 3),
        'n_criteria': len(crit_nodes),
    }
    return stats, criterion, bslice_set, list(bvals)


def _enclosing_class_local(path_nodes, jlt):
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
    from build_cpg_features import CPGBuilder
    src  = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    tree = javalang.parse.parse(src)
    has_pkg = any(isinstance(n, jlt.PackageDeclaration) for _, n in tree)

    rows = []
    bsize_sample = []
    kind_totals = Counter()
    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class_local(path_nodes, jlt)
        kind = 'constructor' if isinstance(node, jlt.ConstructorDeclaration) else 'method'
        mid  = f"{rel_path}::{cls}::{node.name}@{node.position.line}"
        b = CPGBuilder(nx, jlt)
        G, entry = b.build(node)
        st, _crit, _bs, bvals = compute_slices(G, entry)
        bsize_sample.extend(bvals)
        rows.append({
            'method_id': mid, 'file': rel_path,
            'class': cls, 'method': node.name, 'kind': kind, **st,
            'has_method_link': True, 'has_class_ancestor': cls != '<unknown>',
            'has_file_ancestor': True, 'has_pkg_ancestor': bool(has_pkg),
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'bsize_sample': bsize_sample}, f)


def _worker_graph(abs_path, rel_path, wanted_path, out_path):
    import javalang, javalang.tree as jlt, networkx as nx
    from build_cpg_features import CPGBuilder
    wanted = json.load(open(wanted_path, encoding='utf-8'))
    src  = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    tree = javalang.parse.parse(src)
    parsed = defaultdict(list)
    for path_nodes, node in tree:
        if not isinstance(node, (jlt.MethodDeclaration, jlt.ConstructorDeclaration)):
            continue
        if node.position is None or getattr(node, 'body', None) is None:
            continue
        cls  = _enclosing_class_local(path_nodes, jlt)
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
            _st, criterion, bslice_set, _bv = compute_slices(G, entry)
            safe = "".join(c if c.isalnum() else "_" for c in mid)
            (EX_GRAPH_DIR / f"{safe}.pkl").write_bytes(
                pickle.dumps((G, entry, criterion, sorted(bslice_set)), protocol=4))
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

    all_rows = []; bsize_all = []
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
            bsize_all.extend(res['bsize_sample'])
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
                 'n_pdg_nodes','n_stmt_nodes','n_dep_edges',
                 'max_backward_slice','mean_backward_slice',
                 'max_forward_slice','mean_forward_slice','return_slice_size',
                 'slice_coverage','mean_slice_coverage','n_criteria',
                 'has_method_link','has_class_ancestor','has_file_ancestor','has_pkg_ancestor']
    with open(OUT_DIR / "slice_subgraph_stats.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=col_order); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k, '') for k in col_order})
    with open(OUT_DIR / "slice_rows.json", 'w', encoding='utf-8') as f:
        json.dump(all_rows, f)
    # cap the slice-size sample to keep the aggregate file small
    if len(bsize_all) > 20000:
        step = len(bsize_all) // 20000
        bsize_all = bsize_all[::step]
    with open(OUT_DIR / "slice_aggregates.json", 'w', encoding='utf-8') as f:
        json.dump({'bsize_sample': bsize_all, 'n_methods': len(all_rows),
                   'ok_files': len(ok_files), 'skipped_files': skipped_files}, f)

    mb = [r['max_backward_slice'] for r in all_rows]
    sc = [r['slice_coverage'] for r in all_rows]
    print(f"\nMax backward slice : min={min(mb)} med={sorted(mb)[len(mb)//2]} "
          f"mean={sum(mb)/len(mb):.1f} max={max(mb)}")
    print(f"Slice coverage     : mean={sum(sc)/len(sc):.2f} max={max(sc):.2f}")
    print(f"Return slice size  : mean="
          f"{sum(r['return_slice_size'] for r in all_rows)/len(all_rows):.1f} "
          f"max={max(r['return_slice_size'] for r in all_rows)}")

    # ── Phase 2: example slice graphs (large, interesting backward slices) ──
    ok_set = set(ok_files)
    viz_ids = []
    for r in sorted([r for r in all_rows
                     if r['file'] in ok_set and 8 <= r['n_pdg_nodes'] <= 42
                     and r['max_backward_slice'] >= 3],
                    key=lambda r: -r['max_backward_slice'])[:6]:
        viz_ids.append(r['method_id'])
    cc = Counter(r['class'] for r in all_rows
                 if r['file'] in ok_set and 7 <= r['n_pdg_nodes'] <= 48
                 and r['max_backward_slice'] >= 2)
    if cc:
        tcls = cc.most_common(1)[0][0]
        for r in sorted([r for r in all_rows if r['class'] == tcls and r['file'] in ok_set
                         and 7 <= r['n_pdg_nodes'] <= 48 and r['max_backward_slice'] >= 2],
                        key=lambda r: -r['max_backward_slice'])[:6]:
            if r['method_id'] not in viz_ids: viz_ids.append(r['method_id'])

    by_mid = {r['method_id']: r for r in all_rows}
    file_to_wanted = defaultdict(list)
    for mid in viz_ids:
        r = by_mid[mid]
        file_to_wanted[r['file']].append([r['class'], r['method'], r['kind'], mid])

    print(f"\nBuilding {len(viz_ids)} example slice graphs from {len(file_to_wanted)} files...", flush=True)
    built = 0
    for rel, wanted in file_to_wanted.items():
        fd, wp = tempfile.mkstemp(suffix='.json'); os.close(fd)
        with open(wp, 'w', encoding='utf-8') as f: json.dump(wanted, f)
        res = run_worker('--worker-graph', [str(REPO_PATH / rel), rel, wp], PARSE_TIMEOUT_S)
        try: os.unlink(wp)
        except OSError: pass
        if isinstance(res, dict): built += len(res['saved'])
        else: print(f"  graph build failed for {rel}: {res}", flush=True)
    print(f"Example slice graphs saved : {built}  -> {EX_GRAPH_DIR}", flush=True)

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
