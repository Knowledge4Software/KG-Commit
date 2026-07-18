"""
Cross-project aggregation of the KG-Commit evaluation results.
===============================================================

Collects every project's per-project result files under ``outputs/<project>/``
and produces two kinds of GENERAL evaluation across all projects that have
results (projects without a folder or without a given result file are silently
skipped, so this can be re-run after each new project is built and it simply
picks up whatever is present):

  Type 1  "Macro"  -- the unweighted MEAN of a metric across projects
                     (every project counts equally):
                         macro(m) = mean_i  m_i
  Type 2  "Total"  -- the commit-WEIGHTED mean of a metric across projects
                     (bigger projects count more):
                         total(m) = sum_i w_i * m_i  /  sum_i w_i
                     reported under TWO weightings of w_i:
                         w = n_eval   (scored commits = N - warmup; the honest
                                       denominator, since metrics are measured
                                       only over the evaluated stream)
                         w = N        (all labelled commits)

  Example (F1):  A: 100 commits F1=0.4 ; B: 200 commits F1=0.8
      Type 1 = (0.4+0.8)/2                 = 0.60
      Type 2 = (0.4*100 + 0.8*200)/300     = 0.67

NOTE on Type 2 vs a truly-pooled metric: F1/Precision/Recall/AUC are non-linear
in the confusion matrix, so a commit-weighted average of per-project scores is a
principled SUMMARY but is not identical to the metric recomputed on the pooled
per-commit predictions. Where a project stored its raw per-commit stream
(subgraph_rq_results.pkl -> "raw_stream"), we additionally compute a genuinely
POOLED Fusion metric set (see aggregate_pooled_fusion()).

Aggregated families (each skipped per-project if its file is absent):
  * final_experiments_results.pkl  -> 5 methods x 6 graphs, full metric leaf
  * final_fusion_results.pkl       -> deployed fusion F (+ the RN/PPR singles)
  * subgraph_rq_results.pkl        -> 6 subgraph variants (Fusion/T/P per variant)
  * subgraph_layer_stats.json      -> KG size/token tallies (summed + per-project)

Inputs : outputs/<project>/{final_experiments_results.pkl, final_fusion_results.pkl,
         subgraph_rq_results.pkl, subgraph_layer_stats.json}
Output : outputs/aggregate/aggregate_results.pkl   (everything, machine-readable)
         outputs/aggregate/aggregate_summary.json  (human-readable headline tables)
         outputs/aggregate/projects_index.json      (which projects/files were used)

Run (no KGC_PROJECT needed -- this spans all projects):
    python aggregate/aggregate_projects.py
    python aggregate/aggregate_projects.py --only activemq kafka   # restrict set
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# This aggregator is PROJECT-INDEPENDENT: it scans every project under outputs/.
# It must NOT require KGC_PROJECT (which config.project_config enforces), so we
# derive the repo root directly and only put inference/ on path for the optional
# pooled-metric helper (online_jit.final_metrics).
PKG_ROOT = Path(__file__).resolve().parent.parent          # .../kgcommit_repro
PROJECT_ROOT = PKG_ROOT.parent                             # .../KG-Commit
sys.path.insert(0, str(PKG_ROOT / "inference"))
sys.path.insert(0, str(PKG_ROOT))

OUTPUTS = PROJECT_ROOT / "outputs"
AGG_OUT = OUTPUTS / "aggregate"

# result files that mark a directory as a real, evaluated project
FE = "final_experiments_results.pkl"
FF = "final_fusion_results.pkl"
SQ = "subgraph_rq_results.pkl"
SL = "subgraph_layer_stats.json"
BL = "baseline_results.pkl"          # optional: baselines/run_baselines.py
EF = "effort_results.pkl"            # optional: inference/run_effort_eval.py
SCAL = "scalability"                 # dir holding E1-E5 JSONs

# the full metric leaf stored by final_experiments / fusion
METRIC_KEYS = ["ROC_AUC", "PR_AUC", "F1", "MCC", "Brier", "Acc", "F1_online",
               "Precision", "Recall", "Buggy_F1", "Macro_F1", "G_Mean",
               "AUC", "ACC"]
# extra metric keys used by baselines / effort eval
EFFORT_KEYS = ["Popt", "ACC20"]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def discover_projects(only=None):
    """Return sorted list of project names under outputs/ that have at least the
    final_experiments file. Non-project folders (plots, tables, ...) lack it and
    are skipped automatically."""
    projs = []
    for d in sorted(OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        if d.name in ("aggregate",):
            continue
        if only and d.name not in only:
            continue
        if (d / FE).exists():
            projs.append(d.name)
    return projs


def _load_pickle(p):
    try:
        return pickle.load(open(p, "rb"))
    except Exception as e:
        print(f"  !! failed to read {p}: {type(e).__name__}: {e}")
        return None


def _load_json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  !! failed to read {p}: {type(e).__name__}: {e}")
        return None


# --------------------------------------------------------------------------
# generic Type-1 / Type-2 combiner
# --------------------------------------------------------------------------
def combine(per_project):
    """per_project: list of (value, n_eval, N) with value possibly None/nan.
    Returns dict(macro, total_neval, total_N, n_projects, projects_used)."""
    vals, we, wN = [], [], []
    for v, ne, n in per_project:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        vals.append(float(v)); we.append(float(ne)); wN.append(float(n))
    if not vals:
        return dict(macro=None, total_neval=None, total_N=None, n_projects=0)
    vals = np.array(vals); we = np.array(we); wN = np.array(wN)
    return dict(
        macro=float(vals.mean()),
        total_neval=float((vals * we).sum() / we.sum()) if we.sum() > 0 else None,
        total_N=float((vals * wN).sum() / wN.sum()) if wN.sum() > 0 else None,
        n_projects=int(len(vals)),
    )


# --------------------------------------------------------------------------
# family 1: final_experiments  (methods x graphs x metrics)
# --------------------------------------------------------------------------
def aggregate_final_experiments(loaded):
    """loaded: {proj: (fe_dict, n_eval, N)}. Returns nested
    {graph: {method: {metric: combine(...)}}} plus the per-project raw table."""
    # union of graphs/methods/metrics actually present
    graphs, methods = set(), set()
    for proj, (fe, ne, N) in loaded.items():
        for g in fe.get("meta", {}).get("graphs", []):
            graphs.add(g)
        for m in fe.get("meta", {}).get("methods", []):
            methods.add(m)
    graphs = [g for g in ["core", "ast", "cfg", "dfg", "pdg", "seq", "final"] if g in graphs]
    methods = [m for m in ["RN", "PPR", "LP", "DW", "KGE"] if m in methods]

    out = {}
    per_project_rows = []          # flat rows for a CSV-like table
    for g in graphs:
        out[g] = {}
        for m in methods:
            per = []
            for proj, (fe, ne, N) in loaded.items():
                leaf = fe.get(g, {}).get(m, {}).get("metrics") if isinstance(fe.get(g), dict) else None
                for mk in METRIC_KEYS:
                    val = leaf.get(mk) if leaf else None
                    if leaf is not None:
                        per_project_rows.append(dict(project=proj, graph=g, method=m,
                                                     metric=mk, value=val,
                                                     n_eval=ne, N=N))
                per.append((leaf, ne, N))
            out[g][m] = {mk: combine([(leaf.get(mk) if leaf else None, ne, N)
                                      for (leaf, ne, N) in per])
                         for mk in METRIC_KEYS}
    return dict(graphs=graphs, methods=methods, metrics=METRIC_KEYS,
                aggregated=out, per_project_rows=per_project_rows)


# --------------------------------------------------------------------------
# family 2: deployed fusion F (+ singles present in part1)
# --------------------------------------------------------------------------
def aggregate_fusion(loaded_ff):
    """loaded_ff: {proj: (ff_dict, n_eval, N)}. Aggregates each fusion combo in
    part2 (F, F+G, F+M, F+G+M) -- F+G is the DEPLOYED model -- plus the single
    methods from part1, and records each project's chosen fusion composition.

    Note: F itself is a per-project-VARYING composition (e.g. PPR vs RN+PPR),
    chosen by the pipeline; aggregating 'F' across projects therefore aggregates
    each project's own best fusion. F+G = that fusion + the CSTG channel."""
    chosen = {}                       # proj -> chosen fusion composition
    combo_union = {}                  # combo-label (F/F+G/...) -> [(leaf, ne, N)]
    part1_union = {}                  # single method -> [(leaf, ne, N)]
    for proj, (ff, ne, N) in loaded_ff.items():
        chosen[proj] = ff.get("chosen") or ff.get("F")
        p2 = ff.get("part2")
        if isinstance(p2, dict):
            for combo, d in p2.items():
                if isinstance(d, dict) and isinstance(d.get("metrics"), dict):
                    combo_union.setdefault(combo, []).append((d["metrics"], ne, N))
        p1 = ff.get("part1")
        if isinstance(p1, dict):
            for lbl, d in p1.items():
                if isinstance(d, dict) and isinstance(d.get("metrics"), dict):
                    part1_union.setdefault(lbl, []).append((d["metrics"], ne, N))

    def agg_leaves(lst):
        return {mk: combine([(leaf.get(mk) if leaf else None, ne, N)
                             for (leaf, ne, N) in lst]) for mk in METRIC_KEYS}

    combos = {c: agg_leaves(lst) for c, lst in combo_union.items()}
    singles = {lbl: agg_leaves(lst) for lbl, lst in part1_union.items()}
    # deployed model = F+G (fusion + CSTG); fall back to F if F+G absent
    deployed = combos.get("F+G") or combos.get("F")
    return dict(deployed_F=deployed, deployed_label=("F+G" if "F+G" in combos else "F"),
                combos=combos, chosen_per_project=chosen, singles=singles)


# --------------------------------------------------------------------------
# family 3: subgraph RQ  (6 variants; Fusion/T_only/P_only leaves)
# --------------------------------------------------------------------------
def aggregate_subgraph_rq(loaded_sq):
    VAR = ["V1_none", "V2a_cfg", "V2b_dfg", "V2c_pdg", "V2d_seq", "V3_ast"]
    LEAVES = ["Fusion", "T_only", "P_only"]
    out = {}
    for v in VAR:
        out[v] = {}
        for lf in LEAVES:
            per = []
            for proj, (sq, ne, N) in loaded_sq.items():
                vd = sq.get(v, {})
                leaf = vd.get(lf) if isinstance(vd, dict) else None
                # subgraph n_eval per variant if present (else project n_eval)
                v_ne = vd.get("n_eval", ne) if isinstance(vd, dict) else ne
                per.append((leaf, v_ne, N))
            out[v][lf] = {mk: combine([(leaf.get(mk) if leaf else None, ne, N)
                                       for (leaf, ne, N) in per])
                          for mk in METRIC_KEYS}
    return dict(variants=VAR, leaves=LEAVES, aggregated=out)


# --------------------------------------------------------------------------
# family 4: subgraph layer stats (KG size; summed + per-project)
# --------------------------------------------------------------------------
def aggregate_layer_stats(loaded_sl):
    SUM_KEYS = ["nodes", "alive", "delta_inserted", "files", "ADDS", "REMOVES",
                "UPDATES", "MOVES", "delta_total", "commits_with_tokens",
                "n_token_types", "n_node_types"]
    layers, totals, per_project = set(), {}, {}
    for proj, (sl, ne, N) in loaded_sl.items():
        per_project[proj] = {}
        for layer, d in sl.items():
            if layer.startswith("_") or not isinstance(d, dict):
                continue                       # skip _meta and non-layer entries
            layers.add(layer)
            per_project[proj][layer] = {k: d.get(k) for k in SUM_KEYS if k in d}
            for k in SUM_KEYS:
                if isinstance(d.get(k), (int, float)):
                    totals.setdefault(layer, {}).setdefault(k, 0)
                    totals[layer][k] += d[k]
    return dict(layers=sorted(layers), summed=totals, per_project=per_project)


# --------------------------------------------------------------------------
# genuinely-pooled Fusion metrics from raw per-commit streams (if present)
# --------------------------------------------------------------------------
def aggregate_pooled_fusion(loaded_sq):
    """If projects stored subgraph_rq 'raw_stream' (per-commit y + fusion_p), pool
    them ALL and compute one honest metric set on the concatenated commit set.
    This is the true 'metric on total commits' (vs Type-2's weighted average)."""
    ys, ps, used = [], [], []
    for proj, (sq, ne, N) in loaded_sq.items():
        # deployed-fusion raw stream lives on the AST variant (V3_ast) in v4
        rs = None
        for v in ("V3_ast", "V1_none"):
            vd = sq.get(v, {})
            if isinstance(vd, dict) and isinstance(vd.get("raw_stream"), dict):
                rs = vd["raw_stream"]; break
        if not rs:
            continue
        y = np.asarray(rs.get("y", []), int)
        p = np.asarray(rs.get("fusion_p", []), float)
        if len(y) and len(y) == len(p):
            ys.append(y); ps.append(p); used.append(proj)
    if not ys:
        return dict(available=False, note="no raw_stream in any project yet "
                    "(re-run experiments with the updated run_subgraph_rq.py)")
    Y = np.concatenate(ys); P = np.concatenate(ps)
    try:
        # online_jit.final_metrics is project-agnostic, but online_jit imports
        # config.project_config which enforces KGC_PROJECT at import time. Stub a
        # valid project name purely to satisfy that import (no per-project state
        # is used by final_metrics). Restore the env afterwards.
        import os
        _saved = os.environ.get("KGC_PROJECT")
        if not _saved:
            os.environ["KGC_PROJECT"] = used[0]
        try:
            from online_jit import final_metrics
            leaf = final_metrics(Y, np.nan_to_num(P, nan=float(Y.mean())))
            leaf = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in leaf.items()}
        finally:
            if _saved is None:
                os.environ.pop("KGC_PROJECT", None)
    except Exception as e:
        leaf = {"error": f"{type(e).__name__}: {e}"}
    return dict(available=True, projects_used=used, n_commits_pooled=int(len(Y)),
                pooled_bug_rate=float(Y.mean()), metrics=leaf)


# --------------------------------------------------------------------------
# family 5: baselines (per-project within-project JIT baselines)
# --------------------------------------------------------------------------
def aggregate_baselines(loaded_bl):
    """loaded_bl: {proj: (bl_dict, ne, N)}. Aggregate every baseline's metric
    leaf (incl. effort metrics) with Type-1/Type-2 across projects."""
    names, keys = set(), METRIC_KEYS + EFFORT_KEYS
    for proj, (bl, ne, N) in loaded_bl.items():
        names.update(bl.get("baselines", {}).keys())
    order = [b for b in ["B_LR", "B_RF", "B_HGB", "B_ALL1", "B_ALL0", "B_RATE"]
             if b in names] + [b for b in sorted(names)
                               if b not in ("B_LR", "B_RF", "B_HGB", "B_ALL1", "B_ALL0", "B_RATE")]
    out = {}
    for b in order:
        per_metric = {}
        for mk in keys:
            per = [(bl.get("baselines", {}).get(b, {}).get(mk), ne, N)
                   for proj, (bl, ne, N) in loaded_bl.items()]
            per_metric[mk] = combine(per)
        out[b] = per_metric
    return dict(baselines=order, metrics=keys, aggregated=out)


# --------------------------------------------------------------------------
# family 6: effort-aware KG-model evaluation (Popt / ACC@20)
# --------------------------------------------------------------------------
def aggregate_effort(loaded_ef):
    """loaded_ef: {proj: (ef_dict, ne, N)}. Aggregate Popt/ACC20 (+ the classif.
    metrics stored alongside) per KG model channel."""
    models, keys = set(), EFFORT_KEYS + ["Buggy_F1", "PR_AUC", "Macro_F1", "G_Mean", "AUC"]
    for proj, (ef, ne, N) in loaded_ef.items():
        models.update(ef.get("models", {}).keys())
    order = [m for m in ["Fusion(F+G)", "M_metrics", "T_tfidf", "R_priors",
                         "P_ppr", "G_cstg"] if m in models] + \
            [m for m in sorted(models) if m not in
             ("Fusion(F+G)", "M_metrics", "T_tfidf", "R_priors", "P_ppr", "G_cstg")]
    out = {}
    for m in order:
        out[m] = {mk: combine([(ef.get("models", {}).get(m, {}).get(mk), ne, N)
                               for proj, (ef, ne, N) in loaded_ef.items()])
                  for mk in keys}
    return dict(models=order, metrics=keys, aggregated=out)


# --------------------------------------------------------------------------
# family 7: scalability (E1-E5) rolled up across projects
# --------------------------------------------------------------------------
def _load_scal(proj_dir):
    d = {}
    for fn in ("kg_profile", "growth", "build_complexity",
               "prediction_latency", "significance"):
        p = proj_dir / SCAL / f"{fn}.json"
        if p.exists():
            d[fn] = _load_json(p)
    return d


def aggregate_scalability(loaded_scal):
    """loaded_scal: {proj: (scal_dict, ne, N)}. Rolls up the numeric parts that
    aggregate meaningfully: predict latency per method (macro mean of medians),
    build cost per layer (macro mean of ms/file mean), and KG size per layer."""
    GR = ["core", "ast", "cfg", "dfg", "pdg", "seq", "final"]
    # (a) predict latency: per graph x method, macro-mean the per-commit median ms
    lat = {}
    for g in GR:
        methods = set()
        for proj, (sc, ne, N) in loaded_scal.items():
            pl = sc.get("prediction_latency", {}).get(g, {})
            methods.update((pl.get("predict_ms_per_commit") or {}).keys())
        if not methods:
            continue
        lat[g] = {}
        for m in sorted(methods):
            per = []
            for proj, (sc, ne, N) in loaded_scal.items():
                cell = (sc.get("prediction_latency", {}).get(g, {})
                        .get("predict_ms_per_commit", {}).get(m))
                per.append((cell.get("median") if cell else None, ne, N))
            lat[g][m] = combine(per)
    # (b) build complexity: per layer, macro-mean of ms_per_file mean
    bc = {}
    for layer in ["ast", "cfg", "dfg", "pdg", "seq"]:
        per = []
        for proj, (sc, ne, N) in loaded_scal.items():
            cell = sc.get("build_complexity", {}).get(layer, {})
            mpf = (cell.get("ms_per_file") or {}).get("mean") if isinstance(cell, dict) else None
            per.append((mpf, ne, N))
        c = combine(per)
        if c["n_projects"]:
            bc[layer] = c
    # (c) per-project totals for context
    per_project = {}
    for proj, (sc, ne, N) in loaded_scal.items():
        prof = sc.get("kg_profile", {})
        per_project[proj] = {g: (prof.get(g, {}) or {}).get("nodes")
                             for g in GR if g in prof}
    return dict(predict_latency_ms=lat, build_ms_per_file=bc,
                per_project_nodes=per_project)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these project names")
    args = ap.parse_args()

    projs = discover_projects(args.only)
    if not projs:
        raise SystemExit("No projects with results under outputs/. Build one first.")
    print(f"Aggregating {len(projs)} project(s): {', '.join(projs)}")

    # load per-project, tracking which files exist
    index = {}
    fe_l, ff_l, sq_l, sl_l, bl_l, ef_l, scal_l = {}, {}, {}, {}, {}, {}, {}
    for p in projs:
        d = OUTPUTS / p
        fe = _load_pickle(d / FE) if (d / FE).exists() else None
        index[p] = dict(final_experiments=bool(fe))
        if fe is None:
            continue
        N = int(fe["meta"]["N"]); w = int(fe["meta"]["warmup"]); ne = N - w
        fe_l[p] = (fe, ne, N)
        if (d / FF).exists():
            ff = _load_pickle(d / FF)
            if ff is not None:
                ff_l[p] = (ff, ne, N); index[p]["final_fusion"] = True
        if (d / SQ).exists():
            sq = _load_pickle(d / SQ)
            if sq is not None:
                sq_l[p] = (sq, ne, N); index[p]["subgraph_rq"] = True
        if (d / SL).exists():
            sl = _load_json(d / SL)
            if sl is not None:
                sl_l[p] = (sl, ne, N); index[p]["subgraph_layer_stats"] = True
        if (d / BL).exists():
            bl = _load_pickle(d / BL)
            if bl is not None:
                bl_l[p] = (bl, ne, N); index[p]["baselines"] = True
        if (d / EF).exists():
            ef = _load_pickle(d / EF)
            if ef is not None:
                ef_l[p] = (ef, ne, N); index[p]["effort"] = True
        scal = _load_scal(d)
        if scal:
            scal_l[p] = (scal, ne, N); index[p]["scalability"] = True
        index[p].update(N=N, warmup=w, n_eval=ne)

    results = {
        "meta": dict(projects=projs, n_projects=len(projs),
                     metric_keys=METRIC_KEYS,
                     type1="macro (unweighted mean across projects)",
                     type2_neval="commit-weighted by n_eval=N-warmup",
                     type2_N="commit-weighted by N (all labelled commits)"),
        "projects_index": index,
        "final_experiments": aggregate_final_experiments(fe_l) if fe_l else None,
        "fusion": aggregate_fusion(ff_l) if ff_l else None,
        "subgraph_rq": aggregate_subgraph_rq(sq_l) if sq_l else None,
        "layer_stats": aggregate_layer_stats(sl_l) if sl_l else None,
        "pooled_fusion": aggregate_pooled_fusion(sq_l) if sq_l else None,
        "baselines": aggregate_baselines(bl_l) if bl_l else None,
        "effort": aggregate_effort(ef_l) if ef_l else None,
        "scalability": aggregate_scalability(scal_l) if scal_l else None,
    }

    AGG_OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(results, open(AGG_OUT / "aggregate_results.pkl", "wb"))

    # human-readable headline JSON: deployed-F + the winning method per graph
    summary = _headline_summary(results)
    json.dump(summary, open(AGG_OUT / "aggregate_summary.json", "w"), indent=2)
    json.dump(index, open(AGG_OUT / "projects_index.json", "w"), indent=2)

    print(f"\nsaved -> {AGG_OUT / 'aggregate_results.pkl'}")
    print(f"saved -> {AGG_OUT / 'aggregate_summary.json'}")
    print(f"saved -> {AGG_OUT / 'projects_index.json'}")
    _print_headline(results)


def _headline_summary(results):
    s = {"projects": results["meta"]["projects"],
         "n_projects": results["meta"]["n_projects"]}
    if results.get("fusion"):
        s["deployed_fusion"] = results["fusion"]["deployed_F"]
        s["chosen_fusion_per_project"] = results["fusion"]["chosen_per_project"]
    if results.get("final_experiments"):
        fe = results["final_experiments"]
        # for the 'final' graph, each method's Buggy_F1 under all three views
        g = "final" if "final" in fe["graphs"] else fe["graphs"][-1]
        s[f"{g}_graph_BuggyF1"] = {
            m: fe["aggregated"][g][m]["Buggy_F1"] for m in fe["methods"]}
    if results.get("pooled_fusion", {}).get("available"):
        s["pooled_fusion"] = results["pooled_fusion"]
    return s


def _fmt(c):
    if not c or c.get("macro") is None:
        return "  --  "
    return f"{c['macro']:.3f}"


def _print_headline(results):
    print("\n" + "=" * 70)
    print("HEADLINE  (Type-1 macro mean across projects; see .json for Type-2)")
    print("=" * 70)
    fus = results.get("fusion")
    if fus:
        print("\nDeployed Fusion F  (chosen per project):")
        for p, c in fus["chosen_per_project"].items():
            print(f"    {p:12s} -> {c}")
        print("\n  metric          macro   total(n_eval)  total(N)")
        for mk in ["Buggy_F1", "Macro_F1", "G_Mean", "AUC", "PR_AUC", "MCC"]:
            c = fus["deployed_F"][mk]
            tn = f"{c['total_neval']:.3f}" if c and c.get("total_neval") is not None else "  -- "
            tN = f"{c['total_N']:.3f}" if c and c.get("total_N") is not None else "  -- "
            print(f"    {mk:14s}  {_fmt(c)}      {tn}      {tN}")
    pf = results.get("pooled_fusion", {})
    if pf.get("available"):
        print(f"\nGenuinely-pooled Fusion over {pf['n_commits_pooled']} commits "
              f"({len(pf['projects_used'])} projects), bug rate {pf['pooled_bug_rate']:.3f}:")
        for mk in ["Buggy_F1", "Macro_F1", "G_Mean", "AUC", "PR_AUC", "MCC"]:
            v = pf["metrics"].get(mk)
            print(f"    {mk:14s}  {v:.3f}" if isinstance(v, float) else f"    {mk:14s}  --")
    else:
        print(f"\n(pooled-fusion: {pf.get('note', 'n/a')})")


if __name__ == "__main__":
    main()
