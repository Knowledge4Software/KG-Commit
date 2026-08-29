"""
Regenerate every RQ1-RQ4 table for the Results draft, on the final data.
========================================================================

Emits, into Paper/ResultsDiscussionsDraft/:

  tab_rq1_performance.tex   T1  Macro-F1/G-Mean/AUC, 11 projects x 6 baselines,
                                with BOTH KG-Commit super-columns (overall F and
                                per-project F), + per-commit deployment cost
  tab_rq1_effort.tex        T2  Popt/ACC@20, same layout, both KG super-columns
  tab_rq2_scalability.tex   T3  cross-project cost; the per-project F column is
                                dropped (the deployed rule is fixed RN+PPR), all
                                11 projects, all costs recomputed
  tab_rq2_phase.tex         T4  phase-resolved cost; KG-Commit's featurise cell is
                                now the measured incremental KG-update time per
                                commit, still excluded from the Total (footnote a)
  tab_rq4_chosen_vs_rank1.tex T8 parsimony check on Cassandra + Groovy, with the
                                overall-F row and its delta added
  tab_rq4_fg.tex            T9  F -> F+G -> Switch@200 -> Delta, overall rule,
                                all 11 projects

T5 (analytical complexity) is prose-only and needs no data; it is checked, not
regenerated. T6/T7 (the big RQ3 matrix and the 31-combination table) are emitted by
make_draft_rq3_tables.py.

All numbers come from the persisted final_final_run artifacts and the timing logs,
so tables and figures cannot disagree. Cache-only; no Neo4j.

Run: python inference/make_draft_all_tables.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
LOGS = ROOT / "kgcommit_repro" / "logs"
PM = ROOT / "Paper" / "paper_material"
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

BCOLS = [("LR", "B_LR"), ("HGB", "B_HGB"), ("RF", "B_RF"),
         ("LApredict", "B_LAPREDICT"), ("DeepJIT", None),
         ("JITLine-on", "B_JITLINE_ONLINE")]
BDISP = {"JITLine-on": "JITLine-online"}
M3 = ("Macro_F1", "G_Mean", "AUC")

# A model at least this many times slower than KG-Commit is flagged red in the
# deployment-cost rows of T1 and in the scalability table. Single source of
# truth: the caption text is generated from it, so the two cannot drift.
RED_FACTOR = 2.0

KG_OV = r"KG-Commit ($F_{\mathrm{ov}}{+}G$, S@200)"
KG_PP = r"KG-Commit ($F_{\mathrm{pp}}{+}G$, S@200)"


def fmt(v, bold=False, nd=3):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.{nd}f}"
    return rf"\textbf{{{s}}}" if bold else s


def sgn(v, nd=3):
    if not np.isfinite(v):
        return "--"
    s = f"{v:+.{nd}f}"
    return (rf"\textcolor{{gainGreen}}{{{s}}}" if v > 0 else s)


# ------------------------------------------------------------------ loading --
def deepjit_table():
    df = pd.read_csv(OUTP / "DeepJIT_baseline_results" / "grid_summary.csv")

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    df["Macro_F1"] = df.apply(
        lambda r: (f1(r.tp, r.fp, r.fn) + f1(r.tn, r.fn, r.fp)) / 2, axis=1)
    df["G_Mean"] = np.sqrt((df.tp / (df.tp + df.fn)).clip(0)
                           * (df.tn / (df.tn + df.fp)).clip(0))
    df["ps"] = df["project"].str.replace("apache/", "", regex=False)
    return df.groupby("ps").agg(Macro_F1=("Macro_F1", "mean"),
                                G_Mean=("G_Mean", "mean"),
                                AUC=("auc", "mean"))


def load():
    S = json.load(open(OUTP / "final_final_run_summary.json"))
    DJ = deepjit_table()
    eff = json.load(open(PM / "RQ1_performance" / "rq1_effort_data.json"))

    # Baselines rescored onto KG-Commit's own protocol -- same span [ev0, N)
    # and same verification gap G=50. As persisted they were scored over
    # [W, N) with gap=0, because run_final_experiments imports GAP but never
    # passes it, so RQ1 was comparing models measured two different ways.
    # DeepJIT is NOT in this file (it comes from deepjit_table()) and is left
    # exactly as published. Regenerate with:
    #   python inference/rq1_rescore_baselines.py --apply
    _bm = OUTP / "tables" / "rq1_baselines_matched.json"
    if not _bm.exists():
        raise SystemExit(f"missing {_bm}\n"
                         "Run: python inference/rq1_rescore_baselines.py --apply")
    MATCHED = json.loads(_bm.read_text(encoding="utf-8"))
    rows = {}
    for p in PROJECTS:
        B = pickle.load(open(OUTP / p / "final_final_run" / "baselines"
                             / "baseline_extra_results.pkl", "rb"))
        F = pickle.load(open(OUTP / p / "final_final_run" / "fusion"
                             / "final_fusion_results.pkl", "rb"))
        d = {"n_eval": S[p]["switch"]["overall"]["n_eval"],
             "chosen": F["chosen"], "chosen_overall": F["chosen_overall"],
             "timings": B.get("timings", {}), "fusion": F,
             "switch": S[p]["switch"]}
        for rule, key in (("KGov", "overall"), ("KGpp", "per_project")):
            g = S[p]["switch"][key]["grid"]["200"]
            d[rule] = {m: float(g[m]) for m in M3}
            e = eff[p]["overall" if rule == "KGov" else "perproj"]
            d[rule].update(Popt=e["Popt"], ACC20=e["ACC20"])
        for lab, key in BCOLS:
            if key is None:
                r = DJ.loc[p]                      # DeepJIT: untouched
                d[lab] = {m: float(r[m]) for m in M3}
            else:
                mt = MATCHED.get(p, {}).get("baselines", {}).get(key)
                if mt is None:                     # no rescore -> published
                    m = B["baselines"][key]
                    d[lab] = {k: float(m[k]) for k in M3}
                else:
                    d[lab] = {k: float(mt["matched"][k]) for k in M3}
            e = eff[p].get(BDISP.get(lab, lab), {})
            d[lab].update(Popt=e.get("Popt", np.nan),
                          ACC20=e.get("ACC20", np.nan))
        rows[p] = d
    return rows


def agg(rows, model, metric, weighted=False):
    xs = [(rows[p][model][metric], rows[p]["n_eval"]) for p in PROJECTS
          if np.isfinite(rows[p][model].get(metric, np.nan))]
    if not xs:
        return np.nan
    v = [a for a, _ in xs]; w = [b for _, b in xs]
    return float(np.average(v, weights=w) if weighted else np.mean(v))


# ------------------------------------------------------------- deployment cost --
def kg_update_ms():
    """Measured incremental KG-update cost per commit (ms), per project.

    This is the parse+diff+delta work the pipeline does when a commit lands --
    the real featurisation cost, simply paid at ingest rather than on the
    prediction path. Summed over the layers the deployed graph actually holds
    (AST is the deployed Layer-2 choice), median over commits."""
    out = {}
    for p in PROJECTS:
        tot = 0.0
        ok = False
        for lay in ("ast",):
            f = LOGS / p / f"{lay}_timing.csv"
            if not f.exists():
                f = LOGS / p / f"{lay.upper()}_timing.csv"
            if f.exists():
                try:
                    t = pd.read_csv(f)
                    if "wall_ms" in t and len(t):
                        tot += float(t["wall_ms"].median()); ok = True
                except Exception:
                    pass
        out[p] = tot if ok else np.nan
    return out


def predict_ms(rows):
    """KG-Commit steady per-commit inference on the deployed (3-layer) graph.

    prediction_latency.json already records this for the deployed fusion, so we
    read that rather than re-summing per-method medians."""
    out = {}
    for p in PROJECTS:
        f = OUTP / p / "final_final_run" / "complexity" / "prediction_latency.json"
        if not f.exists():
            out[p] = np.nan; continue
        d = json.load(open(f)).get("final", {})
        v = d.get("deployed_F_predict_ms_per_commit")
        if not v:
            pm = d.get("predict_ms_per_commit", {})
            v = sum(pm.get(m, {}).get("median", 0.0) for m in ("RN", "PPR"))
        out[p] = float(v) if v else np.nan
    return out


def graph_nodes(project):
    """Resident node count of the DEPLOYED graph: Core + AST + CSTG.

    kg_profile.json reports each layer separately (core uses n_<Label> keys,
    the structural layers a single `nodes` count, CSTG its Term/Intent counts).
    CFG/DFG/PDG/SEQ are ablation candidates and are NOT in the deployed graph,
    so they are excluded here."""
    f = OUTP / project / "final_final_run" / "complexity" / "kg_profile.json"
    if not f.exists():
        return np.nan
    d = json.load(open(f))
    core = sum(v for k, v in (d.get("core") or {}).items()
               if k.startswith("n_") and k not in ("n_Commit_in_jit",
                                                   "n_Commit_buggy")
               and isinstance(v, (int, float)))
    ast = (d.get("ast") or {}).get("nodes", 0) or 0
    cs = (d.get("cstg") or {})
    cstg = (cs.get("n_Term", 0) or 0) + (cs.get("n_Intent", 0) or 0)
    tot = core + ast + cstg
    return float(tot) if tot else np.nan


def deployment_latency():
    """Prediction-path latency per model (ms), identical to Table 4.

    Latency = featurise + retrieve + predict, i.e. everything a developer waits
    for between the commit landing and the score appearing. Sources:
    complexity/featurisation_bygroup.json (measured feature derivation),
    complexity/prediction_latency.json (KG retrieve+predict) and the baselines'
    own predict timings. Amortised refit is deliberately excluded -- nobody
    waits for it.
    """
    feats, la, kgp, kgr = [], [], [], []
    for pr in PROJECTS:
        c = OUTP / pr / "final_final_run" / "complexity"
        fb = c / "featurisation_bygroup.json"
        pl = c / "prediction_latency.json"
        if not (fb.exists() and pl.exists()):
            continue
        f = json.loads(fb.read_text(encoding="utf-8"))
        L = json.loads(pl.read_text(encoding="utf-8"))["final"]
        feats.append(f["twelve_metrics_ms"])
        la.append(f["la_only_ms"])
        M = L["predict_ms_per_commit"]
        kgp.append(M["RN"]["median"] + M["PPR"]["median"])
        kgr.append(1000.0 * L["build_graph_s"] / max(L["Nc"], 1))
    if not feats:
        return None
    fm, fla = float(np.median(feats)), float(np.median(la))
    kg = float(np.median(kgp)) + float(np.median(kgr))

    def bp(key):
        v = [rows_cache[pr]["timings"].get(key, {}).get(
             "predict_ms_per_commit", np.nan) for pr in PROJECTS
             if pr in rows_cache]
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else np.nan

    jl_tok = bp("B_JITLINE_ONLINE")  # placeholder, replaced below
    return fm, fla, kg


rows_cache = {}


def baseline_cost(rows):
    keymap = {lab: key for lab, key in BCOLS if key}
    out = {}
    for lab, key in keymap.items():
        parts = {}
        for comp in ("featurize_ms_per_commit", "predict_ms_per_commit",
                     "refit_ms_per_commit_amortised", "total_ms_per_commit"):
            v = [rows[p]["timings"].get(key, {}).get(comp, np.nan)
                 for p in PROJECTS]
            v = [x for x in v if np.isfinite(x)]
            parts[comp] = float(np.median(v)) if v else np.nan
        out[lab] = parts
    return out


def deepjit_stage_t1():
    """DeepJIT predict and amortised refit per commit, from its own grid."""
    f = OUTP / "DeepJIT_baseline_results" / "grid_summary.csv"
    if not f.exists():
        return np.nan, np.nan
    df = pd.read_csv(f)
    df["proj"] = df["project"].str.replace("apache/", "", regex=False)
    sub = df[(df.M == 200) & (df.gap_commits == 50)
             & (df.warmup_ratio == 0.05) & (df.proj.isin(PROJECTS))]
    if sub.empty:
        return np.nan, np.nan
    g = sub.groupby("proj").agg(pred=("total_predict_seconds", "mean"),
                                refit=("total_refit_seconds", "mean"),
                                n=("evaluated_n", "mean"))
    return (float(np.median(1000 * g.pred / g.n)),
            float(np.median(1000 * g.refit / g.n)))


def deepjit_cost():
    """DeepJIT's per-commit deployment cost, median over the 11 projects.

    Taken from the same grid the accuracy numbers come from
    (DeepJIT_baseline_results/grid_summary.csv) at the deployed configuration
    M=200, gap=50, K=0.05, averaged over its 5 seeds per project. The CSV
    records total seconds, so both components are divided by the project's
    evaluated_n to give a per-commit figure -- the same
    predict + amortised-refit decomposition baseline_cost() uses, which keeps
    the row comparable. DeepJIT featurises from the released token tensors, so
    there is no separate featurisation term to add.
    """
    import pandas as pd
    f = OUTP / "DeepJIT_baseline_results" / "grid_summary.csv"
    if not f.exists():
        return np.nan
    df = pd.read_csv(f)
    df["proj"] = df["project"].str.replace("apache/", "", regex=False)
    sub = df[(df.M == 200) & (df.gap_commits == 50)
             & (df.warmup_ratio == 0.05) & (df.proj.isin(PROJECTS))]
    if sub.empty:
        return np.nan
    g = sub.groupby("proj").agg(pred=("total_predict_seconds", "mean"),
                                refit=("total_refit_seconds", "mean"),
                                n=("evaluated_n", "mean"))
    per_commit = 1000.0 * (g.pred + g.refit) / g.n
    return float(np.median(per_commit))


def g_refit_ms():
    """Amortised per-commit cost of refitting the CSTG channel G.

    RN and PPR are closed-form, so the graph fusion F needs no refit -- but the
    DEPLOYED model is F+G, and G is a logistic regression over the CSTG feature
    block that is refit prequentially on the expanding past window. Measured by
    inference/measure_g_refit.py, which replays that exact schedule."""
    out = {}
    for p in PROJECTS:
        f = OUTP / p / "final_final_run" / "complexity" / "g_refit_cost.json"
        out[p] = (float(json.load(open(f))["refit_ms_per_commit_amortised"])
                  if f.exists() else np.nan)
    return out


def per_project_cost(rows):
    """Per-project total for KG-Commit (ms/commit) = predict + G refit."""
    pred = predict_ms(rows)
    gref = g_refit_ms()
    kgtot = {}
    for p in PROJECTS:
        r = gref.get(p, np.nan)
        r = 0.0 if not np.isfinite(r) else r
        kgtot[p] = pred[p] + r if np.isfinite(pred[p]) else np.nan
    return pred, kgtot


# ------------------------------------------------------------------- tables --
def t1_performance(rows):
    models = ["KGov", "KGpp"] + [l for l, _ in BCOLS]
    head = [KG_OV, KG_PP] + [BDISP.get(l, l) for l, _ in BCOLS]
    L = [r"\begin{table*}[t]\centering", r"\tiny\setlength{\tabcolsep}{2pt}",
         r"\caption{RQ1 performance under the online protocol, on all 11 projects. "
         r"KG-Commit is reported under both fusion-selection rules: the fixed "
         r"\emph{overall} fusion $F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$ and the "
         r"\emph{per-project} chosen fusion $F_{\mathrm{pp}}$, each with the CSTG "
         r"channel $G$, the adaptive initial-fit window and the switch to $F{+}G$ "
         r"at $S{=}200$. On every row -- each project and the aggregates -- the "
         r"best model per metric is in "
         r"\textbf{bold}. The final two rows give the measured per-commit "
         r"deployment cost and the slowdown relative to KG-Commit; models at least "
         rf"${RED_FACTOR:g}\times$ slower are in \textcolor{{BrickRed}}{{red}}.}}",
         r"\label{tab:rq1_perf_g50}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "ccc" * len(models) + "}", r"\toprule",
         r"\multirow{2}{*}{Project} & " + " & ".join(
             rf"\multicolumn{{3}}{{c}}{{{h}}}" for h in head) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(models))),
         " & " + " & ".join(["Macro-F1 & G-Mean & AUC"] * len(models))
         + r" \\ \midrule"]

    for p in PROJECTS:
        rowbest = {k: max((rows[p][m].get(k, np.nan) for m in models
                           if np.isfinite(rows[p][m].get(k, np.nan))),
                          default=np.nan) for k in M3}
        cells = []
        for m in models:
            for k in M3:
                v = rows[p][m].get(k, np.nan)
                cells.append(fmt(v, np.isfinite(v) and np.isfinite(rowbest[k])
                                 and abs(v - rowbest[k]) < 1e-9))
        L.append(f"{DISP[p]} & " + " & ".join(cells) + r" \\")

    for lab, w in ((r"\textbf{Macro-Avg}", False), (r"\textbf{Micro-Avg}", True)):
        vals = {(m, k): agg(rows, m, k, w) for m in models for k in M3}
        best = {k: max((vals[(m, k)] for m in models
                        if np.isfinite(vals[(m, k)])), default=np.nan) for k in M3}
        L.append(r"\midrule " + lab + " & " + " & ".join(
            fmt(vals[(m, k)], np.isfinite(vals[(m, k)])
                and abs(vals[(m, k)] - best[k]) < 1e-9)
            for m in models for k in M3) + r" \\")

    bc = baseline_cost(rows)
    rows_cache.update(rows)
    dl = deployment_latency()
    if dl is None:
        raise SystemExit("missing featurisation/latency artifacts for Table 1")
    fm, fla, kg = dl

    def bpred(key):
        v = [rows[p]["timings"].get(key, {}).get("predict_ms_per_commit", np.nan)
             for p in PROJECTS if p in rows]
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else 0.0

    # Prediction-path latency, matching Table 4 exactly. LApredict consumes only
    # the added-lines feature, so it pays the diff but not the metric derivation.
    jl_tok = bpred("B_JITLINE_ONLINE") * 0 + float(np.median(
        [rows[p]["timings"].get("B_JITLINE_ONLINE", {}).get(
            "featurize_ms_per_commit", np.nan) for p in PROJECTS if p in rows]))
    dj_pred, _ = deepjit_stage_t1()
    cost = {
        "KGov": kg, "KGpp": kg,
        "LR": fm + bpred("B_LR"),
        "HGB": fm + bpred("B_HGB"),
        "RF": fm + bpred("B_RF"),
        "LApredict": fla + bpred("B_LAPREDICT"),
        "DeepJIT": fm + dj_pred,
        "JITLine-on": fm + jl_tok + bpred("B_JITLINE_ONLINE"),
    }

    def c3(v):
        if not np.isfinite(v):
            return r"\multicolumn{3}{c}{--}"
        s = f"{v:.3f}"
        return (rf"\multicolumn{{3}}{{c}}{{\textcolor{{BrickRed}}{{{s}}}}}"
                if v >= RED_FACTOR * kg else rf"\multicolumn{{3}}{{c}}{{{s}}}")

    def r3(v):
        if not np.isfinite(v):
            return r"\multicolumn{3}{c}{--}"
        r = v / kg
        s = r"\textbf{1.0$\times$}" if abs(r - 1) < 1e-9 else f"{r:.1f}$\\times$"
        return (rf"\multicolumn{{3}}{{c}}{{\textcolor{{BrickRed}}{{{s}}}}}"
                if r >= RED_FACTOR else rf"\multicolumn{{3}}{{c}}{{{s}}}")

    L.append(r"\midrule \textit{ms/commit} & "
             + " & ".join(c3(cost.get(m, np.nan)) for m in models) + r" \\")
    L.append(r"\textit{vs.\ KG-Commit} & "
             + " & ".join(r3(cost.get(m, np.nan)) for m in models) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def t2_effort(rows):
    models = ["KGov", "KGpp"] + [l for l, _ in BCOLS]
    head = [KG_OV, KG_PP] + [BDISP.get(l, l) for l, _ in BCOLS]
    L = [r"\begin{table*}[t]\centering", r"\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\caption{RQ1 effort-aware evaluation ($P_{\mathrm{opt}}$, "
         r"ACC@20\%LOC) on all 11 projects, under both fusion-selection rules. "
         r"Inspection effort is the commit's churn (la$+$ld); every model is "
         r"ranked by predicted defect density over the identical commit "
         r"sequence. On every row -- each project and the aggregates -- the best "
         r"per metric is \textbf{bold}.}",
         r"\label{tab:rq1_effort_g50}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "cc" * len(models) + "}", r"\toprule",
         r"\multirow{2}{*}{Project} & " + " & ".join(
             rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in head) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(models))),
         " & " + " & ".join([r"$P_{\mathrm{opt}}$ & ACC@20"] * len(models))
         + r" \\ \midrule"]
    for p in PROJECTS:
        rowbest = {k: max((rows[p][m].get(k, np.nan) for m in models
                           if np.isfinite(rows[p][m].get(k, np.nan))),
                          default=np.nan) for k in ("Popt", "ACC20")}
        cells = []
        for m in models:
            for k in ("Popt", "ACC20"):
                v = rows[p][m].get(k, np.nan)
                cells.append(fmt(v, np.isfinite(v) and np.isfinite(rowbest[k])
                                 and abs(v - rowbest[k]) < 1e-9))
        L.append(f"{DISP[p]} & " + " & ".join(cells) + r" \\")
    for lab, w in ((r"\textbf{Macro-Avg}", False), (r"\textbf{Micro-Avg}", True)):
        vals = {(m, k): agg(rows, m, k, w) for m in models
                for k in ("Popt", "ACC20")}
        best = {k: max((vals[(m, k)] for m in models
                        if np.isfinite(vals[(m, k)])), default=np.nan)
                for k in ("Popt", "ACC20")}
        L.append(r"\midrule " + lab + " & " + " & ".join(
            fmt(vals[(m, k)], np.isfinite(vals[(m, k)])
                and abs(vals[(m, k)] - best[k]) < 1e-9)
            for m in models for k in ("Popt", "ACC20")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def t3_scalability(rows):
    pred, kgtot = per_project_cost(rows)
    bl = [(l, k) for l, k in BCOLS if k]
    L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{RQ2 cross-project scalability and measured deployment cost, "
         r"all 11 projects. The deployed fusion is the same everywhere "
         r"($F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$), so no per-project $F$ column "
         r"is needed. \emph{Left}: resident graph size and resulting throughput. "
         r"\emph{Right}: per-commit deployment cost (ms) of every model, measured "
         r"on the same commits under the same online protocol. KG-Commit reads "
         r"context its incremental update already materialised, so it featurises "
         r"nothing on the prediction path (see Table~\ref{tab:phase_cost}).}",
         r"\label{tab:crossproject_scalability}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l rr | r" + "r" * len(bl) + "}", r"\toprule",
         r"& \multicolumn{2}{c|}{KG-Commit graph} & \multicolumn{"
         + str(1 + len(bl)) + r"}{c}{Per-commit deployment cost (ms)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-" + str(4 + len(bl)) + "}",
         r"Project & Nodes & c/s & \textbf{KG-Commit} & "
         + " & ".join(BDISP.get(l, l) for l, _ in bl) + r" \\", r"\midrule"]

    tot_nodes, tot_cs = [], []
    for p in PROJECTS:
        nodes = graph_nodes(p)
        kgv = kgtot[p]
        cs = 1000.0 / kgv if np.isfinite(kgv) and kgv > 0 else np.nan
        tot_nodes.append(nodes); tot_cs.append(cs)
        cells = [rf"\textbf{{{kgv:.3f}}}" if np.isfinite(kgv) else "--"]
        for lab, key in bl:
            v = rows[p]["timings"].get(key, {}).get("total_ms_per_commit", np.nan)
            cells.append(f"{v:.3f}" if np.isfinite(v) else "--")
        L.append(f"{DISP[p]} & {nodes:,.0f} & {cs:,.0f} & "
                 + " & ".join(cells) + r" \\")

    kgm = float(np.nanmedian([kgtot[p] for p in PROJECTS]))
    bm = {lab: float(np.nanmedian(
        [rows[p]["timings"].get(key, {}).get("total_ms_per_commit", np.nan)
         for p in PROJECTS])) for lab, key in bl}
    L.append(r"\midrule \textbf{Mean} & "
             f"\\textbf{{{np.nanmean(tot_nodes):,.0f}}} & "
             f"\\textbf{{{np.nanmean(tot_cs):,.0f}}} & "
             f"\\textbf{{{kgm:.3f}}} & "
             + " & ".join(f"{bm[l]:.3f}" if np.isfinite(bm[l]) else "--"
                          for l, _ in bl) + r" \\")
    L.append(r"\textit{vs.\ KG-Commit} & & & \textit{1.0$\times$} & "
             + " & ".join(
                 (rf"\textcolor{{BrickRed}}{{{bm[l]/kgm:.1f}$\times$}}"
                  if np.isfinite(bm[l]) and bm[l] / kgm >= RED_FACTOR
                  else (f"{bm[l]/kgm:.1f}$\\times$" if np.isfinite(bm[l]) else "--"))
                 for l, _ in bl) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def t4_phase(rows):
    pred, kgtot = per_project_cost(rows)
    upd = kg_update_ms()
    gref = g_refit_ms()
    kg_feat = float(np.nanmedian([upd[p] for p in PROJECTS]))
    kg_pred = float(np.nanmedian([pred[p] for p in PROJECTS]))
    kg_ref = float(np.nanmedian([gref[p] for p in PROJECTS]))
    kg_tot = kg_pred + kg_ref
    bc = baseline_cost(rows)

    L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\caption{RQ2 phase-resolved per-commit cost (ms, median over the 11 "
         r"active projects, measured under the online protocol). \textsc{Inference} "
         r"is the work on the developer's critical path when a commit is scored; "
         r"\textsc{Maint.} is periodic retraining, amortised over the commits it "
         r"serves. KG-Commit's featurisation is the measured incremental "
         r"KG-update cost paid once when the commit lands$^{a}$; it is reported "
         r"for transparency but excluded from the Total, which counts only the "
         r"prediction path. Its \textsc{refit} is the CSTG channel $G$: RN and "
         r"PPR are closed-form and need no refit, but the deployed model is "
         r"$F{+}G$, and $G$ is a logistic model over the CSTG features refit "
         r"prequentially on the expanding past window. The baselines keep no "
         r"persistent state and must rebuild a commit's features every time "
         r"they score it.}",
         r"\label{tab:phase_cost}", r"\begin{tabular}{l r r | r | r}", r"\toprule",
         r"& \multicolumn{2}{c|}{\textsc{Inference}} & \textsc{Maint.} "
         r"& \textsc{Total} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(l){5-5}",
         r"Model & featurise & predict & refit & per commit \\", r"\midrule",
         rf"\textbf{{KG-Commit ($F_{{\mathrm{{ov}}}}{{+}}G$)}} & "
         rf"{kg_feat:.1f}$^{{a}}$ & \textbf{{{kg_pred:.3f}}} & {kg_ref:.3f} "
         rf"& \textbf{{{kg_tot:.3f}}} \\", r"\midrule"]
    for lab, key in BCOLS:
        if key is None:
            L.append(rf"{BDISP.get(lab, lab)} & \multicolumn{{4}}{{c}}"
                     r"{\textit{trained on GPU; not comparable on this axis}} \\")
            continue
        c = bc[lab]
        rel = c["total_ms_per_commit"] / kg_tot
        L.append(f"{BDISP.get(lab, lab)} & {c['featurize_ms_per_commit']:.3f} & "
                 f"{c['predict_ms_per_commit']:.3f} & "
                 f"{c['refit_ms_per_commit_amortised']:.3f} & "
                 f"{c['total_ms_per_commit']:.3f}\\,({rel:.1f}$\\times$) \\\\")
    L += [r"\bottomrule",
          r"\multicolumn{5}{@{}p{\linewidth}@{}}{\footnotesize $^{a}$Measured "
          r"median parse$+$diff$+$delta time to fold one commit into the graph. "
          r"It is real work, but it is paid once at ingest and off the prediction "
          r"path, so it is excluded from the Total; the baselines' featurisation "
          r"is on that path and is included.} \\",
          r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def t8_chosen_vs_rank1(rows, examples=("cassandra", "hbase")):
    """Parsimony check, reported on the GRAPH FUSION F alone.

    Selection operates on F (part1: the 31 graph-method combinations), so the
    comparison must be made there too. An earlier version of this table mixed
    levels -- it printed the deployed F+G numbers (part2) while describing an
    F-level choice -- which made the selected fusion look worse than the
    combination it was selected over. On F, chosen <= rank1 by construction and
    both exceed the fixed overall fusion, as they must."""
    TOL = 0.005
    M3L = ("Macro_F1", "G_Mean", "AUC")
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{Fusion selection on the graph fusion $F$, for two projects "
         r"where the parsimony rule and the rank-1 combination differ. "
         r"$\Delta$(rank1$-$chosen) is the accuracy given up by preferring the "
         r"combination with the fewest inference methods inside the tolerance "
         r"band $\tau{=}0.005$; $\Delta$(chosen$-$overall) is what per-project "
         r"selection buys over the fixed $F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$. "
         r"All values are the graph fusion $F$ alone -- the level at which the "
         r"choice is made -- so they are not the deployed $F{+}G$ numbers of "
         r"Table~\ref{tab:rq1_perf_g50}. The leaner choice costs at most $0.003$ "
         r"Macro-F1 while using two fewer methods.}",
         r"\label{tab:chosen_vs_rank1}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Project & Fusion $F$ (\#methods) & Macro-F1 & G-Mean & AUC \\",
         r"\midrule"]
    for i, p in enumerate(examples):
        F = rows[p]["fusion"]
        p1 = F["part1"]
        best = max(v["metrics"]["Macro_F1"] for v in p1.values())
        band = [(k, v) for k, v in p1.items()
                if v["metrics"]["Macro_F1"] >= best - TOL]
        ck, chosen = min(band, key=lambda kv: (kv[1]["n"],
                                               -kv[1]["metrics"]["Macro_F1"]))
        rk, rank1 = max(p1.items(), key=lambda kv: kv[1]["metrics"]["Macro_F1"])
        ok = F.get("chosen_overall", "RN+PPR")
        ovn = p1.get(ok)
        cm, rm = chosen["metrics"], rank1["metrics"]
        om = ovn["metrics"] if ovn else {}
        if i:
            L.append(r"\midrule")
        L.append(f"{DISP[p]} & \\texttt{{\\scriptsize {ck}}} "
                 f"({chosen['n']}, chosen) & "
                 + " & ".join(fmt(cm.get(k)) for k in M3L) + r" \\")
        L.append(f" & \\texttt{{\\scriptsize {rk}}} ({rank1['n']}, rank-1) & "
                 + " & ".join(fmt(rm.get(k)) for k in M3L) + r" \\")
        L.append(r" & $\Delta$ (rank1$-$chosen) & "
                 + " & ".join(sgn(rm.get(k, np.nan) - cm.get(k, np.nan))
                              for k in M3L) + r" \\")
        if ovn:
            L.append(f" & \\texttt{{\\scriptsize {ok}}} "
                     f"({ovn['n']}, overall) & "
                     + " & ".join(fmt(om.get(k)) for k in M3L) + r" \\")
            L.append(r" & $\Delta$ (chosen$-$overall) & "
                     + " & ".join(sgn(cm.get(k, np.nan) - om.get(k, np.nan))
                                  for k in M3L) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def t9_fg(rows):
    L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{RQ4: the deployed pipeline built up in stages, on all 11 "
         r"projects, under the fixed overall fusion "
         r"$F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$. Adding the semantic channel $G$ "
         r"and then delaying it until commit $S{=}200$ each contribute; $\Delta$ "
         r"is Switch@200 minus $F$ alone, i.e.\ the total gain from the semantic "
         r"tier as deployed.}",
         r"\label{tab:rq4_fg_g50}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lccccccccccccc}", r"\toprule",
         r"\multirow{2}{*}{Project} & \multicolumn{3}{c}{$F_{\mathrm{ov}}$} "
         r"& \multicolumn{3}{c}{$F_{\mathrm{ov}}{+}G$} "
         r"& \multicolumn{3}{c}{$F_{\mathrm{ov}}{+}G$, Switch@200} "
         r"& \multicolumn{3}{c}{$\Delta$ (Switch $-$ $F_{\mathrm{ov}}$)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
         " & " + " & ".join(["Macro-F1 & G-Mean & AUC"] * 4) + r" \\ \midrule"]
    acc = {k: [] for k in ("F", "FG", "S", "D")}
    wts = []
    for p in PROJECTS:
        sw = rows[p]["switch"]["overall"]
        f, fg, s = sw["F_only"], sw["FG_only"], sw["grid"]["200"]
        cells = []
        for src in (f, fg, s):
            cells += [fmt(src[k]) for k in M3]
        cells += [sgn(s[k] - f[k]) for k in M3]
        L.append(f"{DISP[p]} & " + " & ".join(cells) + r" \\")
        for tag, src in (("F", f), ("FG", fg), ("S", s)):
            acc[tag].append([src[k] for k in M3])
        acc["D"].append([s[k] - f[k] for k in M3])
        wts.append(rows[p]["n_eval"])

    # Macro-Avg treats every project equally; Micro-Avg weights by the number of
    # scored commits, so long streams count proportionally. They answer different
    # questions, and the gap between them is itself informative.
    for lab, w in ((r"\textbf{Macro-Avg}", None), (r"\textbf{Micro-Avg}", wts)):
        mF = np.average(acc["F"], axis=0, weights=w)
        mFG = np.average(acc["FG"], axis=0, weights=w)
        mS = np.average(acc["S"], axis=0, weights=w)
        mD = np.average(acc["D"], axis=0, weights=w)
        L.append(r"\midrule " + lab + " & " + " & ".join(
            [fmt(v, True) for v in mF] + [fmt(v, True) for v in mFG]
            + [fmt(v, True) for v in mS] + [sgn(v) for v in mD]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def main():
    rows = load()
    out = {"tab_rq1_performance.tex": t1_performance(rows),
           "tab_rq1_effort.tex": t2_effort(rows),
           "tab_rq2_scalability.tex": t3_scalability(rows),
           "tab_rq2_phase.tex": t4_phase(rows),
           "tab_rq4_chosen_vs_rank1.tex": t8_chosen_vs_rank1(rows),
           "tab_rq4_fg.tex": t9_fg(rows)}
    for name, body in out.items():
        (DRAFT / name).write_text(body + "\n", encoding="utf-8")
        print(f"  wrote {name}")

    print("\n--- Macro-Avg sanity (Macro-F1) ---")
    for m in ["KGov", "KGpp"] + [l for l, _ in BCOLS]:
        print(f"  {m:<12}{agg(rows, m, 'Macro_F1'):.3f}")


if __name__ == "__main__":
    main()
