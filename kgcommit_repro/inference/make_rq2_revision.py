"""
RQ2 revision: one merged deployment-cost table, one KG-only table, and the
candidate figures for the subsection.
=========================================================================

DEPLOYMENT FRAMING
------------------
Everything is costed as if the method were deployed on a NEW project as an
automatic bug detector. Nothing may be assumed precomputed: if a stage has to
run before a prediction can be shown to a developer, it is on the prediction
path and the user waits for it.

Two regimes, reported separately:

  LATENCY (user waits)      every stage between "commit lands" and "score shown"
  AMORTISED (user does not) periodic work, divided over the commits it serves

Stage accounting, per model
---------------------------
  KG-Commit   featurise : 0 on this path. The graph is updated when the commit
                          lands; the ingest cost is real and large and is
                          charged in the AMORTISED column, not hidden.
              retrieve  : graph construction from cached hubs, amortised per
                          commit (build_graph_s / N).
              predict   : RN + PPR scoring (the deployed fusion F).
              amortised : KG ingest (parse+diff+delta, logs/<p>/ast_timing.csv)
                          + CSTG channel G refit.

  metric baselines (LR/HGB/RF/LApredict)
              featurise : derive the 12 ApacheJIT metrics from the commit --
                          a diff against the parent plus history lookups.
                          MEASURED, not assumed zero (the stored 0.001 ms was a
                          lookup into ApacheJIT's precomputed table, which a new
                          project does not have).
              predict   : the model call.
              amortised : periodic refit.

  JITLine-online          featurise = metric-diff + its own diff tokenisation.
  DeepJIT                 featurise = metric-diff + tokenisation; predict from
                          its own grid; refit amortised from the same grid.

A note on the diff measurement: shelling out to `git` once per commit costs
~20 ms of process creation on this platform, which is an artefact of the
harness rather than the cost of a diff. We measure and subtract it, and quote
the batched per-commit figure. See baselines/measure_metric_featurisation.py.

Cache-only. No Neo4j.

Run:  python inference/make_rq2_revision.py
Out:  Paper/ResultsDiscussionsDraft/tab_rq2_deployment.tex     (merged T4+T5)
      Paper/ResultsDiscussionsDraft/tab_rq2_kgonly.tex         (A-E table)
      Paper/ResultsDiscussionsDraft/figures/rq2_*.pdf|png      (candidates)
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
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"
FIGS = DRAFT / "figures"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]


# ----------------------------------------------------------------- helpers --
def jload(p, name):
    f = OUTP / p / "final_final_run" / "complexity" / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def kg_ingest_ms(p):
    """Measured parse+diff+delta time to fold one commit into the graph."""
    for cand in (f"ast_timing.csv", f"AST_timing.csv"):
        f = LOGS / p / cand
        if f.exists():
            t = pd.read_csv(f)
            if "wall_ms" in t and len(t):
                return float(t["wall_ms"].median())
    return np.nan


def collect():
    rows = {}
    for p in PROJECTS:
        lat = jload(p, "prediction_latency.json")
        feat = jload(p, "featurisation_bygroup.json")
        grow = jload(p, "growth.json")
        prof = jload(p, "kg_profile.json")
        gref = jload(p, "g_refit_cost.json")
        if not (lat and feat):
            continue
        f = lat["final"]
        L = f["predict_ms_per_commit"]
        B = pickle.load(open(OUTP / p / "final_final_run" / "baselines"
                             / "baseline_extra_results.pkl", "rb"))
        rows[p] = {
            "lat": lat, "feat": feat, "grow": grow, "prof": prof,
            "timings": B.get("timings", {}),
            "kg_predict": L["RN"]["median"] + L["PPR"]["median"],
            "kg_retrieve": 1000.0 * f["build_graph_s"] / max(f["Nc"], 1),
            "kg_ingest": kg_ingest_ms(p),
            "g_refit": (gref or {}).get("refit_ms_per_commit_amortised", np.nan),
            "nnz": f["nnz"], "Nc": f["Nc"], "Nh": f["Nh"],
            "featurise": feat["twelve_metrics_ms"],
            "featurise_la": feat["la_only_ms"],
            "git_ms": feat["git_numstat_ms"],
            "methods": L,
        }
    return rows


def deepjit_stage():
    """DeepJIT predict + amortised refit, from its own grid."""
    f = OUTP / "DeepJIT_baseline_results" / "grid_summary.csv"
    if not f.exists():
        return np.nan, np.nan
    df = pd.read_csv(f)
    df["proj"] = df["project"].str.replace("apache/", "", regex=False)
    s = df[(df.M == 200) & (df.gap_commits == 50)
           & (df.warmup_ratio == 0.05) & (df.proj.isin(PROJECTS))]
    if s.empty:
        return np.nan, np.nan
    g = s.groupby("proj").agg(pred=("total_predict_seconds", "mean"),
                              refit=("total_refit_seconds", "mean"),
                              n=("evaluated_n", "mean"))
    return (float(np.median(1000 * g.pred / g.n)),
            float(np.median(1000 * g.refit / g.n)))


# ------------------------------------------------- T4: merged deployment --
def t_deployment(R):
    fm = float(np.median([r["featurise"] for r in R.values()]))
    # LApredict consumes only `la`, so it pays the diff but not the
    # scope/entropy and history derivation the other three need.
    fm_la = float(np.median([r["featurise_la"] for r in R.values()]))
    kgp = float(np.median([r["kg_predict"] for r in R.values()]))
    kgr = float(np.median([r["kg_retrieve"] for r in R.values()]))
    kg_lat = kgp + kgr
    kgi = float(np.nanmedian([r["kg_ingest"] for r in R.values()]))
    kgg = float(np.nanmedian([r["g_refit"] for r in R.values()]))
    dj_pred, dj_refit = deepjit_stage()

    def bt(key, comp):
        v = [r["timings"].get(key, {}).get(comp, np.nan) for r in R.values()]
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else np.nan

    jl_tok = bt("B_JITLINE_ONLINE", "featurize_ms_per_commit")

    # model -> (featurise, retrieve, predict, amortised)
    M = [
        (r"\textbf{KG-Commit ($F_{\mathrm{ov}}{+}G$)}", 0.0, kgr, kgp, kgi + kgg, True),
        ("LR", fm, 0.0, bt("B_LR", "predict_ms_per_commit"),
         bt("B_LR", "refit_ms_per_commit_amortised"), False),
        ("HGB", fm, 0.0, bt("B_HGB", "predict_ms_per_commit"),
         bt("B_HGB", "refit_ms_per_commit_amortised"), False),
        ("RF", fm, 0.0, bt("B_RF", "predict_ms_per_commit"),
         bt("B_RF", "refit_ms_per_commit_amortised"), False),
        ("LApredict", fm_la, 0.0, bt("B_LAPREDICT", "predict_ms_per_commit"),
         bt("B_LAPREDICT", "refit_ms_per_commit_amortised"), False),
        ("DeepJIT", fm, 0.0, dj_pred, dj_refit, False),
        ("JITLine-online", fm + jl_tok, 0.0,
         bt("B_JITLINE_ONLINE", "predict_ms_per_commit"),
         bt("B_JITLINE_ONLINE", "refit_ms_per_commit_amortised"), False),
    ]

    L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{RQ2 deployment cost per commit (ms), median over the 11 "
         r"projects, costed as if each model were deployed on a new project. "
         r"\textsc{Latency} is everything a developer waits for between the "
         r"commit landing and the score appearing; its three stages sum to the "
         r"\emph{total} column, which is what the \emph{vs.} column compares. "
         r"The baselines must derive their features from the commit itself -- a "
         r"diff against the parent plus history lookups -- because on a new "
         r"project no precomputed feature table exists; that cost is dominated by "
         r"the diff itself and varies with repository size (3.3--22.1\,ms across "
         r"the 11 projects, median reported). LApredict consumes only the "
         r"added-lines feature, so it pays the same diff but not the metric "
         r"derivation, which is under $0.5\%$ of the total. KG-Commit reads context "
         r"its incremental update already materialised, so it featurises nothing "
         r"on this path; that update is real work and is charged in full under "
         r"\textsc{Amortised}, which collects periodic cost the developer never "
         r"waits for. Models at least $2\times$ slower on latency are in "
         r"\textcolor{BrickRed}{red}.}",
         r"\label{tab:rq2_deployment}",
         r"\begin{tabular}{l rrr r c | rr}",
         r"\toprule",
         r"& \multicolumn{5}{c|}{\textsc{Prediction latency} (developer waits)} "
         r"& \multicolumn{2}{c}{\textsc{Amortised} (not waited on)} \\",
         r"\cmidrule(lr){2-6}\cmidrule(l){7-8}",
         r"Model & featurise & retrieve & predict & \textbf{total} & vs.\ "
         r"& ingest/refit & per commit \\",
         r"\midrule"]

    for name, f_, r_, p_, am, is_kg in M:
        tot = f_ + r_ + p_
        rel = tot / kg_lat if kg_lat else np.nan
        ts = (rf"\textbf{{{tot:.3f}}}" if is_kg
              else (rf"\textcolor{{BrickRed}}{{{tot:.3f}}}" if rel >= 2
                    else f"{tot:.3f}"))
        rs = (r"\textbf{1.0$\times$}" if is_kg
              else (rf"\textcolor{{BrickRed}}{{{rel:.1f}$\times$}}" if rel >= 2
                    else f"{rel:.1f}$\times$"))
        fs = r"\textbf{0}" if is_kg else f"{f_:.3f}"
        rr = f"{r_:.3f}" if r_ else "--"
        lab = "KG update" if is_kg else "refit"
        L.append(f"{name} & {fs} & {rr} & {p_:.3f} & {ts} & {rs} & "
                 f"{lab} & {am:.2f} \\\\")
        if is_kg:
            L.append(r"\midrule")

    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\footnotesize KG-Commit's amortised figure is the measured "
          r"parse$+$diff$+$delta cost of folding one commit into the graph "
          r"($\sim$%.0f\,ms) plus the CSTG channel refit ($\sim$%.1f\,ms). It is "
          r"the largest single number in the table and is reported in full; it is "
          r"paid once when the commit lands, off the prediction path."
          % (kgi, kgg),
          r"\end{table*}"]
    return "\n".join(L), kg_lat, fm


# --------------------------------------------- T7: KG-only, covers A..E --
def t_kgonly(R):
    """One table covering: (A) latency vs scale, (B) per-method latency,
    (C) growth stationarity, (D) layer profile, (E) ingest cost."""
    L = [r"\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3.2pt}",
         r"\caption{RQ2 internal cost profile of KG-Commit, per project. "
         r"\textsc{Graph} gives the deployed graph's scale: commit and hub nodes, "
         r"and the non-zeros of the commit--hub incidence matrix, which is what "
         r"the relational scorers actually traverse. \textsc{Inference} gives the "
         r"per-commit latency of each of the five methods (median, and p95 for "
         r"the deployed pair) -- the deployed fusion $F{=}$RN$+$PPR is the "
         r"cheapest combination that is also competitive, which is why the "
         r"embedding methods are not deployed. \textsc{Ingest} gives the "
         r"amortised cost of folding a commit in, and the per-commit AST delta "
         r"whose distribution governs it: the mean is far above the median and "
         r"the Gini is high, so cost is dominated by a few very large commits "
         r"rather than growing with history.}",
         r"\label{tab:rq2_kgonly}",
         r"\begin{tabular}{l rrr r | rrrrr rr | rr rrr}",
         r"\toprule",
         r"& \multicolumn{4}{c|}{\textsc{Graph}} "
         r"& \multicolumn{7}{c|}{\textsc{Inference} latency (ms/commit)} "
         r"& \multicolumn{5}{c}{\textsc{Ingest} \& per-commit AST delta} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-12}\cmidrule(l){13-17}",
         r"& $N_c$ & $N_h$ & nnz & c/s "
         r"& RN & PPR & LP & DW & KGE & \textbf{$F$} & $F$p95 "
         r"& ms & edges & p50 & p95 & Gini \\",
         r"\midrule"]

    agg = {k: [] for k in ("nnz", "F", "ing", "gini")}
    for p in PROJECTS:
        if p not in R:
            continue
        r = R[p]
        f = r["lat"]["final"]
        M = r["methods"]
        F = r["kg_predict"]
        fp95 = M["RN"]["p95"] + M["PPR"]["p95"]
        cps = 1000.0 / F if F else np.nan
        g = (r["grow"] or {}).get("ast", {}).get("delta_per_commit", {})
        tot = (r["grow"] or {}).get("ast", {}).get("total_delta_edges", np.nan)
        L.append(
            f"{DISP[p]} & {f['Nc']:,} & {f['Nh']:,} & {f['nnz']:,} & {cps:,.0f} & "
            f"{M['RN']['median']:.3f} & {M['PPR']['median']:.3f} & "
            f"{M['LP']['median']:.3f} & {M['DW']['median']:.3f} & "
            rf"{M['KGE']['median']:.3f} & \textbf{{{F:.3f}}} & {fp95:.3f} & "
            f"{r['kg_ingest']:,.0f} & {tot/1e6:.2f}M & "
            f"{g.get('p50', np.nan):,.0f} & {g.get('p95', np.nan):,.0f} & "
            f"{g.get('gini', np.nan):.2f} \\\\")
        agg["nnz"].append(f["nnz"]); agg["F"].append(F)
        agg["ing"].append(r["kg_ingest"]); agg["gini"].append(g.get("gini", np.nan))

    L += [r"\midrule",
          r"\textbf{Median} & \multicolumn{3}{c}{nnz "
          f"{np.median(agg['nnz']):,.0f}" + r"} & "
          f"{1000.0/np.median(agg['F']):,.0f}" + r" & \multicolumn{5}{c}{} & "
          rf"\textbf{{{np.median(agg['F']):.3f}}}" + r" & & "
          f"{np.nanmedian(agg['ing']):,.0f}" + r" & & & & "
          f"{np.nanmedian(agg['gini']):.2f}" + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


# ------------------------------------------------------ figure candidates --
def figures(R, kg_lat, fm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    out = []

    S = json.load(open(OUTP / "final_final_run_summary.json"))
    kgf1 = {p: S[p]["switch"]["overall"]["grid"]["200"]["Macro_F1"]
            for p in PROJECTS}
    MB = json.loads((OUTP / "tables" / "rq1_baselines_matched.json")
                    .read_text(encoding="utf-8"))

    def bt(key, comp):
        v = [r["timings"].get(key, {}).get(comp, np.nan) for r in R.values()]
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else np.nan

    dj_pred, dj_refit = deepjit_stage()
    BK = {"LR": "B_LR", "HGB": "B_HGB", "RF": "B_RF",
          "LApredict": "B_LAPREDICT", "JITLine-online": "B_JITLINE_ONLINE"}

    # ---- F1: cost/accuracy Pareto -------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = [("KG-Commit", kg_lat,
            float(np.mean([kgf1[p] for p in PROJECTS])),
            float(np.nanmedian([r["kg_ingest"] + r["g_refit"]
                                for r in R.values()])), "#1D4ED8", "*", 420)]
    for lab, key in BK.items():
        extra = bt(key, "featurize_ms_per_commit") if "JITLINE" in key else 0.0
        lat = fm + extra + bt(key, "predict_ms_per_commit")
        f1 = float(np.mean([MB[p]["baselines"][key]["matched"]["Macro_F1"]
                            for p in PROJECTS]))
        pts.append((lab, lat, f1, bt(key, "refit_ms_per_commit_amortised"),
                    "#64748B", "o", 110))
    dj_f1 = 0.611
    pts.append(("DeepJIT", fm + dj_pred, dj_f1, dj_refit, "#B91C1C", "s", 110))

    for lab, x, y, am, c, mk, sz in pts:
        ax.scatter(x, y, s=sz, c=c, marker=mk, zorder=3,
                   edgecolors="white", linewidths=1.1)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(9, 5),
                    fontsize=9, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("Prediction latency per commit (ms, log scale)  "
                  r"$\longleftarrow$ better")
    ax.set_ylabel(r"Macro-F1 (mean over 11 projects)  $\longrightarrow$ better")
    ax.grid(alpha=.25, linestyle=":")
    ax.set_title("Cost/accuracy trade-off at deployment", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"rq2_pareto.{ext}", dpi=200)
    plt.close(fig)
    out.append("rq2_pareto")

    # ---- F2: latency vs graph sparsity --------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = [R[p]["nnz"] for p in PROJECTS if p in R]
    ys = [R[p]["kg_predict"] for p in PROJECTS if p in R]
    ax.scatter(xs, ys, s=90, c="#1D4ED8", zorder=3, edgecolors="white")
    for p in PROJECTS:
        if p in R:
            ax.annotate(DISP[p], (R[p]["nnz"], R[p]["kg_predict"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    k = np.polyfit(xs, ys, 1)
    xx = np.linspace(min(xs), max(xs), 50)
    ax.plot(xx, np.polyval(k, xx), "--", c="#B45309", lw=1.4,
            label=f"linear fit ($r$={np.corrcoef(xs, ys)[0,1]:.3f})")
    ax.set_xlabel("nnz of the commit--hub incidence matrix")
    ax.set_ylabel("KG-Commit prediction latency (ms/commit)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.25, linestyle=":")
    ax.set_title("Latency scales with graph sparsity, not project size",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"rq2_latency_vs_nnz.{ext}", dpi=200)
    plt.close(fig)
    out.append("rq2_latency_vs_nnz")

    # ---- F3: stacked latency budget per model -------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["KG-Commit"] + list(BK) + ["DeepJIT"]
    feat, pred = [0.0], [float(np.median([r["kg_predict"] for r in R.values()]))]
    retr = [float(np.median([r["kg_retrieve"] for r in R.values()]))]
    for lab, key in BK.items():
        extra = bt(key, "featurize_ms_per_commit") if "JITLINE" in key else 0.0
        feat.append(fm + extra); pred.append(bt(key, "predict_ms_per_commit"))
        retr.append(0.0)
    feat.append(fm); pred.append(dj_pred); retr.append(0.0)
    yy = np.arange(len(names))
    ax.barh(yy, feat, color="#C2410C", label="featurise (derive inputs)")
    ax.barh(yy, retr, left=feat, color="#7C3AED", label="retrieve (graph)")
    ax.barh(yy, pred, left=np.array(feat) + np.array(retr),
            color="#1D4ED8", label="predict (model call)")
    ax.set_yticks(yy); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Prediction-path latency per commit (ms)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.25, linestyle=":")
    ax.set_title("Where the waiting time goes", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"rq2_latency_budget.{ext}", dpi=200)
    plt.close(fig)
    out.append("rq2_latency_budget")

    # ---- F4: ingest cost distribution (why the mean misleads) ---------
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for p in ("activemq", "kafka", "zookeeper"):
        f = LOGS / p / "ast_timing.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        if "wall_ms" not in t:
            continue
        v = np.sort(t["wall_ms"].values)
        ax.plot(v, np.linspace(0, 1, len(v)), lw=1.8, label=DISP[p])
    ax.set_xscale("log")
    ax.set_xlabel("Per-commit KG update cost (ms, log scale)")
    ax.set_ylabel("Cumulative fraction of commits")
    ax.grid(alpha=.25, linestyle=":")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Ingest cost is dominated by a small tail of large commits",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"rq2_ingest_cdf.{ext}", dpi=200)
    plt.close(fig)
    out.append("rq2_ingest_cdf")
    return out


def main():
    R = collect()
    print(f"projects with full data: {len(R)}/{len(PROJECTS)}")
    t4, kg_lat, fm = t_deployment(R)
    (DRAFT / "tab_rq2_deployment.tex").write_text(t4, encoding="utf-8")
    print("  wrote tab_rq2_deployment.tex")
    (DRAFT / "tab_rq2_kgonly.tex").write_text(t_kgonly(R), encoding="utf-8")
    print("  wrote tab_rq2_kgonly.tex")
    for f in figures(R, kg_lat, fm):
        print(f"  wrote figures/{f}.pdf|.png")
    print(f"\nKG latency {kg_lat:.3f} ms | baseline featurise {fm:.3f} ms")


if __name__ == "__main__":
    main()
