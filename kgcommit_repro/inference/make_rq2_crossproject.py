"""
RQ2 cross-project (pooled) artifacts, cache-only:
  - complexity_fit: pooled per-commit AST build cost vs current graph size, ALL
    projects, with a fitted slope + R^2 (shows O(delta), flat in N).
  - build_cost_ast_perproject: per-commit AST build cost vs commit index, one line
    per project (the AST-only build-cost figure requested for RQ2).
  - space_vs_churn: total graph nodes vs total commits (proxy churn), pooled.
  - tab_crossproject_scalability: per-project build-median, predict-latency,
    throughput, graph size side-by-side + Macro-Avg.

Sources: logs/<p>/ast_timing.csv (per-commit build cost),
         outputs/<p>/scalability/{prediction_latency,kg_profile,build_complexity}.json.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
LOGS = ROOT / "kgcommit_repro" / "logs"
PM = ROOT / "Paper" / "paper_material" / "RQ2_scalability"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
CMAP = plt.get_cmap("tab10")


def read_ast_timing(folder):
    f = LOGS / folder / "ast_timing.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f, encoding="utf-8", errors="ignore")))
    if not rows:
        return None
    # find the per-commit time + any size column
    cols = rows[0].keys()
    tcol = next((c for c in cols if "ms" in c.lower() or "sec" in c.lower() or "time" in c.lower()), None)
    if not tcol:
        return None
    t = []
    for r in rows:
        try:
            t.append(float(r[tcol]))
        except (ValueError, TypeError):
            t.append(np.nan)
    return np.asarray(t, float)


def complexity_fit():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    allx, ally = [], []
    for i, (disp, folder) in enumerate(PROJECTS):
        t = read_ast_timing(folder)
        if t is None or len(t) < 20:
            continue
        t = t[np.isfinite(t)]
        n = np.arange(len(t))                       # graph size proxy = commit index
        # downsample for the scatter
        step = max(1, len(t)//300)
        ax.scatter(n[::step], t[::step], s=6, color=CMAP(i % 10), alpha=0.35, label=disp)
        allx.append(n); ally.append(t)
    if allx:
        X = np.concatenate(allx).astype(float); Y = np.concatenate(ally)
        good = np.isfinite(Y)
        slope, intercept = np.polyfit(X[good], Y[good], 1)
        yhat = slope * X[good] + intercept
        ss_res = np.sum((Y[good]-yhat)**2); ss_tot = np.sum((Y[good]-Y[good].mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        xs = np.linspace(X.min(), X.max(), 50)
        ax.plot(xs, slope*xs+intercept, "k--", lw=1.6,
                label=f"fit: slope={slope:.2e}, $R^2$={r2:.3f}")
    ax.set_xlabel("Commit index")
    ax.set_ylabel("AST build cost (ms)")
    ax.set_title("Build cost vs. history", weight="bold")
    ax.legend(fontsize=7, ncol=2); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"complexity_fit.{e}", bbox_inches="tight")
    plt.close(fig)


def build_cost_perproject():
    fig, ax = plt.subplots(figsize=(7, 4.4)); any_ = False
    for i, (disp, folder) in enumerate(PROJECTS):
        t = read_ast_timing(folder)
        if t is None:
            continue
        # rolling mean to declutter
        w = max(1, len(t)//200); roll = np.convolve(np.nan_to_num(t), np.ones(w)/w, "valid")
        ax.plot(np.arange(len(roll)), roll, color=CMAP(i % 10), lw=1.2, label=disp); any_ = True
    if not any_:
        plt.close(fig); return
    ax.set_xlabel("Commit index"); ax.set_ylabel("AST build cost (ms)")
    ax.set_title("AST build cost", weight="bold")
    ax.legend(fontsize=7, ncol=3); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"build_cost_ast_perproject.{e}", bbox_inches="tight")
    plt.close(fig)


def space_vs_churn():
    xs, ys, labs = [], [], []
    for disp, folder in PROJECTS:
        kp = OUTP / folder / "scalability" / "kg_profile.json"
        if not kp.exists():
            continue
        P = json.load(open(kp))
        tot = 0
        for g in ["ast", "cfg", "dfg", "pdg", "seq", "cstg", "core"]:
            gg = P.get(g, {})
            tot += gg.get("nodes", 0) if isinstance(gg, dict) else 0
        nc = P.get("core", {}).get("nodes") if isinstance(P.get("core"), dict) else None
        if tot:
            xs.append(nc or 0); ys.append(tot); labs.append(disp)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.scatter(xs, ys, s=40, color="#0072B2")
    for x, y, l in zip(xs, ys, labs):
        ax.annotate(l, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Commits"); ax.set_ylabel("Total nodes")
    ax.set_title("Space vs. churn", weight="bold")
    ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"space_vs_churn.{e}", bbox_inches="tight")
    plt.close(fig)


def crossproject_table():
    """RQ2 Table 7: KG-Commit's deployed cost and graph size per project, alongside
    the MEASURED per-commit deployment cost of the baselines (featurise + predict +
    amortised refit, from the instrumented baseline runs). This is the table that
    carries the efficiency story, so the baselines belong in it rather than in a
    separate one: at matched accuracy (Table~\\ref{tab:rq1_perf_g50}) the only
    baseline that ties KG-Commit costs an order of magnitude more per commit.

    KG's own cost is computed from the project's ACTUAL chosen F (which varies:
    PPR / RN+PPR / RN+PPR+LP / RN+LP / RN+KGE), not the hard-coded RN+PPR proxy
    stored as `deployed_F_predict_ms_per_commit`.
    """
    from make_rq2_deployment_cost import kg_cost, baseline_timings

    BL = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApred."),
          ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]

    L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{RQ2 cross-project scalability and measured deployment cost. "
         r"\emph{Left}: KG-Commit's deployed configuration per project --- the selected "
         r"fusion $F$, the resident graph size, and the resulting throughput. "
         r"\emph{Right}: per-commit deployment cost (ms) of every model, measured on the "
         r"same commits under the same online protocol, as featurisation $+$ inference "
         r"$+$ amortised refit. KG-Commit reads context that its incremental update "
         r"already materialised, so it featurises nothing at inference. Its cost is "
         r"governed by the selected $F$ rather than by graph size: ActiveMQ scores a "
         r"$5.6$M-node graph in $0.17$~ms using PPR alone, whereas Zookeeper is the "
         r"dearest project on the second-smallest graph, being the only one whose $F$ "
         r"includes the embedding-based KGE and its periodic refit. "
         r"The change-metric baselines are cheaper still "
         r"(they compute almost nothing from twelve scalars) but trail on accuracy; "
         r"JITLine, the one baseline that matches KG-Commit on Macro-F1 and G-Mean "
         r"(Table~\ref{tab:rq1_perf_g50}), costs $\sim$56$\times$ more per commit, "
         r"dominated by re-tokenising every diff.}",
         r"\label{tab:crossproject_scalability}",
         r"\begin{tabular}{ll rr | r" + "r" * len(BL) + "}", r"\toprule",
         r"& & \multicolumn{2}{c|}{KG-Commit graph} & "
         r"\multicolumn{" + str(1 + len(BL)) + r"}{c}{Per-commit deployment cost (ms)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-" + str(5 + len(BL)) + "}",
         r"Project & $F$ & Nodes & c/s & \textbf{KG-Commit} & " +
         " & ".join(d for _, d in BL) + r" \\",
         r"\midrule"]

    lat_all, thr_all, nodes_all = [], [], []
    bsum = {k: [] for k, _ in BL}
    for disp, folder in PROJECTS:
        pl = OUTP / folder / "scalability" / "prediction_latency.json"
        kp = OUTP / folder / "scalability" / "kg_profile.json"
        tot = 0
        if kp.exists():
            K = json.load(open(kp))
            for g in ["ast", "cfg", "dfg", "pdg", "seq", "cstg", "core"]:
                gg = K.get(g, {}); tot += gg.get("nodes", 0) if isinstance(gg, dict) else 0

        c = kg_cost(folder) if pl.exists() else None
        if c is None:
            L.append(f"{disp} & -- & {tot:,} & -- & -- & " +
                     " & ".join("--" for _ in BL) + r" \\")
            continue
        predict, refit, Flabel = c
        kg_tot = predict + refit
        thr = 1000.0 / kg_tot if kg_tot > 0 else 0.0
        lat_all.append(kg_tot); thr_all.append(thr)
        if tot:
            nodes_all.append(tot)

        t = baseline_timings(folder)
        cells = []
        for key, _ in BL:
            v = t.get(key, {}).get("total_ms_per_commit")
            if v is not None:
                bsum[key].append(v)
            cells.append(f"{v:.3f}" if v is not None else "--")
        L.append(rf"{disp} & \texttt{{\scriptsize {Flabel}}} & {tot:,} & "
                 rf"{thr:,.0f} & \textbf{{{kg_tot:.3f}}} & " +
                 " & ".join(cells) + r" \\")

    if lat_all:
        mean_b = []
        for key, _ in BL:
            v = bsum[key]
            mean_b.append(f"{np.mean(v):.3f}" if v else "--")
        mean_nodes = (f"{np.mean(nodes_all):,.0f}" if nodes_all else "--")
        L.append(r"\midrule \textbf{Mean} & & \textbf{" + mean_nodes + r"} & \textbf{"
                 + f"{np.mean(thr_all):,.0f}" + r"} & \textbf{"
                 + f"{np.mean(lat_all):.3f}" + r"} & " + " & ".join(mean_b) + r" \\")
        rel = []
        km = np.mean(lat_all)
        for key, _ in BL:
            v = bsum[key]
            rel.append(rf"{np.mean(v)/km:.1f}$\times$" if v else "--")
        L.append(r"\textit{vs.\ KG-Commit} & & & & \textit{1.0$\times$} & "
                 + " & ".join(rel) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (PM / "tab_crossproject_scalability.tex").write_text("\n".join(L), encoding="utf-8")


def main():
    PM.mkdir(parents=True, exist_ok=True)
    complexity_fit(); build_cost_perproject(); space_vs_churn(); crossproject_table()
    print("wrote complexity_fit, build_cost_ast_perproject, space_vs_churn, "
          "tab_crossproject_scalability")


if __name__ == "__main__":
    main()
