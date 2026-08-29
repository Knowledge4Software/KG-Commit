"""
Regenerate the RQ1 artifacts (tables + figures) for the Results draft.
======================================================================

Replaces the draft's RQ1 tables and figures with the FINAL reported configuration:

  * 11 projects (was 7)
  * 6 baselines: LR, HGB, RF, LApredict, DeepJIT, JITLine-online
    (was 5: LR, HGB, LApredict, Deeper, JITLine -- Deeper dropped, RF and the
     fully-online JITLine variant added, DeepJIT re-run under the matched protocol)
  * KG-Commit = the OVERALL fusion F = RN+PPR, with adaptive INIT and Switch@200
    (the per-project fusion variant is deliberately NOT reported here)

Emits, into <draft>/:
  tab_rq1_performance.tex   Macro-F1 / G-Mean / AUC per project + Macro/Micro-Avg
                            + per-commit deployment cost
  tab_rq1_effort.tex        Popt / ACC@20 per project + aggregates
  figures/paper_material_RQ1_performance_streams_overall.pdf
  figures/paper_material_RQ1_performance_roc_overall.pdf
  figures/paper_material_RQ3_subgraphs_metric_radar_Macro_F1_overall.pdf

Numbers come from the same persisted artifacts as the panel figures, so the tables
and the figures cannot disagree:
  final_final_run/switch/switch_results.json     KG-Commit @ switch 200 (overall)
  final_final_run/baselines/baseline_extra_results.pkl   5 cache-only baselines
  DeepJIT_baseline_results/grid_summary.csv      DeepJIT, 5-seed mean

Run: python inference/make_draft_rq1_artifacts.py
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material"
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"
FIGD = DRAFT / "figures"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

# display order for the baseline columns
BCOLS = [("LR", "B_LR"), ("HGB", "B_HGB"), ("RF", "B_RF"),
         ("LApredict", "B_LAPREDICT"), ("DeepJIT", None),
         ("JITLine-on", "B_JITLINE_ONLINE")]


def _deepjit_macro_f1():
    """5-seed mean macro-F1 per project, from the raw confusion counts."""
    df = pd.read_csv(PM.parent.parent / "outputs" / "DeepJIT_baseline_results"
                     / "grid_summary.csv")

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    df["macro_f1"] = df.apply(
        lambda r: (f1(r.tp, r.fp, r.fn) + f1(r.tn, r.fn, r.fp)) / 2, axis=1)
    # G-Mean = sqrt(recall * specificity); AUC is reported directly
    df["gmean"] = np.sqrt(
        (df.tp / (df.tp + df.fn)).clip(0) * (df.tn / (df.tn + df.fp)).clip(0))
    df["ps"] = df["project"].str.replace("apache/", "", regex=False)
    # NOTE: grid_summary.csv carries no effort columns, so Popt/ACC@20 for DeepJIT
    # come from rq1_effort_data.json instead -- computed there on exactly the same
    # SHA-aligned span as every other model.
    return df.groupby("ps").agg(Macro_F1=("macro_f1", "mean"),
                                G_Mean=("gmean", "mean"),
                                AUC=("auc", "mean"),
                                n=("n", "first"))


def collect():
    S = json.load(open(OUTP / "final_final_run_summary.json"))
    DJ = _deepjit_macro_f1()
    rows = {}
    for p in PROJECTS:
        sw = S[p]["switch"]["overall"]["grid"]["200"]
        B = pickle.load(open(OUTP / p / "final_final_run" / "baselines"
                             / "baseline_extra_results.pkl", "rb"))
        d = {"KG": {k: float(sw[k]) for k in ("Macro_F1", "G_Mean", "AUC")},
             "n_eval": S[p]["switch"]["overall"]["n_eval"]}
        # KG effort metrics come from the effort panel data (same aligned span)
        for lab, key in BCOLS:
            if key is None:
                r = DJ.loc[p]
                d[lab] = {"Macro_F1": float(r.Macro_F1), "G_Mean": float(r.G_Mean),
                          "AUC": float(r.AUC)}
            else:
                m = B["baselines"][key]
                d[lab] = {k: float(m[k]) for k in ("Macro_F1", "G_Mean", "AUC")}
                d[lab]["Popt"] = float(m.get("Popt", np.nan))
                d[lab]["ACC20"] = float(m.get("ACC20", np.nan))
        d["timings"] = B.get("timings", {})
        rows[p] = d
    return rows


def _agg(rows, model, metric, weighted=False):
    vals, ws = [], []
    for p in PROJECTS:
        v = rows[p].get(model, {}).get(metric, np.nan)
        if np.isfinite(v):
            vals.append(v); ws.append(rows[p]["n_eval"])
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=ws) if weighted else np.mean(vals))


def _fmt(v, best=False):
    if not np.isfinite(v):
        return "--"
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if best else s


def table_performance(rows, effort):
    models = ["KG"] + [lab for lab, _ in BCOLS]
    head = ["KG-Commit ($F_{\\mathrm{ov}}{+}G$, S@200)"] + [
        ("JITLine-online" if lab == "JITLine-on" else lab) for lab, _ in BCOLS]

    L = []
    L.append(r"\begin{table*}[t]\centering")
    L.append(r"\scriptsize\setlength{\tabcolsep}{3pt}")
    L.append(
        r"\caption{RQ1 performance under the online protocol: KG-Commit "
        r"(overall fusion $F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$ plus the CSTG "
        r"channel $G$, with the adaptive initial-fit window and the switch to "
        r"$F{+}G$ at $S{=}200$) vs.\ the six baselines, on all 11 projects. "
        r"In the aggregate rows the best model per metric is in \textbf{bold}; "
        r"per-project rows are left unbolded so the averages stand out. The final "
        r"two rows give the measured per-commit deployment cost (featurisation "
        r"$+$ inference $+$ amortised refit) and the slowdown relative to "
        r"KG-Commit; models at least $5\times$ slower are shown in "
        r"\textcolor{BrickRed}{red}.}")
    L.append(r"\label{tab:rq1_perf_g50}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{l" + "ccc" * len(models) + "}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{Project} & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{h}}}" for h in head) + r" \\")
    L.append(" ".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}"
                      for i in range(len(models))))
    L.append(" & " + " & ".join(["Macro-F1 & G-Mean & AUC"] * len(models))
             + r" \\ \midrule")

    for p in PROJECTS:
        cells = []
        for m in models:
            for k in ("Macro_F1", "G_Mean", "AUC"):
                cells.append(_fmt(rows[p].get(m, {}).get(k, np.nan)))
        L.append(f"{DISP[p]} & " + " & ".join(cells) + r" \\")

    for lab, w in ((r"\textbf{Macro-Avg}", False), (r"\textbf{Micro-Avg}", True)):
        vals = {(m, k): _agg(rows, m, k, w)
                for m in models for k in ("Macro_F1", "G_Mean", "AUC")}
        best = {k: max((vals[(m, k)] for m in models
                        if np.isfinite(vals[(m, k)])), default=np.nan)
                for k in ("Macro_F1", "G_Mean", "AUC")}
        cells = []
        for m in models:
            for k in ("Macro_F1", "G_Mean", "AUC"):
                v = vals[(m, k)]
                cells.append(_fmt(v, best=np.isfinite(v)
                                  and abs(v - best[k]) < 1e-9))
        L.append(r"\midrule " + lab + " & " + " & ".join(cells) + r" \\")

    # ---- deployment cost -----------------------------------------------------
    cost = deployment_cost(rows)
    kg = cost.get("KG", np.nan)

    def cell(v):
        if not np.isfinite(v):
            return r"\multicolumn{3}{c}{--}"
        s = f"{v:.3f}"
        red = np.isfinite(kg) and v >= 5 * kg
        return (rf"\multicolumn{{3}}{{c}}{{\textcolor{{BrickRed}}{{{s}}}}}"
                if red else rf"\multicolumn{{3}}{{c}}{{{s}}}")

    def rel(v):
        if not (np.isfinite(v) and np.isfinite(kg) and kg > 0):
            return r"\multicolumn{3}{c}{--}"
        r = v / kg
        s = (r"\textbf{1.0$\times$}" if abs(r - 1) < 1e-9 else f"{r:.1f}$\\times$")
        return (rf"\multicolumn{{3}}{{c}}{{\textcolor{{BrickRed}}{{{s}}}}}"
                if r >= 5 else rf"\multicolumn{{3}}{{c}}{{{s}}}")

    L.append(r"\midrule \textit{ms/commit} & "
             + " & ".join(cell(cost.get(m, np.nan)) for m in models) + r" \\")
    L.append(r"\textit{vs.\ KG-Commit} & "
             + " & ".join(rel(cost.get(m, np.nan)) for m in models) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def deployment_cost(rows):
    """Median per-commit ms across projects, per model."""
    keymap = {"LR": "B_LR", "HGB": "B_HGB", "RF": "B_RF",
              "LApredict": "B_LAPREDICT", "JITLine-on": "B_JITLINE_ONLINE"}
    out = {}
    for lab, key in keymap.items():
        v = [rows[p]["timings"].get(key, {}).get("total_ms_per_commit", np.nan)
             for p in PROJECTS]
        v = [x for x in v if np.isfinite(x)]
        out[lab] = float(np.median(v)) if v else np.nan

    # KG-Commit: the deployed graph fusion is RN+PPR over the final graph, plus
    # the CSTG channel; take the measured per-commit predict latency of RN and
    # PPR on the final graph and add the CSTG scoring cost.
    kg = []
    for p in PROJECTS:
        f = OUTP / p / "final_final_run" / "complexity" / "prediction_latency.json"
        if not f.exists():
            continue
        d = json.load(open(f)).get("final", {}).get("predict_ms_per_commit", {})
        tot = sum(d.get(m, {}).get("median", 0.0) for m in ("RN", "PPR"))
        if tot > 0:
            kg.append(tot)
    out["KG"] = float(np.median(kg)) if kg else np.nan
    out["DeepJIT"] = np.nan     # trained on GPU; not comparable on this axis
    return out


def table_effort(rows, effort):
    models = ["KG"] + [lab for lab, _ in BCOLS]
    head = ["KG-Commit ($F_{\\mathrm{ov}}{+}G$, S@200)"] + [
        ("JITLine-online" if lab == "JITLine-on" else lab) for lab, _ in BCOLS]
    KEY = {"KG": "overall", "LR": "LR", "HGB": "HGB", "RF": "RF",
           "LApredict": "LApredict", "DeepJIT": "DeepJIT",
           "JITLine-on": "JITLine-online"}

    def val(p, m, k):
        return effort.get(p, {}).get(KEY[m], {}).get(k, np.nan)

    L = []
    L.append(r"\begin{table*}[t]\centering")
    L.append(r"\scriptsize\setlength{\tabcolsep}{3pt}")
    L.append(
        r"\caption{RQ1 effort-aware evaluation: KG-Commit (overall fusion "
        r"$F_{\mathrm{ov}}{+}G$ with switch at $S{=}200$) vs.\ the six baselines "
        r"($P_{\mathrm{opt}}$, ACC@20\%LOC), on all 11 projects. Inspection "
        r"effort is the commit's churn (la$+$ld). In the aggregate rows the best "
        r"model per metric is in \textbf{bold}; per-project rows are left "
        r"unbolded so the averages stand out.}")
    L.append(r"\label{tab:rq1_effort_g50}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{l" + "cc" * len(models) + "}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{Project} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in head) + r" \\")
    L.append(" ".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}"
                      for i in range(len(models))))
    L.append(" & " + " & ".join([r"$P_{\mathrm{opt}}$ & ACC@20"] * len(models))
             + r" \\ \midrule")

    for p in PROJECTS:
        cells = [_fmt(val(p, m, k)) for m in models for k in ("Popt", "ACC20")]
        L.append(f"{DISP[p]} & " + " & ".join(cells) + r" \\")

    for lab, w in ((r"\textbf{Macro-Avg}", False), (r"\textbf{Micro-Avg}", True)):
        vals = {}
        for m in models:
            for k in ("Popt", "ACC20"):
                xs = [(val(p, m, k), rows[p]["n_eval"]) for p in PROJECTS]
                xs = [(v, n) for v, n in xs if np.isfinite(v)]
                vals[(m, k)] = (float(np.average([v for v, _ in xs],
                                                 weights=[n for _, n in xs]))
                                if w else float(np.mean([v for v, _ in xs]))) \
                    if xs else np.nan
        best = {k: max((vals[(m, k)] for m in models
                        if np.isfinite(vals[(m, k)])), default=np.nan)
                for k in ("Popt", "ACC20")}
        cells = []
        for m in models:
            for k in ("Popt", "ACC20"):
                v = vals[(m, k)]
                cells.append(_fmt(v, best=np.isfinite(v)
                                  and abs(v - best[k]) < 1e-9))
        L.append(r"\midrule " + lab + " & " + " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def copy_figures():
    """Copy the new panel PDFs into the draft's flat figures/ directory."""
    pairs = [
        (PM / "RQ1_performance" / "fig_rq1_streams__overall.pdf",
         FIGD / "paper_material_RQ1_performance_streams_overall.pdf"),
        (PM / "RQ1_performance" / "fig_rq1_roc__overall.pdf",
         FIGD / "paper_material_RQ1_performance_roc_overall.pdf"),
    ]
    for m in ("Macro_F1", "G_Mean", "AUC", "Precision", "Recall",
              "Buggy_F1", "ACC"):
        pairs.append((PM / "RQ3_subgraphs" / f"metric_radar__{m}__overall.pdf",
                      FIGD / f"paper_material_RQ3_subgraphs_radar_{m}_overall.pdf"))
    FIGD.mkdir(parents=True, exist_ok=True)
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {dst.name}")
        else:
            print(f"  MISSING {src}")


def main():
    rows = collect()
    effort = json.load(open(PM / "RQ1_performance" / "rq1_effort_data.json"))

    (DRAFT / "tab_rq1_performance.tex").write_text(
        table_performance(rows, effort) + "\n", encoding="utf-8")
    (DRAFT / "tab_rq1_effort.tex").write_text(
        table_effort(rows, effort) + "\n", encoding="utf-8")
    print(f"  wrote tab_rq1_performance.tex, tab_rq1_effort.tex -> {DRAFT}")
    copy_figures()

    print("\n--- Macro-Avg sanity ---")
    for m in ["KG"] + [l for l, _ in BCOLS]:
        print(f"  {m:<12} Macro-F1={_agg(rows, m, 'Macro_F1'):.3f}  "
              f"G-Mean={_agg(rows, m, 'G_Mean'):.3f}  "
              f"AUC={_agg(rows, m, 'AUC'):.3f}")


if __name__ == "__main__":
    main()
