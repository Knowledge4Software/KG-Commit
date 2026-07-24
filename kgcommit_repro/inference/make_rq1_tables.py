"""
RQ1 Table 5 (performance: Macro-F1 / G-Mean / AUC) and Table 6 (effort-aware:
Popt / ACC@20), KG-Commit(F+G) vs the FINAL FIVE baselines
(LR, HGB, LApredict, Deeper, JITLine), with Macro-Avg and Micro-Avg rows.

Two paired settings, selectable by --gap / --warmup (defaults = Setting A):
  Setting A: gap=0,  K=0.40  -> suffix _g0   (uses stored metrics as-is)
  Setting B: gap=50, K=0.20  -> suffix _g50  (re-scores F+G and baselines from
             cached raw per-commit scores with the gap; warm-up re-derived at K)

Cache-only (no Neo4j) for every project that has:
  final_fusion_results.pkl  (F+G raw p or metrics) AND baseline_extra_results.pkl.

Run:  python inference/make_rq1_tables.py [--gap 0|50] [--warmup 0.40|0.20] [--out DIR]
Out:  <out>/table5_performance_<suffix>.tex, table6_effort_<suffix>.tex (+ .csv)
"""
import argparse
import pickle
import csv
from pathlib import Path
import numpy as np

import _kgc_paths  # noqa: F401
import run_final_fusion as rff
import run_final_experiments as rfe
from online_jit import final_metrics
import effort_metrics as em
import common_window as cw

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"

from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
BASE5 = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
         ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]
PERF = ["Macro_F1", "G_Mean", "AUC"]
EFFORT = ["Popt", "ACC20"]
PP = {"Macro_F1": "Macro-F1", "G_Mean": "G-Mean", "AUC": "AUC",
      "Popt": "$P_{\\mathrm{opt}}$", "ACC20": "ACC@20"}


def _fg_scores(folder):
    """Raw per-commit F+G score p and its eval y, or (None,None)."""
    ff = OUTP / folder / "final_fusion_results.pkl"
    rf = OUTP / folder / "raw_fusion_scores.pkl"
    if rf.exists():
        d = pickle.load(open(rf, "rb"))
        # raw_fusion stores p over [ev0,N); need aligned y
        return d
    return None


def project_metrics(folder, gap, warmup):
    """Return {'FG':{m:..}, 'B_LR':{...},...} for perf+effort at (gap,warmup).
    gap=0 & warmup=0.40 -> use stored metrics (fast path). Otherwise re-score."""
    bx_p = OUTP / folder / "baseline_extra_results.pkl"
    ff_p = OUTP / folder / "final_fusion_results.pkl"
    if not (bx_p.exists() and ff_p.exists()):
        return None
    bx = pickle.load(open(bx_p, "rb"))
    ff = pickle.load(open(ff_p, "rb"))
    out = {}

    # COMMON-WINDOW FAIRNESS FIX: KG's fusion starts scoring at ev0 = W + INIT, later
    # than the baselines (which start at W). To compare identical commits, every model
    # is scored on the common window [ev0, N). F+G already starts at ev0 (its natural
    # window); the baselines are re-scored on [ev0, N) from their stored raw predictions.
    ev0 = cw.common_ev0(folder)
    rfs = OUTP / folder / "raw_fusion_scores.pkl"
    if ev0 is None or not rfs.exists():
        return None
    d = pickle.load(open(rfs, "rb")); N = d["N"]
    yv = np.asarray(d["y"], int); p_ev = np.asarray(d["scores"]["F+G"], float)
    fm = final_metrics(yv, p_ev, gap=gap)                  # F+G on [ev0, N)
    out["FG"] = {m: float(fm.get(m, np.nan)) for m in PERF}
    # F+G effort (Popt/ACC20): read from effort_results.pkl (run_effort_eval.py's
    # "Fusion(F+G)" subset). final_fusion_results.pkl's metrics dict has NO effort
    # keys, which previously left the F+G effort columns as "--". Effort is not
    # gap-tuned, so the stored-span value is used for both settings (baselines do the
    # same at their stored span below).
    ef_p = OUTP / folder / "effort_results.pkl"
    fge = {}
    if ef_p.exists():
        fge = pickle.load(open(ef_p, "rb")).get("models", {}).get("Fusion(F+G)", {})
    for e in EFFORT:
        out["FG"][e] = float(fge.get(e, np.nan))
    raws = bx.get("raws", {})
    for key, _ in BASE5:
        if key not in raws:
            continue
        bm = cw.metrics_from_raw(raws[key], ev0, gap=gap)   # baseline on [ev0, N)
        if bm is None:
            continue
        rec = {m: float(bm.get(m, np.nan)) for m in PERF}
        bm0 = bx["baselines"].get(key, {})
        for e in EFFORT:                                     # effort: stored (own span)
            rec[e] = float(bm0.get(e, np.nan))
        out[key] = rec
    return out or None


def _avg(rows, keys, metrics, weights=None):
    """Macro (unweighted) or Micro (weighted) average per model per metric."""
    agg = {}
    for mk in ["FG"] + keys:
        vals, ws = [], []
        for i, r in enumerate(rows):
            if r and mk in r and mk_ok(r[mk], metrics):
                vals.append([r[mk][m] for m in metrics]); ws.append(weights[i] if weights else 1.0)
        if vals:
            vals = np.array(vals); ws = np.array(ws)
            agg[mk] = {m: float(np.average(vals[:, j], weights=ws)) for j, m in enumerate(metrics)}
    return agg


def mk_ok(d, metrics):
    return all(m in d and d[m] == d[m] for m in metrics)


def render_table(rows_by_proj, metrics, caption, label, suffix):
    keys = [k for k, _ in BASE5]
    ncol = (1 + len(BASE5)) * len(metrics)
    lines = [r"\begin{table*}[t]\centering", r"\scriptsize\setlength{\tabcolsep}{3pt}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\begin{tabular}{l" + ("c" * ncol) + "}", r"\toprule"]
    head1 = [r"\multirow{2}{*}{Project}", r"\multicolumn{%d}{c}{KG-Commit ($F{+}G$)}" % len(metrics)]
    for _, lab in BASE5:
        head1.append(r"\multicolumn{%d}{c}{%s}" % (len(metrics), lab))
    lines.append(" & ".join(head1) + r" \\")
    cmids = []; s = 2
    for _ in range(1 + len(BASE5)):
        cmids.append(r"\cmidrule(lr){%d-%d}" % (s, s + len(metrics) - 1)); s += len(metrics)
    lines.append(" ".join(cmids))
    sub = [""] + [PP[m] for m in metrics] * (1 + len(BASE5))
    lines.append(" & ".join(sub) + r" \\ \midrule")

    disp_rows = []
    for disp, folder in PROJECTS:
        r = rows_by_proj.get(folder)
        cells = [disp]
        for mk in ["FG"] + keys:
            for m in metrics:
                v = r[mk][m] if (r and mk in r and m in r[mk] and r[mk][m] == r[mk][m]) else None
                cells.append("%.3f" % v if v is not None else "--")
        lines.append(" & ".join(cells) + r" \\")
        disp_rows.append(r)
    # Macro + Micro averages
    weights = [ (pickle.load(open(OUTP/f/"baseline_extra_results.pkl","rb"))["n_eval"]
                 if (OUTP/f/"baseline_extra_results.pkl").exists() else 0)
                for _, f in PROJECTS]
    for avglab, w in [("Macro-Avg", None), ("Micro-Avg", weights)]:
        agg = _avg(disp_rows, keys, metrics, w)
        cells = [r"\textbf{%s}" % avglab]
        for mk in ["FG"] + keys:
            for m in metrics:
                v = agg.get(mk, {}).get(m)
                cells.append(r"\textbf{%.3f}" % v if v is not None else "--")
        lines.append(r"\midrule " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.40)
    ap.add_argument("--out", default=str(ROOT / "Paper" / "paper_material" / "RQ1_performance"))
    args = ap.parse_args()
    suffix = "g0" if (args.gap == 0 and abs(args.warmup - 0.40) < 1e-9) else f"g{args.gap}"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for disp, folder in PROJECTS:
        m = project_metrics(folder, args.gap, args.warmup)
        if m:
            rows[folder] = m
            print(f"  {disp}: ok")
        else:
            print(f"  {disp}: MISSING (needs baseline_extra + fusion)")

    setting = ("Setting A ($G{=}0,K{=}0.40$)" if suffix == "g0"
               else f"Setting B ($G{{=}}{args.gap},K{{=}}{args.warmup:g}$)")
    cw_note = (r" All models are scored on the \emph{common evaluation window} "
               r"$[W{+}300,N)$, so every metric is over identical commits.")
    t5 = render_table(rows, PERF,
                      f"RQ1 performance under the online protocol, {setting}: KG-Commit "
                      r"($F{+}G$) vs.\ the five baselines. Best average in \textbf{bold}."
                      + cw_note,
                      f"tab:rq1_perf_{suffix}", suffix)
    (out / f"table5_performance_{suffix}.tex").write_text(t5, encoding="utf-8")
    t6 = render_table(rows, EFFORT,
                      f"RQ1 effort-aware evaluation, {setting}: KG-Commit ($F{{+}}G$) vs.\\ "
                      r"the five baselines ($P_{\mathrm{opt}}$, ACC@20)." + cw_note,
                      f"tab:rq1_effort_{suffix}", suffix)
    (out / f"table6_effort_{suffix}.tex").write_text(t6, encoding="utf-8")
    print(f"wrote table5_performance_{suffix}.tex + table6_effort_{suffix}.tex -> {out}")


if __name__ == "__main__":
    main()
