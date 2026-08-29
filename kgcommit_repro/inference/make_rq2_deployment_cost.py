"""
RQ2 deployment-cost comparison: KG-Commit vs the baselines, all measured.

Replaces the earlier synthetic microbenchmark (make_rq2_baseline_latency.py, which
fitted models on rng.standard_normal and used a 200-dim proxy for JITLine). Every
number here is measured on the real project data, under the real online protocol,
in the same session:

  * baseline components  <- outputs/<p>/baseline_extra_results.pkl["timings"],
    written by the instrumented run_baselines / run_extra_baselines (timing_probe).
  * KG-Commit predict    <- outputs/<p>/scalability/prediction_latency.json
    (deployed F predict ms/commit on the final Core+AST+CSTG graph).

Why three components and not one number. A 12-feature tabular classifier's
predict_proba is trivially fast, so reporting predict alone would say nothing and
would flatter the baselines. The deployment question is the TOTAL per-commit cost:
build this commit's features, score it, and carry the model's periodic retraining.
KG-Commit has no featurisation step at inference (its context was materialised
incrementally when the commit arrived) and no retraining over history at all.

Out: Paper/paper_material/RQ2_scalability/
       tab_deployment_cost.tex        per-project totals, KG vs baselines
       tab_deployment_cost_breakdown.tex   featurize / predict / refit split
Run:  python inference/make_rq2_deployment_cost.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ2_scalability"

DISPLAY = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
           ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]


# Protocol constants governing how often KG-Commit's trainable parts are refit.
# These MUST track the runtime, not be restated: the refit term is amortised as
# cost/(REFIT_EMB*BLOCK), so a stale literal misreports the cost directly.
from protocol import BLOCK, REFIT_EVERY as REFIT_EMB  # noqa: F401


def kg_cost(folder):
    """KG-Commit's measured per-commit cost, using the project's ACTUAL chosen F.

    Two corrections over the stored `deployed_F_predict_ms_per_commit`, which is
    hard-coded to RN+PPR:
      * F is chosen per project (PPR / RN+PPR / RN+PPR+LP / RN+LP / RN+KGE), so the
        predict cost must sum over the methods that project actually deploys;
      * RN/PPR/LP are non-parametric (they read the graph directly), but DW and KGE
        carry a trainable embedding + logistic head that is refit every REFIT_EMB
        blocks. Where such a method is in F, that refit is a real recurring cost and
        is amortised here over the commits it serves.
    Returns (predict_ms, refit_ms_per_commit, F_label) or None.
    """
    lf = OUTP / folder / "scalability" / "prediction_latency.json"
    ff = OUTP / folder / "final_fusion_results.pkl"
    if not lf.exists() or not ff.exists():
        return None
    lat = json.load(open(lf)).get("final", {})
    F = pickle.load(open(ff, "rb"))["chosen"].split("+")
    pm = lat.get("predict_ms_per_commit", {})
    tm = lat.get("train_ms_per_block", {})
    predict = sum(pm[m]["median"] for m in F if m in pm)
    refit_block = sum(tm[m]["median"] for m in F if m in tm)
    refit = refit_block / (REFIT_EMB * BLOCK) if refit_block else 0.0
    return predict, refit, "+".join(F)


def kg_predict_ms(folder):
    c = kg_cost(folder)
    return None if c is None else c[0] + c[1]


def baseline_timings(folder):
    f = OUTP / folder / "baseline_extra_results.pkl"
    if not f.exists():
        return {}
    return pickle.load(open(f, "rb")).get("timings", {}) or {}


def _f(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "--"


def tab_totals():
    """Per-project total deployment cost per commit: KG vs each baseline."""
    L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{RQ2 measured per-commit \emph{deployment} cost (ms): featurisation "
         r"$+$ inference $+$ amortised retraining, under the online protocol on the real "
         r"project data. KG-Commit scores a commit by reading graph state that its "
         r"incremental update already materialised, so it has no inference-time "
         r"featurisation step and never retrains over history; the baselines must build "
         r"each commit's features and periodically refit. JITLine's cost is dominated by "
         r"diff tokenisation, which is precisely the context KG-Commit amortises.}",
         r"\label{tab:deployment_cost}",
         r"\begin{tabular}{ll" + "r" * (1 + len(DISPLAY)) + "}", r"\toprule",
         r"Project & $F$ & \textbf{KG-Commit ($F{+}G$)} & " +
         " & ".join(d for _, d in DISPLAY) + r" \\",
         r"\midrule"]
    sums = {k: [] for k, _ in DISPLAY}
    kgs = []
    for disp, folder in PROJECTS:
        kg = kg_predict_ms(folder)
        t = baseline_timings(folder)
        if kg is not None:
            kgs.append(kg)
        c = kg_cost(folder)
        cells = [rf"\texttt{{\scriptsize {c[2]}}}" if c else "--",
                 rf"\textbf{{{_f(kg)}}}" if kg is not None else "--"]
        for key, _ in DISPLAY:
            v = t.get(key, {}).get("total_ms_per_commit")
            if v is not None:
                sums[key].append(v)
            cells.append(_f(v))
        L.append(f"{disp} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    # summary rows carry an empty cell for the F column
    mean_cells = ["", rf"\textbf{{{_f(sum(kgs)/len(kgs))}}}" if kgs else "--"]
    for key, _ in DISPLAY:
        v = sums[key]
        mean_cells.append(_f(sum(v) / len(v)) if v else "--")
    L.append(r"\textbf{Mean} & " + " & ".join(mean_cells) + r" \\")
    # speed-up row, relative to KG
    if kgs:
        kgm = sum(kgs) / len(kgs)
        sp = [r"\textit{vs.\ KG-Commit}", "", r"\textit{1.0$\times$}"]
        for key, _ in DISPLAY:
            v = sums[key]
            sp.append(rf"{(sum(v)/len(v))/kgm:.1f}$\times$" if v else "--")
        L.append(" & ".join(sp) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def tab_breakdown():
    """Component split, averaged over projects: where each model's time goes."""
    agg = {}
    for _, folder in PROJECTS:
        for key, v in baseline_timings(folder).items():
            a = agg.setdefault(key, {"f": [], "p": [], "r": [], "t": []})
            a["f"].append(v["featurize_ms_per_commit"])
            a["p"].append(v["predict_ms_per_commit"])
            a["r"].append(v["refit_ms_per_commit_amortised"])
            a["t"].append(v["total_ms_per_commit"])
    kgs = [kg_predict_ms(f) for _, f in PROJECTS if kg_predict_ms(f) is not None]

    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{Where the per-commit cost goes (ms, mean over the active projects). "
         r"The model call is cheap for everyone -- that is not the differentiator. The "
         r"differentiator is \emph{featurisation}: KG-Commit performs it incrementally at "
         r"commit-arrival, so at inference the context already exists, whereas JITLine "
         r"re-tokenises every diff. KG-Commit's refit cost is small but not zero: RN, PPR "
         r"and LP are non-parametric graph reads, while DW and KGE carry a trainable "
         r"embedding that is refit periodically. Six of the seven projects select an $F$ "
         r"of RN/PPR/LP only and so incur no embedding refit at all; Zookeeper "
         r"($F=$RN$+$KGE) is the exception and carries the whole of the mean shown here.}",
         r"\label{tab:deployment_cost_breakdown}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Model & Featurise & Predict & Refit$^{\dagger}$ & Total \\", r"\midrule"]
    costs = [kg_cost(f) for _, f in PROJECTS]
    costs = [c for c in costs if c is not None]
    if costs:
        n = len(costs)
        kp = sum(c[0] for c in costs) / n
        kr = sum(c[1] for c in costs) / n
        L.append(rf"\textbf{{KG-Commit ($F{{+}}G$)}} & \textit{{amortised}}$^{{\ddagger}}$ & "
                 rf"\textbf{{{kp:.3f}}} & {kr:.3f} & \textbf{{{kp + kr:.3f}}} \\")
        L.append(r"\midrule")
    for key, d in DISPLAY:
        a = agg.get(key)
        if not a:
            L.append(f"{d} & -- & -- & -- & -- " + r"\\")
            continue
        n = len(a["t"])
        L.append(f"{d} & {sum(a['f'])/n:.3f} & {sum(a['p'])/n:.3f} & "
                 f"{sum(a['r'])/n:.3f} & {sum(a['t'])/n:.3f} " + r"\\")
    L += [r"\bottomrule",
          r"\multicolumn{5}{@{}p{0.95\linewidth}@{}}{\footnotesize $^{\dagger}$Refit "
          r"wall-clock amortised over the commits it serves. For the baselines this is "
          r"the periodic retrain on the expanding past window; for KG-Commit it is the "
          r"DW/KGE embedding refit, incurred only where such a method is in that "
          r"project's $F$. $^{\ddagger}$KG-Commit builds a commit's representation when "
          r"the commit arrives (the incremental graph update, reported as build cost in "
          r"Table~\ref{tab:build_cost_censored}), not at inference time; the inference "
          r"path reads state that already exists.} \\",
          r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _, f in PROJECTS if baseline_timings(f))
    if not n:
        print("no baseline timings found -- run the instrumented "
              "baselines/run_extra_baselines.py first")
        return
    for name, fn in [("tab_deployment_cost", tab_totals),
                     ("tab_deployment_cost_breakdown", tab_breakdown)]:
        (PM / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
        print(f"  wrote {PM / (name + '.tex')}")
    print(f"\n{n}/{len(PROJECTS)} projects with measured baseline timings.")


if __name__ == "__main__":
    main()
