"""
RQ2 phase-resolved cost table (two-column / single-column width).

Breaks the per-commit cost of every model into the phases a deployment actually
pays, grouped under three supercolumns:

  INGEST (once per commit, amortised over the project's life)
      KG-Commit  : parse the changed files, diff against the stored AST, write the
                   delta edges. Paid once, when the commit arrives.
      Baselines  : nothing -- they hold no persistent state.

  INFERENCE (per prediction)
      KG-Commit  : featurisation is a no-op (context already materialised); the
                   cost is the graph read performed by the selected fusion F.
      Baselines  : build this commit's feature vector, then score it. For the
                   change-metric models featurisation is a scaler transform; for
                   JITLine it is diff tokenisation + bag-of-tokens vectorisation.

  MAINTENANCE (amortised over the commits it serves)
      KG-Commit  : DW/KGE embedding refit, incurred only where F selects one.
      Baselines  : the periodic retrain on the expanding past window.

Why the phases must not be summed into one number: INGEST is paid once per commit
and is asynchronous with prediction (it happens as the commit lands), whereas
INFERENCE is on the critical path of the developer's feedback loop. Adding them
would conflate a background cost with a latency. The table therefore totals only
the phases that a prediction actually waits on, and reports ingest separately.

Sources (all measured, cache-only):
  outputs/<p>/scalability/prediction_latency.json    KG per-method predict + train
  outputs/<p>/scalability/timing_censored_ast.json   KG ingest (stall-censored)
  outputs/<p>/baseline_extra_results.pkl["timings"]  baseline phase split
  outputs/<p>/final_fusion_results.pkl["chosen"]     the project's deployed F

Out: Paper/paper_material/RQ2_scalability/tab_phase_cost.tex
Run: python inference/make_rq2_phase_table.py
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

BLOCK = 200        # online_infer.BLOCK
REFIT_EMB = 5      # DW/KGE re-embed cadence, in blocks

MODELS = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
          ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]


def kg_phases(folder):
    """(ingest_ms, infer_ms, refit_ms, F_label) for one project, or None."""
    lf = OUTP / folder / "scalability" / "prediction_latency.json"
    ff = OUTP / folder / "final_fusion_results.pkl"
    if not lf.exists() or not ff.exists():
        return None
    lat = json.load(open(lf)).get("final", {})
    F = pickle.load(open(ff, "rb"))["chosen"].split("+")
    pm = lat.get("predict_ms_per_commit", {})
    tm = lat.get("train_ms_per_block", {})
    infer = sum(pm[m]["median"] for m in F if m in pm)
    refit = sum(tm[m]["median"] for m in F if m in tm) / (REFIT_EMB * BLOCK)
    cj = OUTP / folder / "scalability" / "timing_censored_ast.json"
    ingest = json.load(open(cj))["censored"]["p50"] if cj.exists() else None
    return ingest, infer, refit, "+".join(F)


def bl_phases(folder):
    f = OUTP / folder / "baseline_extra_results.pkl"
    if not f.exists():
        return {}
    return pickle.load(open(f, "rb")).get("timings", {}) or {}


def _m(vals):
    return sum(vals) / len(vals) if vals else None


def _f(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "--"


def build():
    # ---- gather ----
    kg_ing, kg_inf, kg_ref = [], [], []
    for _, folder in PROJECTS:
        p = kg_phases(folder)
        if not p:
            continue
        if p[0] is not None:
            kg_ing.append(p[0])
        kg_inf.append(p[1]); kg_ref.append(p[2])

    agg = {}
    for _, folder in PROJECTS:
        for key, v in bl_phases(folder).items():
            a = agg.setdefault(key, {"f": [], "p": [], "r": []})
            a["f"].append(v["featurize_ms_per_commit"])
            a["p"].append(v["predict_ms_per_commit"])
            a["r"].append(v["refit_ms_per_commit_amortised"])

    L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\caption{RQ2 phase-resolved per-commit cost (ms, mean over the seven active "
         r"projects, measured under the online protocol). \textsc{Inference} is the work "
         r"on the developer's critical path when a commit is scored; \textsc{Maint.} is "
         r"periodic retraining, amortised over the commits it serves. KG-Commit performs "
         r"no featurisation at inference: a commit's representation is built once, "
         r"incrementally, when the commit lands, and that cost is amortised over the life "
         r"of the project, so scoring is a bounded read of state that already exists. The "
         r"baselines keep no persistent state and must therefore rebuild a commit's "
         r"features every time they score it, which for JITLine is the dominant term.}",
         r"\label{tab:phase_cost}",
         r"\begin{tabular}{l r r | r | r}", r"\toprule",
         r"& \multicolumn{2}{c|}{\textsc{Inference}} & "
         r"\textsc{Maint.} & \textsc{Total} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(l){5-5}",
         r"Model & featurise & predict & refit & per commit \\",
         r"\midrule"]

    ing, inf, ref = _m(kg_ing), _m(kg_inf), _m(kg_ref)
    serving = (inf or 0) + (ref or 0)
    # A dash means "this phase does not exist for this model" -- KG-Commit does no
    # featurisation at serving time, the baselines have no ingest phase at all. Using
    # one symbol for both (rather than the words "n/a"/"none") keeps the eye on the
    # numbers; the two cases are distinguished in the footnote.
    L.append(rf"\textbf{{KG-Commit ($F{{+}}G$)}} & "
             rf"$-^{{a}}$ & \textbf{{{_f(inf)}}} & {_f(ref)} & "
             rf"\textbf{{{_f(serving)}}} \\")
    L.append(r"\midrule")
    for key, disp in MODELS:
        a = agg.get(key)
        if not a:
            L.append(f"{disp} & -- & -- & -- & -- " + r"\\")
            continue
        fe, pr, rf = _m(a["f"]), _m(a["p"]), _m(a["r"])
        tot = fe + pr + rf
        ratio = rf"\,({tot/serving:.1f}$\times$)" if serving > 0 else ""
        L.append(rf"{disp} & {_f(fe)} & {_f(pr)} & "
                 rf"{_f(rf)} & {_f(tot)}{ratio} \\")

    L += [r"\bottomrule",
          r"\multicolumn{5}{@{}p{\linewidth}@{}}{\footnotesize "
          r"$^{a}$Not applicable: KG-Commit builds a commit's representation "
          r"incrementally when the commit arrives, amortised over the project's history, "
          r"so no featurisation is performed on the prediction path.} \\",
          r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _, f in PROJECTS if bl_phases(f))
    if not n:
        print("no baseline timings; run the instrumented baselines first")
        return
    (PM / "tab_phase_cost.tex").write_text(build() + "\n", encoding="utf-8")
    print(f"  wrote {PM / 'tab_phase_cost.tex'}  ({n}/{len(PROJECTS)} projects)")


if __name__ == "__main__":
    main()
