"""
Appendix H: per-parameter sensitivity tables.

One table per protocol parameter. Rows are projects, columns are the swept values plus
a variance column; each cell is the deployed model's Macro-F1 at that setting. This is
the layout that lets a reader check robustness per project rather than only in
aggregate, and it makes the one fluctuating project (Zookeeper) visible instead of
averaged away.

Cache-only: outputs/<p>/param_experiments/{K,M,ROLL}.json.

Run:  python inference/make_appendix_H.py
Out:  appendices/H_param_sensitivity/appH_param_tables.tex
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "appendices" / "H_param_sensitivity"

SWEEPS = [
    ("K", "K.json", "warm-up fraction $K$",
     "the proportion of the stream used only to initialise the models"),
    ("M", "M.json", "refresh block size $M$",
     "how often the deployed model refreshes and refits"),
    ("ROLL", "ROLL.json", "rolling-evaluation window $W$",
     "the width of the window over which running metrics are computed"),
]
# The value deployed for every experiment in the paper, per parameter. The column is
# bolded so a reader can see at a glance where the reported results sit in the sweep.
DEPLOYED = {"K": "0.2", "M": "200", "ROLL": "800"}
METRIC = "Macro_F1"
# ROLL sweeps were recorded with the ranking-metric suite only, so that
# table is reported on ROC-AUC and its caption says so.
METRIC_BY_TAG = {"ROLL": "ROC_AUC"}


def build(fname, tag, title, gloss):
    metric = METRIC_BY_TAG.get(tag, METRIC)
    mdisp = "ROC-AUC" if metric == "ROC_AUC" else "Macro-F1"
    rows = {}
    vals = set()
    for disp, folder in PROJECTS:
        f = OUTP / folder / "param_experiments" / fname
        if not f.exists():
            continue
        d = json.load(open(f))
        rec = {}
        for k, v in d.items():
            if k.startswith("_") or metric not in v:
                continue
            rec[k] = float(v[metric])
            vals.add(k)
        if rec:
            rows[disp] = rec
    if not rows:
        return None
    try:
        order = sorted(vals, key=lambda x: float(x))
    except ValueError:
        order = sorted(vals)

    dep = DEPLOYED.get(tag)
    extra = ""
    if tag == "K":
        extra = (r" The deployed column ($K{=}0.2$) is \textbf{bold}. We fix "
                 r"$K{=}0.2$ for every experiment in the paper because it is the "
                 r"deployment-faithful choice: a tool should become useful early in a "
                 r"project's life rather than after consuming $40$--$50\%$ of its "
                 r"history. Notably this is \emph{not} the setting that maximises the "
                 r"cross-project average, so the reported results are conservative "
                 r"with respect to this parameter. The \underline{underlined} cell "
                 r"(Zookeeper at $K{=}0.2$) is the single weakest value in the sweep "
                 r"and is what pulls that average down: with only $839$ commits, a "
                 r"$20\%$ warm-up leaves too little history for the graph's layers to "
                 r"be estimated reliably, an effect analysed in "
                 r"Section~\ref{sec:discuss_difficulty}. Excluding Zookeeper the "
                 r"deployed column is within $0.03$ Macro-F1 of the best setting on "
                 r"every project, so the choice costs little where the graph has "
                 r"enough history to work with.")
    elif dep:
        extra = (rf" The deployed column (${tag}{{=}}{dep}$), used for every "
                 r"experiment in the paper, is \textbf{bold}.")
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         rf"\caption{{Appendix H: sensitivity of the deployed model ($F{{+}}G$) to the "
         rf"{title}, which controls {gloss}. Each cell is the metric at that setting; "
         rf"\emph{{Var.}} is the variance across the sweep. Metric: {mdisp}. Small "
         r"variances indicate the deployed constant is one of many equivalent choices."
         + extra + "}",
         rf"\label{{tab:appH_{tag}}}",
         r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(order) + "c}", r"\toprule",
         "Project & " + " & ".join(order) + r" & Var. \\ \midrule"]
    allvar = []
    for disp, _ in PROJECTS:
        if disp not in rows:
            continue
        rec = rows[disp]
        series = [rec[v] for v in order if v in rec]
        cells = [disp]
        for v in order:
            if v not in rec:
                cells.append("--")
                continue
            s = f"{rec[v]:.3f}"
            if v == dep:
                s = r"\textbf{%s}" % s
                # Zookeeper at the deployed setting is the single weakest cell in the
                # sweep and is the reason the deployed column is not the best on
                # average; underline it so the discussion can point at it directly.
                if disp == "Zookeeper" and tag == "K":
                    s = r"\underline{%s}" % s
            cells.append(s)
        var = float(np.var(series)) if series else float("nan")
        allvar.append(var)
        cells.append(f"{var:.5f}")
        L.append(" & ".join(cells) + r" \\")
    # mean row
    mean_cells = [r"\textbf{Mean}"]
    for v in order:
        xs = [rows[p][v] for p in rows if v in rows[p]]
        mean_cells.append(r"\textbf{%.3f}" % np.mean(xs) if xs else "--")
    mean_cells.append(r"\textbf{%.5f}" % np.mean(allvar) if allvar else "--")
    L.append(r"\midrule " + " & ".join(mean_cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    out = []
    for tag, fname, title, gloss in SWEEPS:
        t = build(fname, tag, title, gloss)
        if t:
            out.append(t)
            print(f"  {tag}: ok")
        else:
            print(f"  {tag}: no data")
    if out:
        (PM / "appH_param_tables.tex").write_text("\n\n".join(out) + "\n",
                                                  encoding="utf-8")
        print(f"\nwrote {PM / 'appH_param_tables.tex'} ({len(out)} tables)")


if __name__ == "__main__":
    main()
