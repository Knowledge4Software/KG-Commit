"""
D2/D3: cross-project robustness of the deployed model to the protocol parameters.

The existing D2 table reports a single project (Groovy). Every project has a stored
sweep, so this builds the Macro-Avg view: for each parameter value, the metric is
averaged over the seven active projects, and a variance column summarises stability
across the sweep.

Three parameters, from outputs/<p>/param_experiments/:
  K    warm-up fraction        {0.05, 0.10, 0.20, 0.30, 0.40, 0.50}
  M    refresh block size      {10, 25, 50, 100, 200}
  ROLL rolling-window length

Note on baselines: the stored sweeps cover the deployed KG model only. LR and HGB can
be re-swept cheaply from the change-metric CSV (see make_d2_combined.py), but
LApredict, Deeper and JITLine would each need their own pipeline re-run per parameter
value, so a full five-baseline sweep is out of scope here and the cross-project table
reports KG-Commit's own robustness.

Cache-only. No Neo4j.

Run:  python inference/make_d2_macroavg.py
Out:  discussions/D2_warmup_K/tab_param_macroavg.tex
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
PM = ROOT / "Paper" / "paper_material" / "discussions" / "D2_warmup_K"

SWEEPS = [
    ("K", "K.json", r"Warm-up fraction $K$"),
    ("M", "M.json", r"Refresh block size $M$"),
]
METRICS = [("Macro_F1", "Macro-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC")]


def load(knob_file):
    """{param_value: {metric: [per-project values]}}"""
    acc = {}
    for _, folder in PROJECTS:
        f = OUTP / folder / "param_experiments" / knob_file
        if not f.exists():
            continue
        d = json.load(open(f))
        for k, v in d.items():
            if k.startswith("_"):
                continue
            for mk, _ in METRICS:
                if mk in v:
                    acc.setdefault(k, {}).setdefault(mk, []).append(float(v[mk]))
    return acc


def build():
    blocks = []
    for tag, fname, title in SWEEPS:
        acc = load(fname)
        if not acc:
            continue
        try:
            vals = sorted(acc, key=lambda x: float(x))
        except ValueError:
            vals = sorted(acc)
        n = len(vals)

        L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
             rf"\caption{{D2: robustness of the deployed model to {title}, averaged over "
             r"the seven active projects. Each cell is the cross-project mean at that "
             r"parameter value; \emph{Var.} is the variance of that mean across the "
             r"sweep. A small variance means the deployed constant is one of many "
             r"equivalent choices rather than a tuned optimum.}",
             rf"\label{{tab:d2_macroavg_{tag}}}",
             r"\begin{tabular}{l" + "c" * n + "c}", r"\toprule",
             "Metric & " + " & ".join(vals) + r" & Var. \\ \midrule"]
        for mk, disp in METRICS:
            row = [disp]
            series = []
            for v in vals:
                xs = acc[v].get(mk, [])
                if xs:
                    m = float(np.mean(xs))
                    series.append(m)
                    row.append("%.3f" % m)
                else:
                    row.append("--")
            row.append((r"\textbf{%.5f}" % float(np.var(series))) if series else "--")
            L.append(" & ".join(row) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        blocks.append((tag, "\n".join(L)))
    return blocks


def main():
    PM.mkdir(parents=True, exist_ok=True)
    blocks = build()
    if not blocks:
        print("no param_experiments found")
        return
    out = "\n\n".join(b for _, b in blocks)
    (PM / "tab_param_macroavg.tex").write_text(out + "\n", encoding="utf-8")
    print(f"  wrote {PM / 'tab_param_macroavg.tex'} ({len(blocks)} tables)")
    for tag, _ in blocks:
        print(f"    - {tag}")


if __name__ == "__main__":
    main()
