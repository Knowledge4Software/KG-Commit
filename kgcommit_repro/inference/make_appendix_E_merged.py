"""
Appendix E, merged layout.

The previous layout emitted one 31-row table per project, which collide when typeset.
This version pairs projects two per table (projects as supercolumns, seven metrics as
columns beneath each) and adds Macro-Avg and Micro-Avg tables over all seven projects,
so the appendix is both readable and complete.

Cache-only: final_fusion_results.pkl (part1 = the 31 combinations).

Run:  python inference/make_appendix_E_merged.py
Out:  appendices/E_fusion_combos/appE_merged.tex
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "appendices" / "E_fusion_combos"

M7 = [("Precision", "P"), ("Recall", "R"), ("Macro_F1", "MF1"),
      ("Buggy_F1", "BF1"), ("G_Mean", "GM"), ("AUC", "AUC"), ("ACC", "Acc")]

# The tolerance that actually drove selection (run_final_fusion.py:121). Imported
# rather than re-declared so the highlighted band can never drift from the real one.
TOL = 0.005
try:
    from run_final_fusion import TOL as _T  # noqa: E402
    TOL = _T
except Exception:                            # needs KGC_PROJECT; fall back to the constant
    pass


def load(folder):
    f = OUTP / folder / "final_fusion_results.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def combos_of(d):
    return sorted(d["part1"], key=lambda k: (d["part1"][k]["n"], k))


def _tab(caption, label, colgroups, order, getter):
    """colgroups = [(display, payload)]; getter(payload, combo, metric) -> float|None."""
    ncol = len(colgroups) * len(M7)
    L = [r"\begin{table*}[tp]\centering\tiny",
         r"\setlength{\tabcolsep}{2.6pt}\renewcommand{\arraystretch}{0.95}",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
         "Combination & " + " & ".join(
             r"\multicolumn{%d}{c}{%s}" % (len(M7), disp) for disp, _ in colgroups)
         + r" \\"]
    cm = []
    st = 2
    for _ in colgroups:
        cm.append(r"\cmidrule(lr){%d-%d}" % (st, st + len(M7) - 1))
        st += len(M7)
    L.append(" ".join(cm))
    # Repeat the LABELS then join. Multiplying the joined string instead concatenates
    # the blocks with no "&" between them, fusing the last label of one project with
    # the first of the next ("Acc" + "P" -> "AccP"): the header then has one cell too
    # few and every label in the right-hand block is displayed one column to the left.
    L.append(" & " + " & ".join([lbl for _, lbl in M7] * len(colgroups))
             + r" \\ \midrule")
    # Same emphasis as the per-project Table 11 (make_rq4_tables.table9_combos), applied
    # independently within each project's block: bold every Macro-F1 inside the selection
    # band (within TOL of that project's best), and underline the deployed fusion. The
    # aggregate panel carries no chosen fusion, so it is left unemphasised.
    band, chosen_of = {}, {}
    for disp, payload in colgroups:
        if not isinstance(payload, dict) or "part1" not in payload:
            continue
        vals = [getter(payload, c, "Macro_F1") for c in order]
        vals = [v for v in vals if v is not None and v == v]
        if vals:
            band[disp] = max(vals) - TOL
        chosen_of[disp] = payload.get("chosen")

    for combo in order:
        cells = [combo.replace("+", r"{+}")]
        for disp, payload in colgroups:
            for mk, _ in M7:
                v = getter(payload, combo, mk)
                s = "--" if v is None else f"{v:.3f}".lstrip("0")
                if v is not None and v == v and mk == "Macro_F1":
                    if disp in band and v >= band[disp]:
                        s = r"\textbf{%s}" % s
                    if combo == chosen_of.get(disp):
                        s = r"\underline{%s}" % s
                cells.append(s)
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    data = {}
    for disp, folder in PROJECTS:
        d = load(folder)
        if d:
            data[disp] = d
    if not data:
        print("no data")
        return
    order = combos_of(next(iter(data.values())))

    def per_project(d, combo, mk):
        e = d["part1"].get(combo)
        return None if not e else float(e["metrics"].get(mk, float("nan")))

    out = []
    names = list(data)
    for i in range(0, len(names), 2):
        pair = names[i:i + 2]
        out.append(_tab(
            "Appendix E: all 31 fusion combinations on "
            + " and ".join(pair)
            + r". \textbf{Bold} marks every combination whose Macro-F1 lies within "
            + rf"the selection tolerance $\tau={TOL}$ of that project's best; "
            + r"\underline{underline} marks the deployed fusion, chosen from that "
            + r"band as the one with the fewest inference methods.",
            f"tab:appE_merged_{i//2}",
            [(p, data[p]) for p in pair], order, per_project))

    # aggregates
    weights = {}
    for disp, folder in PROJECTS:
        bx = OUTP / folder / "baseline_extra_results.pkl"
        weights[disp] = (pickle.load(open(bx, "rb"))["n_eval"] if bx.exists() else 1)

    def agg(kind, combo, mk):
        vals, ws = [], []
        for p, d in data.items():
            e = d["part1"].get(combo)
            if e and e["metrics"].get(mk) == e["metrics"].get(mk):
                vals.append(float(e["metrics"][mk]))
                ws.append(weights[p] if kind == "micro" else 1.0)
        return float(np.average(vals, weights=ws)) if vals else None

    out.append(_tab(
        "Appendix E: all 31 fusion combinations, averaged over the seven active "
        "projects (unweighted Macro-Avg and commit-weighted Micro-Avg). These "
        "aggregates identify the best combination \\emph{on average}; the deployed "
        "model still selects $F$ per project, since no single combination is best "
        "everywhere.",
        "tab:appE_merged_avg",
        [("Macro-Avg", "macro"), ("Micro-Avg", "micro")], order, agg))

    (PM / "appE_merged.tex").write_text("\n\n".join(out) + "\n", encoding="utf-8")
    print(f"  wrote {PM / 'appE_merged.tex'} ({len(out)} tables, {len(order)} combos)")


if __name__ == "__main__":
    main()
