"""
LaTeX tables for the parameter-sensitivity (Discussion) experiments.

One table per swept parameter (K, M, ROLL, l): rows = parameter values, column
groups = the five methods + the deployed fusion F, cells = whole-stream headline
metrics (Buggy-F1 and AUC by default -- the two most informative). A companion
"F-only, all 7 metrics" table gives the full metric response of the deployed
fusion to each parameter.

Tables are written to outputs/param_experiments/ and copied to
docs/figures/v4/param/ so the paper can \input{figures/v4/param/...}.

Run:  python scalability/make_param_tables.py
"""
import json

import _common as C

PARAMS = ["K", "M", "ROLL", "l", "KM"]
PARAM_LABEL = {"K": "Warmup fraction $K$", "M": "Label gap $M$ (commits)",
               "ROLL": "Rolling window (commits)", "l": "CSTG refresh $l$ (commits)",
               "KM": "Warmup fraction $K$ (at fixed gap $M{=}200$)"}
PARAM_NOTE = {
    "K": "fraction of the stream used to fit before scoring begins",
    "M": "labels of the $M$ commits just before each block are withheld "
         "(late-arriving labels); graph structure is still visible",
    "ROLL": "trailing window each stream-plot point averages over "
            "(a reporting choice; whole-stream scores are invariant to it)",
    "l": "the CSTG Term-hub weights are refreshed from past-only NPMI every $l$ "
         "commits (the graph the methods walk; the CSTG classifier $G$ is not used)",
    "KM": "warmup swept (including very low warmup) under a fixed realistic label "
          "gap $M{=}200$; labels-only baselines (JIT-metrics LR, naive prior rate) "
          "are evaluated in the identical setup for a real-world-constraint comparison",
}
METHODS = ["RN", "PPR", "LP", "DW", "KGE", "F"]
BASELINES = ["JIT_LR", "Naive"]
BASE_LABEL = {"JIT_LR": "JIT-LR", "Naive": "Naive"}
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
M7_SHORT = {"Precision": "Prec.", "Recall": "Rec.", "Macro_F1": "Macro-F1",
            "Buggy_F1": "Buggy-F1", "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Acc."}


def z(x, d=3):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("0.") else (("-" + s[2:]) if s.startswith("-0.") else s)


def _load(param):
    p = C.OUT_PARAM / f"{param}.json"
    return json.load(open(p)) if p.exists() else None


def write(name, body):
    C.OUT_PARAM.mkdir(parents=True, exist_ok=True)
    (C.OUT_PARAM / name).write_text(body, encoding="utf-8")
    C.FIG_PARAM.mkdir(parents=True, exist_ok=True)
    (C.FIG_PARAM / name).write_text(body, encoding="utf-8")
    print(f"  {name}")


def _vals(d):
    grid = d["_meta"]["grid"]
    return [str(v) for v in grid]


def tab_param_bf1auc(param, d):
    """rows = param values; per method: Buggy-F1 / AUC."""
    vals = _vals(d)
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in METHODS)
    sub = " & ".join(["BF1 & AUC"] * len(METHODS))
    rows = []
    for v in vals:
        cells = []
        for m in METHODS:
            w = d[v][m]["whole"]
            cells.append(f"{z(w['Buggy_F1'])} & {z(w['AUC'])}")
        rows.append(f"{v} & " + " & ".join(cells) + " \\\\")
    body = (
        "% Auto-generated -- do not edit by hand.\n"
        "\\begin{table}[t]\n\\centering\\small\n"
        f"\\caption{{Sensitivity to the {PARAM_LABEL[param]}: Buggy-F1 (BF1) and AUC "
        f"of the five methods and the deployed fusion $F$ on the final "
        f"\\emph{{Core+AST+CSTG}} graph. {PARAM_NOTE[param]}.}}\n"
        f"\\label{{tab:param-{param}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l" + "rr" * len(METHODS) + "}\n\\toprule\n"
        f"{PARAM_LABEL[param]} & {header} \\\\\n"
        f" & {sub} \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    write(f"tab_param_{param}.tex", body)


def tab_fusion_all7(param, d):
    """F-only, all 7 metrics vs param value."""
    vals = _vals(d)
    rows = []
    for v in vals:
        w = d[v]["F"]["whole"]
        rows.append(f"{v} & " + " & ".join(z(w[k]) for k in M7) + " \\\\")
    header = " & ".join(M7_SHORT[k] for k in M7)
    body = (
        "% Auto-generated -- do not edit by hand.\n"
        "\\begin{table}[t]\n\\centering\\small\n"
        f"\\caption{{Deployed fusion $F=$~RN$+$PPR: all seven headline metrics vs.\\ "
        f"the {PARAM_LABEL[param]} (final \\emph{{Core+AST+CSTG}} graph, no $G$/$M$ "
        f"channels).}}\n\\label{{tab:param-{param}-F}}\n"
        "\\begin{tabular}{l" + "r" * len(M7) + "}\n\\toprule\n"
        f"{PARAM_LABEL[param]} & {header} \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(f"tab_param_{param}_F.tex", body)


def tab_km_compare(d):
    """KM: ours (F + 5 methods) vs labels-only baselines, under fixed gap M=200,
    across the warmup sweep. Reports Buggy-F1 / AUC / G-Mean."""
    vals = _vals(d)
    cols = ["F", "PPR", "RN", "LP", "DW", "KGE", "JIT_LR", "Naive"]
    collab = {**{m: m for m in METHODS}, **BASE_LABEL}
    submetrics = [("Buggy_F1", "BF1"), ("AUC", "AUC"), ("G_Mean", "GM")]
    header = " & ".join(f"\\multicolumn{{3}}{{c}}{{{collab[c]}}}" for c in cols)
    sub = " & ".join(["BF1 & AUC & GM"] * len(cols))
    rows = []
    for v in vals:
        cells = []
        for c in cols:
            w = d[v][c]["whole"]
            cells.append(" & ".join(z(w[mk]) for mk, _ in submetrics))
        rows.append(f"{v} & " + " & ".join(cells) + " \\\\")
    body = (
        "% Auto-generated -- do not edit by hand.\n"
        "\\begin{table}[t]\n\\centering\\scriptsize\n"
        "\\caption{Real-world-constraint comparison on the final "
        "\\emph{Core+AST+CSTG} graph: our five methods and the deployed fusion $F$ "
        "vs.\\ labels-only baselines (JIT-metrics logistic regression; naive prior "
        "bug-rate), under a \\emph{fixed} label gap $M{=}200$ and swept warmup $K$ "
        "(including very low warmup). Every predictor uses the identical prequential "
        "setup and only the labelled past $[0,i{-}M)$. Buggy-F1 (BF1), AUC and "
        "G-Mean (GM).}\n\\label{tab:param-KM-compare}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l" + "rrr" * len(cols) + "}\n\\toprule\n"
        f"$K$ & {header} \\\\\n & {sub} \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    write("tab_param_KM_compare.tex", body)


def main():
    print("rendering param tables ->", C.OUT_PARAM)
    for pm in PARAMS:
        d = _load(pm)
        if not d:
            print(f"  (skip {pm}: no results yet)")
            continue
        if pm == "KM":
            tab_km_compare(d)
            tab_fusion_all7(pm, d)
        else:
            tab_param_bf1auc(pm, d)
            tab_fusion_all7(pm, d)
    print("done.")


if __name__ == "__main__":
    main()
