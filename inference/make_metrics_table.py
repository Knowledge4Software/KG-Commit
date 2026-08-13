"""
Comprehensive online-metrics table for every method (baselines + KG-v1/v2/v3):
Precision, Recall, Buggy-F1, Macro-F1, G-Mean, MCC (operating point, online
threshold) + ROC-AUC, PR-AUC (threshold-free). Prints a readable table and
LaTeX-ready rows for docs. All post-hoc from stored (y,p).

Run:  python inference/make_metrics_table.py
"""
import pickle
from pathlib import Path
import stream_metrics as sm

OUT = Path(__file__).resolve().parent.parent / "outputs"
R = pickle.load(open(OUT / "online_jit_results.pkl", "rb"))
ABL = pickle.load(open(OUT / "online_jit_ablation_v3.pkl", "rb"))
Mn = {v["name"]: v for v in R["methods"].values()}

# (display, group, kind, key)  kind: 'R' from results, 'A' from ablation
METHODS = [
    ("Naive prior bug-rate",        "base", "R", "Naive prior bug-rate"),
    ("LR / JIT metrics",            "base", "R", "LR / JIT metrics (incremental)"),
    ("RandomForest / JIT metrics",  "base", "R", "RandomForest / JIT metrics"),
    ("GradBoost / JIT metrics",     "base", "R", "GradBoost / JIT metrics"),
    ("Relational priors (wvRN)",    "v1",   "R", "Relational priors (wvRN)"),
    ("KG embedding (SVD/LSA)",      "v1",   "R", "KG embedding (SVD/LSA)"),
    ("Structural TF-IDF",           "v1",   "R", "Structural TF-IDF (incremental)"),
    ("Personalized PageRank",       "v1",   "R", "Personalized PageRank"),
    ("Fusion M+T+R+P (v1)",         "v1",   "A", "M+T+R+P"),
    ("Commit-text X (v2)",          "v2",   "A", "X"),
    ("Fusion+text M+T+R+P+X (v2)",  "v2",   "A", "M+T+R+P+X"),
    ("Fusion M+T+R (v2)",           "v2",   "A", "M+T+R"),
    ("CSTG G alone (v3)",           "v3",   "A", "G"),
    ("KG+CSTG M+T+R+G (v3)",        "v3",   "A", "M+T+R+G"),
    ("KG+CSTG M+T+R+P+G (FINAL)",   "v3",   "A", "M+T+R+P+G"),
]


def get(kind, key):
    src = Mn[key] if kind == "R" else ABL[key]
    f = sm.final_metrics(src["y"], src["p"])
    c = src["cum"]
    return (f["precision"], f["recall"], f["buggy_f1"], f["macro_f1"],
            f["gmean"], f["mcc"], c["ROC_AUC"], c["PR_AUC"])


def main():
    cols = ["Prec", "Rec", "BuggyF1", "MacroF1", "GMean", "MCC", "ROC", "PR"]
    print(f"{'method':<30}" + "".join(f"{c:>8}" for c in cols))
    print("-" * (30 + 8 * len(cols)))
    latex = []
    for disp, grp, kind, key in METHODS:
        v = get(kind, key)
        print(f"{disp:<30}" + "".join(f"{x:8.3f}" for x in v))
        latex.append(f"{disp} & " + " & ".join(f"{x:.3f}" for x in v) + r" \\")
    print("\n% ---- LaTeX rows (Prec Rec BuggyF1 MacroF1 GMean MCC ROC PR) ----")
    for grp in ("base", "v1", "v2", "v3"):
        for (disp, g, kind, key), lx in zip(METHODS, latex):
            if g == grp:
                print(lx)
        print(r"\midrule")


if __name__ == "__main__":
    main()
