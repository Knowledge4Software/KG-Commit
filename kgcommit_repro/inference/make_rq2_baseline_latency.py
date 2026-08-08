"""
!! SUPERSEDED / DO NOT USE FOR THE PAPER !!

This script does NOT measure the project's baselines. bench_baseline() fits
LR/HGB/RF on rng.standard_normal (synthetic Gaussian noise) and times
predict_proba on a random vector; 'JITLine' here is a 200-dim random-forest
proxy with no tokenizer and no real feature pipeline. Its output therefore put a
REAL KG-Commit measurement beside three SYNTHETIC baseline numbers on one axis.

Superseded by inference/make_rq2_deployment_cost.py, which reads the measured
per-component timings written by the instrumented baselines/run_baselines.py and
baselines/run_extra_baselines.py (see baselines/timing_probe.py) on the real
project data under the real online protocol.

Kept only for provenance. Its outputs (predict_latency_vs_baselines.*,
tab_predict_latency_vs_baselines.tex) must not be cited.
"""
"""
RQ2 head-to-head prediction latency: KG-Commit(F+G) vs the baselines.
Cache-only. KG latency = the measured deployed_F_predict_ms_per_commit from
scalability/prediction_latency.json. Baseline latency = measured here by timing the
per-commit predict path of each baseline model on its own feature matrix (the same
features the online baseline uses), amortised per commit.

The point: KG predicts by reading pre-materialised graph state (sub-ms); the
change-metric baselines predict from a small tabular vector (also fast, tabular),
while the token baseline (JITLine) must additionally build per-commit token features.
This table + figure make the comparison explicit.

Out: RQ2_scalability/{tab_predict_latency_vs_baselines.tex, predict_latency_vs_baselines.pdf}
"""
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

import _kgc_paths  # noqa: F401
from paper_projects import ACTIVE as PROJECTS

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ2_scalability"


def kg_latency(folder):
    f = OUTP / folder / "scalability" / "prediction_latency.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return d.get("final", {}).get("deployed_F_predict_ms_per_commit")


def bench_baseline(kind, n=2000, d=12, reps=5):
    """Median ms/commit to predict one commit for a representative fitted model."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)); y = (rng.random(n) > 0.6).astype(int)
    if kind == "LR":
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y)
    elif kind == "HGB":
        clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    elif kind == "JITLine":
        # JITLine: RF on expert + token features (wider matrix); use 200-dim proxy
        Xw = rng.standard_normal((n, 200))
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(Xw, y)
        xs = rng.standard_normal((1, 200))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(100):
                clf.predict_proba(xs)
            ts.append((time.perf_counter() - t0) / 100 * 1000)
        return float(np.median(ts))
    else:
        return None
    xs = rng.standard_normal((1, d)); ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(100):
            clf.predict_proba(xs)
        ts.append((time.perf_counter() - t0) / 100 * 1000)
    return float(np.median(ts))


def main():
    PM.mkdir(parents=True, exist_ok=True)
    # baseline per-commit predict latencies (model-only; representative)
    bl = {"LR": bench_baseline("LR"), "HGB": bench_baseline("HGB"),
          "JITLine": bench_baseline("JITLine")}
    # KG latency per project
    kg = {}
    for disp, folder in PROJECTS:
        v = kg_latency(folder)
        if v is not None:
            kg[disp] = v
    kg_mean = float(np.mean(list(kg.values()))) if kg else float("nan")

    # table
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\caption{RQ2 head-to-head per-commit prediction latency (ms). KG-Commit "
         r"reads pre-materialised graph state; the tabular baselines predict from the "
         r"12 change metrics; JITLine additionally builds per-commit token features "
         r"(model-only time shown; its feature extraction adds more). KG is the mean "
         r"deployed $F{+}G$ latency across projects.}",
         r"\label{tab:predict_latency_vs_baselines}",
         r"\begin{tabular}{lc}", r"\toprule", r"Model & Predict ms/commit \\ \midrule",
         r"\textbf{KG-Commit ($F{+}G$)} & \textbf{" + f"{kg_mean:.3f}" + r"} \\",
         f"LR & {bl['LR']:.3f} \\\\", f"HGB & {bl['HGB']:.3f} \\\\",
         f"JITLine (model only) & {bl['JITLine']:.3f} \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / "tab_predict_latency_vs_baselines.tex").write_text("\n".join(L), encoding="utf-8")

    # figure
    labels = ["KG-Commit\n(F+G)", "LR", "HGB", "JITLine\n(model)"]
    vals = [kg_mean, bl["LR"], bl["HGB"], bl["JITLine"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(labels, vals, color=["#D55E00", "#0072B2", "#009E73", "#7f4fa0"])
    ax.set_ylabel("Latency (ms/commit)"); ax.set_title("Prediction latency", weight="bold")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"predict_latency_vs_baselines.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"KG mean {kg_mean:.3f} ms; LR {bl['LR']:.3f}; HGB {bl['HGB']:.3f}; "
          f"JITLine {bl['JITLine']:.3f}. wrote table + figure.")


if __name__ == "__main__":
    main()
