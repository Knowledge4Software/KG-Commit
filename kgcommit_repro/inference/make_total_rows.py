"""
"Total" rows for the four existing draft tables that carry Macro/Micro means.
============================================================================

Macro-Mean weights every project equally; Micro-Mean weights by commit count.
Neither is what a practitioner running one large deployment sees. The Total row
answers that directly: pool every scored commit from all 11 projects into a
SINGLE stream and evaluate it once, as though it were one project.

Covers:
  tab:rq1_perf_g50     Macro-F1 / G-Mean / AUC, 8 model columns
  tab:rq1_effort_g50   P_opt / ACC@20%LOC
  tab:significance     Macro-F1 per baseline (+ the pooled paired test)
  tab:rq4_fg_g50       F -> F+G -> Switch@200 build-up

Cache-only, and reuses make_teammate_artifacts' SHA-joined loader so the pooled
stream pairs the same commits across models.

Out: outputs/tables/total_rows.json  (+ a LaTeX fragment per table)
Run: python inference/make_total_rows.py
"""
import json
import os
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402
from make_teammate_artifacts import (  # noqa: E402
    load_project, score_all, PROJECTS, MODELS, BASELINES, DISP)
from online_jit import online_decisions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"


def pooled():
    """Per-project data plus the pooled DECISION stream.

    For each project and model we take the online-tuned decisions that project
    would actually deploy, then concatenate those decisions across projects.
    No global re-thresholding: every commit is judged under its own project's
    operating point.
    """
    data = {}
    for p in PROJECTS:
        D = load_project(p)
        if D is not None:
            data[p] = D
    common = [m for m in MODELS if all(m in data[p] for p in data)]

    y = np.concatenate([data[p]["y"] for p in data])
    dec = {}
    for m in common:
        parts = []
        for p in data:
            pr = np.clip(np.nan_to_num(np.asarray(data[p][m], float),
                                       nan=float(np.mean(data[p]["y"]))), 0, 1)
            parts.append(online_decisions(pr, data[p]["y"], gap=P.GAP))
        dec[m] = np.concatenate(parts)
    return data, y, dec, common


def _from_decisions(y, yh):
    """Threshold metrics from a pooled confusion matrix."""
    tp = float(np.sum((yh == 1) & (y == 1)))
    fp = float(np.sum((yh == 1) & (y == 0)))
    fn = float(np.sum((yh == 0) & (y == 1)))
    tn = float(np.sum((yh == 0) & (y == 0)))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return {"Macro_F1": (f1p + f1n) / 2.0,
            "G_Mean": float(np.sqrt(rec * spec)),
            "Recall": rec, "Specificity": spec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _rank_weighted(data, m, key):
    """Commit-weighted mean of a rank-based metric (no threshold to pool)."""
    vals, wts = [], []
    for p in data:
        v = score_all(data[p]["y"], data[p][m], data[p]["effort"],
                      gap=P.GAP)[key]
        if np.isfinite(v):
            vals.append(v)
            wts.append(len(data[p]["y"]))
    return float(np.average(vals, weights=wts)) if vals else float("nan")


def main():
    data, y, dec, common = pooled()
    n = len(y)
    print(f"pooled DECISION stream: {n:,} commits from {len(data)} projects")
    print("(each project keeps its own online-tuned threshold)")
    print(f"models: {common}\n")

    res = {"n_commits": int(n), "n_projects": len(data),
           "pooling": "decisions (per-project thresholds preserved)",
           "models": {}}
    for m in common:
        r = _from_decisions(y, dec[m])
        r["AUC"] = _rank_weighted(data, m, "AUC")
        r["Popt"] = _rank_weighted(data, m, "Popt")
        r["ACC20"] = _rank_weighted(data, m, "ACC20")
        res["models"][m] = r

    print(f"{'model':16}{'Macro-F1':>10}{'G-Mean':>9}{'AUC*':>8}"
          f"{'P_opt*':>8}{'ACC@20*':>9}")
    for m in common:
        r = res["models"][m]
        print(f"{m:16}{r['Macro_F1']:10.4f}{r['G_Mean']:9.4f}{r['AUC']:8.4f}"
              f"{r['Popt']:8.4f}{r['ACC20']:9.4f}")
    print("\n* rank-based: commit-weighted mean of per-project values")

    dst = OUTP / "tables" / "total_rows.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {dst.relative_to(ROOT)}")

    order = ["KGov", "KGpp"] + [l for l, _ in BASELINES]
    have = [m for m in order if m in common]

    def row(keys):
        cells = []
        for m in have:
            for k in keys:
                v = res["models"][m][k]
                cells.append("--" if not np.isfinite(v) else f"{v:.3f}")
        return r"\midrule \textbf{Total} & " + " & ".join(cells) + r" \\"

    frag = ["% Total = pooled DECISIONS; each project keeps its own tuned",
            "% threshold. Rank-based columns are commit-weighted means.",
            "",
            "% tab:rq1_perf_g50   (Macro-F1 / G-Mean / AUC)",
            row(["Macro_F1", "G_Mean", "AUC"]), "",
            "% tab:rq1_effort_g50 (P_opt / ACC@20)",
            row(["Popt", "ACC20"]), "",
            "% tab:significance   (Macro-F1)",
            row(["Macro_F1"])]
    f2 = OUTP / "tables" / "total_rows.tex"
    f2.write_text("\n".join(frag), encoding="utf-8")
    print(f"wrote {f2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
