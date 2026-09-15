#!/usr/bin/env python3
"""
The online-tuned decision threshold over the stream, all 11 projects.
=====================================================================

y  = the online threshold actually in force at each point of the stream
x  = % through each project's own evaluated stream (so projects of very
     different lengths overlay on one axis)
One line per project, no confidence band.

The threshold is the leakage-free operating point: every `STEP` commits it is
re-tuned by `_best_threshold` to maximise buggy-F1 on the labels seen so far
(lagged by the verification gap G). This script replays exactly the update rule
in online_jit.online_decisions, with the same defaults, so the curve is the same
operating point at which the paper's Precision/Recall/F1/G-Mean/Accuracy are
reported -- it is a trace of the real protocol, not a re-derivation.

Cache-only: reads the per-commit score vectors already on disk. No Neo4j, no
refit -- the scores are fixed, only the threshold moves.

Out: Paper/paper_material/discussions/Threshold_trace/
       fig_online_threshold_perproject.{pdf,png}
       fig_online_threshold_perproject.csv     the plotted points
       threshold_trace__<project>.csv          full per-update trace
Run: python kgcommit_repro/inference/make_threshold_trace_fig.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# online_jit's import chain reads the single-project config; this script is
# cross-project and uses only the pure-numpy threshold helper, so a placeholder
# satisfies the import without anything project-specific being read from it.
os.environ.setdefault("KGC_PROJECT", "zookeeper")

import _kgc_paths  # noqa: E402,F401
from online_jit import _best_threshold  # noqa: E402
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
DEST = ROOT / "Paper" / "paper_material" / "discussions" / "Threshold_trace"

# the deployed protocol's defaults, matching online_jit.online_decisions
INIT, STEP, GAP = 300, 150, 0


def threshold_trace(y, p, init=INIT, step=STEP, gap=GAP):
    """Replay online_decisions' threshold updates, recording the value in force.

    Returns (i, thr) for every commit index i: the threshold that would be
    applied to commit i. Starts at the 0.5 default until the first re-tune.
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    thr = 0.5
    out = np.empty(len(y), float)
    for i in range(len(y)):
        out[i] = thr                      # the threshold commit i is judged at
        if i + 1 >= init and (i + 1) % step == 0:
            hi = (i + 1) - gap
            if hi >= 2 and len(np.unique(y[:hi])) > 1:
                thr = _best_threshold(y[:hi], p[:hi])
    return out


def load(folder):
    f = OUTP / folder / "final_final_run" / "fusion" / "raw_fusion_scores.pkl"
    if not f.exists():
        return None
    d = pickle.load(open(f, "rb"))
    sc = d.get("scores", {})
    if "F+G" not in sc:
        return None
    return np.asarray(d["y"], int), np.asarray(sc["F+G"], float)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    series, meta = {}, {}

    for disp, folder in PROJECTS:
        got = load(folder)
        if got is None:
            print(f"  {disp}: no cached scores, skipped")
            continue
        y, p = got
        thr = threshold_trace(y, p)
        pct = (np.arange(len(y)) + 1) / len(y) * 100.0
        series[disp] = (pct, thr)
        meta[disp] = {"n_eval": int(len(y)), "n_buggy": int(y.sum()),
                      "n_updates": int(max(0, (len(y) - INIT) // STEP + 1)),
                      "thr_min": float(thr.min()), "thr_max": float(thr.max()),
                      "thr_final": float(thr[-1]),
                      "source": f"outputs/{folder}/final_final_run/fusion/"
                                "raw_fusion_scores.pkl"}
        # full per-commit trace for this project
        with (DEST / f"threshold_trace__{folder}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["commit_rank", "pct_through_stream", "threshold"])
            for i in range(len(y)):
                w.writerow([i + 1, f"{pct[i]:.4f}", f"{thr[i]:.6f}"])

    if not series:
        raise SystemExit("no projects had cached scores")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    cmap = plt.get_cmap("tab20")
    for i, disp in enumerate(sorted(series)):
        pct, thr = series[disp]
        ax.plot(pct, thr, linewidth=1.6, color=cmap(i % 20), label=disp,
                alpha=0.9, drawstyle="steps-post")

    # the 0.5 default every project starts from, before the first re-tune
    ax.axhline(0.5, color="#666", linestyle="--", linewidth=1.0, alpha=0.6,
               zorder=0)
    ax.annotate("0.5 (initial default)", (0.4, 0.5), xytext=(0, 4),
                textcoords="offset points", fontsize=7, color="#666")

    ax.set_xlabel("% through each project's stream")
    ax.set_ylabel("online decision threshold")
    ax.set_title(f"Online-tuned decision threshold over the stream "
                 f"(all {len(series)} projects)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, linestyle=":")
    # legend outside the axes: 11 curves span the full height, so any in-axes
    # placement hides one of them
    ax.legend(fontsize=8, ncol=1, framealpha=0.95,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(DEST / f"fig_online_threshold_perproject.{ext}",
                    bbox_inches="tight", dpi=200)
    plt.close(fig)

    # the plotted points, one tidy long-format table
    with (DEST / "fig_online_threshold_perproject.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "pct_through_stream", "threshold"])
        for disp in sorted(series):
            pct, thr = series[disp]
            for a, b in zip(pct, thr):
                w.writerow([disp, f"{a:.4f}", f"{b:.6f}"])

    (DEST / "MANIFEST.json").write_text(json.dumps({
        "figure": "fig_online_threshold_perproject.{pdf,png}",
        "y": "online decision threshold in force at each commit",
        "x": "% through each project's own evaluated stream",
        "model": "deployed F+G",
        "protocol": {"INIT": INIT, "STEP": STEP, "GAP": GAP,
                     "rule": "every STEP commits, re-tune to maximise buggy-F1 "
                             "on labels seen so far (lagged by GAP); 0.5 until "
                             "the first re-tune"},
        "replays": "online_jit.online_decisions (same defaults), so the curve is "
                   "the operating point behind the reported threshold metrics",
        "requires_neo4j": False,
        "run": "python kgcommit_repro/inference/make_threshold_trace_fig.py",
        "projects": meta,
    }, indent=2))

    print(f"{len(series)} projects -> {DEST}")
    print(f"{'project':<11}{'n':>7}{'updates':>9}{'min':>8}{'max':>8}{'final':>8}")
    for disp in sorted(meta):
        m = meta[disp]
        print(f"{disp:<11}{m['n_eval']:>7}{m['n_updates']:>9}"
              f"{m['thr_min']:>8.3f}{m['thr_max']:>8.3f}{m['thr_final']:>8.3f}")


if __name__ == "__main__":
    main()
