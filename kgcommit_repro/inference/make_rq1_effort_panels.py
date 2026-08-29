"""
RQ1 effort-aware comparison: Popt and ACC@20, KG-Commit vs. the final baselines.
===============================================================================

The threshold metrics (Macro-F1) say how well a model separates buggy from benign.
The effort-aware pair says something a reviewer cares about more directly: if you
can only inspect a fraction of the changed lines, how many bugs do you actually
catch? This script produces the per-project comparison for both measures under the
FINAL methodology (adaptive INIT + Switch@200).

  Popt    normalised effort-vs-bugs curve area (Kamei et al.); 0.5 = random
  ACC@20  recall of buggy commits after spending 20% of the total la+ld effort

Emitted in the same three formats as the stream/ROC panels:
  overall   KG-Commit (overall F = RN+PPR, switch@200)   vs. baselines
  perproj   KG-Commit (per-project chosen F, switch@200) vs. baselines
  both      BOTH KG-Commit variants as separate bars     vs. baselines

ALIGNMENT (the part that is easy to get wrong)
----------------------------------------------
Effort is joined to the evaluation span BY COMMIT SHA, not by row number. The
fusion's `commit_index` counts rows of the KG stream (the label CSV sorted by
author_date, restricted to commits the graph holds), while the CSV has extra rows
on six projects. `run_effort_eval.py` indexes the FULL CSV with KG-stream
positions, which silently pairs each commit with another commit's churn; this
script restricts the CSV to the KG commits first, so position i means the same
commit in every series.

Cache-only: reads the fusion score pkls, the baseline pkl, DeepJIT's
predictions.csv.gz, and the label CSV. No Neo4j, no re-inference.

Out: <paper_material>/RQ1_performance/
       fig_rq1_effort__<variant>.{pdf,png}        grouped bars, Popt + ACC@20
       fig_rq1_effort_curves__<variant>.{pdf,png} cost-effectiveness curves
       rq1_effort_data.json                       the plotted values

Run: python inference/make_rq1_effort_panels.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401
import effort_metrics as em  # noqa: E402

from make_rq1_panels import (  # noqa: E402
    PROJECTS, DISP, BASELINES, KG_STYLE, OUTP, PM, load_project, _kg_order,
    SWITCH_S,
)


def _effort_for(project, idx):
    """Per-commit inspection effort (la+ld), restricted to the KG stream and
    sliced to the evaluation span -- so effort[i] is the churn of the SAME commit
    that carries score[i]."""
    import os
    os.environ["KGC_PROJECT"] = project
    for m in [m for m in list(sys.modules)
              if m.startswith(("config", "_kgc_paths"))]:
        del sys.modules[m]
    import _kgc_paths  # noqa: F401
    from config.project_config import CSV_PATH
    import pandas as pd
    import pickle

    cache = OUTP / project / "kg_stream_cache.pkl"
    obj = pickle.load(open(cache, "rb"))
    shas = set((obj[0] if isinstance(obj, tuple) else obj).keys())
    csv = pd.read_csv(CSV_PATH).sort_values("author_date").reset_index(drop=True)
    sub = csv[csv["commit_id"].isin(shas)].reset_index(drop=True)
    eff = (sub["la"].fillna(0) + sub["ld"].fillna(0)).to_numpy(float) + 1.0
    return eff[idx]


def collect():
    """project -> series -> {Popt, ACC20} (+ the cost-effectiveness curve)."""
    out = {}
    for p in PROJECTS:
        S = load_project(p)
        if S is None:
            print(f"  skip {p}")
            continue
        idx = S["overall"][0]
        eff = _effort_for(p, idx)
        d = {}
        for key in list(KG_STYLE) + [lab for _, lab, _ in BASELINES]:
            if key not in S:
                continue
            i_k, y_k, p_k = S[key]
            # every series is on the same span, so the same effort vector applies
            e_k = eff if len(i_k) == len(idx) else _effort_for(p, i_k)
            d[key] = {
                "Popt": float(em.popt(y_k, p_k, e_k)),
                "ACC20": float(em.recall_at_effort(y_k, p_k, e_k, 0.20)),
            }
        out[p] = d
        print(f"  {p:<11} " + "  ".join(
            f"{k.split('(')[0].strip()[:9]}={d[k]['Popt']:.3f}/{d[k]['ACC20']:.3f}"
            for k in ("overall", "perproj") if k in d))
    return out


def _variant_series(variant):
    if variant == "overall":
        return [("overall",) + KG_STYLE["overall"]]
    if variant == "perproj":
        return [("perproj",) + KG_STYLE["perproj"]]
    return [("overall",) + KG_STYLE["overall"], ("perproj",) + KG_STYLE["perproj"]]


def panel_bars(data, variant, out_stem):
    """Grouped bars: one subplot per metric, projects on x, series as bars."""
    kg = _variant_series(variant)
    series = [(k, lab, col) for k, lab, col in kg] + \
             [(lab, lab, col) for _, lab, col in BASELINES]
    projs = [p for p in PROJECTS if p in data]

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.4))
    for ax, metric, nice in zip(axes, ("Popt", "ACC20"),
                                ("$P_{opt}$", "ACC@20%LOC")):
        n_s = len(series)
        w = 0.8 / n_s
        x = np.arange(len(projs))
        for k, (key, lab, col) in enumerate(series):
            vals = [data[p].get(key, {}).get(metric, np.nan) for p in projs]
            is_kg = key in ("overall", "perproj")
            ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=lab, color=col,
                   edgecolor="black" if is_kg else "none",
                   linewidth=.7 if is_kg else 0,
                   zorder=3 if is_kg else 2, alpha=1.0 if is_kg else .85)
        # cross-project mean of each series, as a reference line for the leader
        ax.set_xticks(x)
        ax.set_xticklabels([DISP[p] for p in projs], fontsize=9.5)
        ax.set_ylabel(nice, fontsize=11)
        ax.grid(axis="y", alpha=.25, lw=.6, zorder=0)
        ax.set_axisbelow(True)
        vals_all = [v for p in projs for s, _, _ in series
                    for v in [data[p].get(s, {}).get(metric, np.nan)]
                    if np.isfinite(v)]
        if vals_all:
            lo, hi = min(vals_all), max(vals_all)
            ax.set_ylim(max(0, lo - (hi - lo) * .18), min(1.0, hi + (hi - lo) * .08))
        ax.tick_params(labelsize=9)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), frameon=False,
               fontsize=10.5, bbox_to_anchor=(.5, -.004))
    fig.suptitle("RQ1 — effort-aware performance "
                 f"(final protocol: adaptive INIT + switch@{SWITCH_S})",
                 fontsize=13.5, fontweight="bold", y=.985)
    fig.tight_layout(rect=[0, .062 + .014 * (len(l) > 4), 1, .968])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {Path(out_stem).name}.{{pdf,png}}")


def panel_curves(data_series, variant, out_stem):
    """Cost-effectiveness curves: % effort inspected vs % bugs found."""
    kg = _variant_series(variant)
    n = len(PROJECTS)
    ncol, nrow = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    handles, labels = [], []

    for ax, proj in zip(axes, PROJECTS):
        S = data_series.get(proj)
        if not S:
            ax.set_visible(False)
            continue
        idx = S["overall"][0]
        eff = _effort_for(proj, idx)
        plot_set = [(k, lab, col, 2.4) for k, lab, col in kg] + \
                   [(lab, lab, col, 1.1) for _, lab, col in BASELINES]
        for key, lab, col, lw in plot_set:
            if key not in S:
                continue
            _, y_k, p_k = S[key]
            order = em._order(p_k, eff, True)
            x, yv = em._cumulative_curve(y_k, eff, order)
            ln, = ax.plot(x * 100, yv * 100, color=col, lw=lw,
                          zorder=5 if key in ("overall", "perproj") else 3)
            if lab not in labels:
                handles.append(ln); labels.append(lab)
        ax.axvline(20, color="k", ls=":", lw=.9, alpha=.6)
        ax.plot([0, 100], [0, 100], "k--", lw=.7, alpha=.45)
        ax.set_title(DISP[proj], fontsize=11, fontweight="bold")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.grid(alpha=.25, lw=.6)
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    for k, ax in enumerate(axes[:n]):
        if ax.get_visible():
            if k % ncol == 0:
                ax.set_ylabel("% bugs found", fontsize=9)
            if k >= n - ncol:
                ax.set_xlabel("% effort inspected (la+ld)", fontsize=9)

    order = [labels.index(l) for _, l, _ in kg if l in labels] + \
            [labels.index(l) for _, l, _ in BASELINES if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=min(len(order), 4), frameon=False,
               fontsize=10, bbox_to_anchor=(.5, -.005))
    fig.suptitle("RQ1 — cost-effectiveness (dotted line = the 20% effort budget)",
                 fontsize=13, fontweight="bold", y=.997)
    fig.tight_layout(rect=[0, .05 + .012 * (len(order) > 4), 1, .985])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {Path(out_stem).name}.{{pdf,png}}")


def main():
    PM.mkdir(parents=True, exist_ok=True)
    print("loading per-commit scores + effort ...")
    series = {p: load_project(p) for p in PROJECTS}
    series = {p: s for p, s in series.items() if s}
    data = collect()

    for variant in ("overall", "perproj", "both"):
        print(f"\n[{variant}]")
        panel_bars(data, variant, str(PM / f"fig_rq1_effort__{variant}"))
        panel_curves(series, variant, str(PM / f"fig_rq1_effort_curves__{variant}"))

    json.dump(data, open(PM / "rq1_effort_data.json", "w"), indent=2)

    # cross-project means -- the headline read
    print(f"\n{'series':<34}{'Popt':>9}{'ACC@20':>9}")
    for key in list(KG_STYLE) + [lab for _, lab, _ in BASELINES]:
        lab = KG_STYLE[key][0] if key in KG_STYLE else key
        po = np.nanmean([data[p][key]["Popt"] for p in data if key in data[p]])
        ac = np.nanmean([data[p][key]["ACC20"] for p in data if key in data[p]])
        print(f"{lab:<34}{po:>9.3f}{ac:>9.3f}")
    print(f"\nsaved -> {PM}")


if __name__ == "__main__":
    main()
