#!/usr/bin/env python3
"""
Discussion D4 -- robustness of KG-Commit to its online hyperparameters.

Cache-only: reads outputs/<p>/param_experiments/{K,M,ROLL}.json and (for the
commit-count weights) outputs/<p>/raw_fusion_scores.csv. No Neo4j, no graph
build, no model refit.

Produces two figure families, each for three metrics (Macro-F1, G-Mean, AUC):

  (1) per-hyperparameter: one figure per knob, one curve per project.
        fig_robust_perproject_<KNOB>_<metric>.{pdf,png}

  (2) overall: one figure per aggregation, one curve per knob, x-axis being the
      knob's grid index normalised to [0,1] so the three knobs share an axis.
        fig_robust_overall_<agg>_<metric>.{pdf,png}
      aggregations: macro   -- unweighted mean over projects
                    micro   -- commit-count-weighted mean over projects
                    total   -- positive(buggy)-count-weighted mean over projects

Every figure is accompanied by the exact CSV it was drawn from, so the numbers
in the paper are traceable without re-running anything.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
DEST = ROOT / "Paper" / "paper_material" / "discussions" / "D4_robustness"

# knob file stem -> (display name, axis label, log-scaled x?)
KNOBS = {
    "K": ("Warm-up fraction $K$", "warm-up fraction of the stream", False),
    "M": ("Refit block $M$", "commits between refits", True),
    "ROLL": ("Rolling window $R$", "rolling-window length (commits)", True),
}

METRICS = {
    "Macro_F1": "Macro-F1",
    "G_Mean": "G-Mean",
    "AUC": "AUC",
    "Buggy_F1": "F1 (buggy class)",
}

AGGS = {
    "macro": "Macro-average (unweighted over projects)",
    "micro": "Micro-average (commit-weighted)",
    "total": "Total (buggy-commit-weighted)",
}

# a project needs more than one grid point to show a trend
MIN_POINTS = 2

# The ROLL sweeps were written by an earlier code path that names ROC_AUC and F1
# where K/M name AUC and Buggy_F1; it never recorded Macro_F1 or G_Mean at all.
# Accept the equivalent key where one exists (both ROLL's "F1" and K/M's
# "Buggy_F1" are the positive/buggy-class F1, so they are the same quantity);
# where no equivalent exists the metric is genuinely absent from the cache and
# the knob is skipped for it rather than substituted with something else.
ALIASES = {
    "AUC": ("AUC", "ROC_AUC"),
    "Buggy_F1": ("Buggy_F1", "F1"),
    "Macro_F1": ("Macro_F1",),
    "G_Mean": ("G_Mean",),
}


def get_metric(point: dict, metric: str):
    """Value for `metric`, or None if absent or NaN.

    A NaN appears where a sweep point was degenerate (e.g. Camel at ROLL=200
    scores no positives, so AUC is undefined). Such a point is dropped rather
    than propagated into an average.
    """
    for key in ALIASES.get(metric, (metric,)):
        if key in point:
            v = point[key]
            if v is None or v != v:  # NaN
                return None
            return v
    return None


def load_knob(folder: str, knob: str):
    """Return {x_value(float): {metric: value}} or None."""
    f = OUTP / folder / "param_experiments" / f"{knob}.json"
    if not f.exists():
        return None
    raw = json.loads(f.read_text())
    pts = {}
    for k, v in raw.items():
        if k == "_meta" or not isinstance(v, dict):
            continue
        try:
            pts[float(k)] = v
        except ValueError:
            continue
    return pts or None


def load_weights():
    """(n_commits, n_buggy) per project folder, from the cached fusion scores."""
    w = {}
    for _disp, folder in PROJECTS:
        f = OUTP / folder / "raw_fusion_scores.csv"
        n = pos = 0
        if f.exists():
            with f.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    n += 1
                    try:
                        pos += int(float(row.get("y", 0)))
                    except (TypeError, ValueError):
                        pass
        w[folder] = (n, pos)
    return w


def collect(knob: str):
    """{folder: {'disp':..., 'pts': {x: {metric: val}}}} for projects with a real sweep."""
    out = {}
    for disp, folder in PROJECTS:
        pts = load_knob(folder, knob)
        if pts and len(pts) >= MIN_POINTS:
            out[folder] = {"disp": disp, "pts": pts}
    return out


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save(fig, stem: str):
    DEST.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(DEST / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------- family (1)
def fig_per_hyperparameter(knob: str, metric: str, data):
    disp_knob, xlabel, logx = KNOBS[knob]
    label = METRICS[metric]

    # nothing to draw if this knob's cache never recorded this metric
    if not any(get_metric(pt, metric) is not None
               for d in data.values() for pt in d["pts"].values()):
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]

    xs_all = sorted({x for d in data.values() for x in d["pts"]})
    rows = []
    for i, (folder, d) in enumerate(sorted(data.items(), key=lambda kv: kv[1]["disp"])):
        xs = sorted(d["pts"])
        ys = [get_metric(d["pts"][x], metric) for x in xs]
        keep = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if len(keep) < MIN_POINTS:
            continue
        kx, ky = zip(*keep)
        ax.plot(kx, ky, marker=markers[i % len(markers)], markersize=4.5,
                linewidth=1.4, color=cmap(i % 20), label=d["disp"], alpha=0.9)
        for x, y in keep:
            rows.append([d["disp"], x, y])

    ax.set_xlabel(f"{disp_knob} -- {xlabel}")
    ax.set_ylabel(label)
    ax.set_title(f"Robustness to {disp_knob}: per-project {label}")
    if logx:
        ax.set_xscale("log")
        # label only the sweep's own values; suppress matplotlib's decade minor ticks
        ax.set_xticks(xs_all)
        ax.set_xticks([], minor=True)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        ax.set_xticks(xs_all)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9, loc="best")
    fig.tight_layout()

    stem = f"fig_robust_perproject_{knob}_{metric}"
    save(fig, stem)
    write_csv(DEST / f"{stem}.csv", ["project", knob, metric], rows)
    return stem


# --------------------------------------------------------------- family (2)
def aggregate(data, metric: str, weights, agg: str):
    """Return [(grid_index_normalised, x_value, aggregated_value, n_projects)]."""
    xs_all = sorted({x for d in data.values() for x in d["pts"]})
    if len(xs_all) < 2:
        return []
    out = []
    for j, x in enumerate(xs_all):
        num = den = 0.0
        n_proj = 0
        for folder, d in data.items():
            v = get_metric(d["pts"].get(x, {}), metric)
            if v is None:
                continue
            n, pos = weights.get(folder, (0, 0))
            if agg == "macro":
                w = 1.0
            elif agg == "micro":
                w = float(n)
            else:  # total
                w = float(pos)
            if w <= 0:
                continue
            num += w * v
            den += w
            n_proj += 1
        if den > 0:
            out.append((j / (len(xs_all) - 1), x, num / den, n_proj))
    return out


def fig_overall(metric: str, agg: str, per_knob, weights):
    label = METRICS[metric]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = {"K": "#1f77b4", "M": "#d62728", "ROLL": "#2ca02c"}
    markers = {"K": "o", "M": "s", "ROLL": "^"}

    rows = []
    n_seen = set()
    for knob, data in per_knob.items():
        series = aggregate(data, metric, weights, agg)
        if len(series) < MIN_POINTS:
            continue
        tx = [s[0] for s in series]
        ty = [s[2] for s in series]
        n_seen.update(s[3] for s in series)
        ax.plot(tx, ty, marker=markers[knob], markersize=5, linewidth=1.8,
                color=colors[knob],
                label=f"{KNOBS[knob][0]}  (n={max(s[3] for s in series)})")
        for t, x, y, n in series:
            rows.append([knob, x, t, y, n])
        for t, x, y, _n in series:
            ax.annotate(f"{x:g}", (t, y), textcoords="offset points",
                        xytext=(0, 5), fontsize=6, color=colors[knob],
                        ha="center")

    if not rows:
        plt.close(fig)
        return None

    span = f"{min(n_seen)}" if len(n_seen) == 1 else f"{min(n_seen)}-{max(n_seen)}"
    ax.set_xlabel("normalised position in the sweep grid\n"
                  "(0 = smallest value, 1 = largest; point labels give the "
                  "actual value)", fontsize=9)
    ax.set_ylabel(f"{label} -- {AGGS[agg].split(' (')[0]}")
    ax.set_title(f"Overall robustness across {span} projects: "
                 f"{label} ({AGGS[agg].split(' (')[0]})")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    stem = f"fig_robust_overall_{agg}_{metric}"
    save(fig, stem)
    write_csv(DEST / f"{stem}.csv",
              ["hyperparameter", "value", "normalised_grid_position",
               f"{metric}_{agg}", "n_projects"], rows)
    return stem


def main():
    weights = load_weights()
    per_knob = {k: collect(k) for k in KNOBS}

    print("Projects contributing a real sweep (>=2 grid points):")
    for knob, data in per_knob.items():
        names = sorted(d["disp"] for d in data.values())
        print(f"  {knob:<4} n={len(names):2d}  {', '.join(names)}")
    missing = [disp for disp, f in PROJECTS
               if any(f not in per_knob[k] for k in KNOBS)]
    if missing:
        print(f"  excluded where the grid is a single point: {', '.join(sorted(set(missing)))}")

    made = []
    skipped = []
    for metric in METRICS:
        for knob, data in per_knob.items():
            if not data:
                continue
            stem = fig_per_hyperparameter(knob, metric, data)
            if stem:
                made.append(stem)
            else:
                skipped.append(f"{knob}/{metric}")
        for agg in AGGS:
            made.append(fig_overall(metric, agg, per_knob, weights))

    if skipped:
        print("\nNot drawn -- the metric was never recorded in that knob's "
              "cached sweep: " + ", ".join(skipped))

    # a single machine-readable provenance record for the whole family
    prov = {
        "generated_from": "outputs/<project>/param_experiments/{K,M,ROLL}.json",
        "weights_from": "outputs/<project>/raw_fusion_scores.csv",
        "requires_neo4j": False,
        "hyperparameters": {k: {"display": v[0], "axis": v[1]} for k, v in KNOBS.items()},
        "metrics": METRICS,
        "aggregations": {
            "macro": "unweighted mean of per-project scores",
            "micro": "mean weighted by each project's evaluated commit count",
            "total": "mean weighted by each project's buggy(positive) commit count",
        },
        "projects_per_knob": {
            k: sorted(d["disp"] for d in data.values()) for k, data in per_knob.items()
        },
        "project_weights": {
            folder: {"n_commits": n, "n_buggy": pos}
            for folder, (n, pos) in weights.items()
        },
        "figures": sorted(made),
        "not_available": sorted(skipped),
        "not_available_reason": (
            "The ROLL sweeps were produced by an earlier code path that recorded "
            "ROC_AUC/PR_AUC/F1/MCC only; Macro-F1 and G-Mean were never computed "
            "for that knob. Re-deriving them would require re-running the ROLL "
            "sweep, which is not a cache-only operation."
        ),
    }
    (DEST / "robustness_provenance.json").write_text(json.dumps(prov, indent=2))

    print(f"\n{len(made)} figures -> {DEST}")


if __name__ == "__main__":
    main()
