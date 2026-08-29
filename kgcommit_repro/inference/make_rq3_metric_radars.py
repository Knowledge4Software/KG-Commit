"""
RQ3 per-METRIC radars (the transpose of the per-project radars).
================================================================

make_rq3_figures.py draws ONE radar PER PROJECT whose axes are the 7 metrics. That
answers "for this project, how does each layer do across metrics?" -- but it cannot
show whether a layer's advantage is CONSISTENT across projects, because every
project lives in its own figure.

This script draws the transpose: ONE radar PER METRIC, whose axes are the projects.
A layer that genuinely carries signal traces a large, roughly convex polygon across
all 11 projects; a layer that wins only on a few projects shows visible dents. Same
underlying numbers, so the two views cannot disagree -- but this one makes
cross-project consistency legible at a glance.

Out: <paper_material>/RQ3_subgraphs/metric_radar__<metric>.{pdf,png}   (7 figures)
     + metric_radar_data.json  (the plotted values, for re-rendering without re-running)

Run: python inference/make_rq3_metric_radars.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from protocol import PROJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ3_subgraphs"
PM.mkdir(parents=True, exist_ok=True)

M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
M7L = ["Prec.", "Rec.", "Macro-F1", "Buggy-F1", "G-Mean", "AUC", "Acc."]

SERIES = [("Core", "core", "#9e9e9e"),
          ("Core+AST", "ast", "#1f77b4"),
          ("Core+AST+CSTG (F)", "final", "#2ca02c")]

DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase", "hdfs": "HDFS"})


def _fus(fe, g):
    n = fe.get(g, {}).get("Fusion")
    return n["metrics"] if isinstance(n, dict) else None


def collect():
    """project -> series-label -> {metric: value}; plus F+G from the fusion pkl."""
    data = {}
    for p in PROJECTS:
        fe = OUTP / p / "final_experiments_results.pkl"
        ff = OUTP / p / "final_fusion_results.pkl"
        if not (fe.exists() and ff.exists()):
            print(f"  skip {p}: missing results")
            continue
        E = pickle.load(open(fe, "rb"))
        F = pickle.load(open(ff, "rb"))
        d = {}
        for lab, g, _ in SERIES:
            m = _fus(E, g)
            if m:
                d[lab] = {k: float(m.get(k, np.nan)) for k in M7}
        if "part2" in F and "F+G" in F["part2"]:
            d["F+G"] = {k: float(F["part2"]["F+G"]["metrics"].get(k, np.nan))
                        for k in M7}
        # second selection rule, when present
        if "part2_overall" in F and "F+G" in F.get("part2_overall", {}):
            d["F+G (overall F)"] = {
                k: float(F["part2_overall"]["F+G"]["metrics"].get(k, np.nan))
                for k in M7}
        data[p] = d
    return data


def radar_for_metric(data, metric, label):
    projs = [p for p in PROJECTS if p in data]
    if len(projs) < 3:
        return False
    labels = ["Core", "Core+AST", "Core+AST+CSTG (F)", "F+G"]
    if any("F+G (overall F)" in data[p] for p in projs):
        labels.append("F+G (overall F)")
    cols = {"Core": "#9e9e9e", "Core+AST": "#1f77b4",
            "Core+AST+CSTG (F)": "#2ca02c", "F+G": "#D55E00",
            "F+G (overall F)": "#7B3294"}

    ang = np.linspace(0, 2 * np.pi, len(projs), endpoint=False)
    ang = np.concatenate([ang, [ang[0]]])
    fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw=dict(polar=True))
    for lab in labels:
        vals = [data[p].get(lab, {}).get(metric, np.nan) for p in projs]
        if all(np.isnan(v) for v in vals):
            continue
        v = np.array(vals + [vals[0]])
        ax.plot(ang, v, lw=2.2 if lab.startswith("F+G") else 1.4,
                label=lab, color=cols.get(lab))
        ax.fill(ang, v, alpha=0.05, color=cols.get(lab))
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([DISP.get(p, p) for p in projs], fontsize=8)
    ax.set_title(f"{label} across projects", weight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.42, 1.12))
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"metric_radar__{metric}.{e}", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    data = collect()
    if not data:
        sys.exit("no project results found")
    json.dump(data, open(PM / "metric_radar_data.json", "w"), indent=2)
    n = 0
    for metric, label in zip(M7, M7L):
        if radar_for_metric(data, metric, label):
            print(f"  wrote metric_radar__{metric}")
            n += 1
    print(f"\n{n} per-metric radars over {len(data)} projects -> {PM}")
    print(f"values persisted -> {PM/'metric_radar_data.json'}")


if __name__ == "__main__":
    main()
