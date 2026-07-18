"""
Render per-project parameter-sensitivity tables and figures from the sweeps
produced by run_param_experiments.py.

Reads (per project):
  outputs/<project>/param_experiments/{K,M,ROLL}.json
Writes:
  tables/v4/tab_param_<knob>.{tex,csv}          one table per swept knob
  figures/v4/param/fig_param_<knob>.{png,pdf}   sensitivity curve per knob
  figures/v4/param/fig_param_all.{png,pdf}      combined 3-panel overview

Each sweep varies one online-protocol knob (K=warm-up fraction, M=block size,
ROLL=rolling window) and re-scores the deployed fusion; the report shows how the
headline metrics move with the setting, so the tuned choices can be justified.

Cache-only downstream of run_param_experiments (which itself is cache-only): no
Neo4j. Run:  KGC_PROJECT=activemq python inference/make_param_report.py
"""
import json
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402

PARAM_DIR = OUT / "param_experiments"
FIG = OUT / "figures" / "v4" / "param"
TAB = OUT / "tables" / "v4"
plt.rcParams.update({"savefig.dpi": 200, "font.size": 11, "font.family": "DejaVu Sans"})

# knob file -> (pretty name, x-axis label, the metrics we plot as curves)
KNOBS = {
    "K":    ("warm-up fraction", "warm-up fraction (K)"),
    "M":    ("block size", "block size (M)"),
    "ROLL": ("rolling window", "rolling window (ROLL)"),
}
CURVE_METRICS = ["Macro_F1", "Buggy_F1", "G_Mean", "AUC", "PR_AUC"]
PRETTY = {"Macro_F1": "Macro-F1", "Buggy_F1": "Buggy-F1", "G_Mean": "G-Mean",
          "AUC": "AUC", "PR_AUC": "PR-AUC", "MCC": "MCC",
          "ROC_AUC": "ROC-AUC", "F1": "F1"}


def _load(knob):
    p = PARAM_DIR / f"{knob}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    meta = d.get("_meta", {})
    vals = [k for k in d if k != "_meta"]
    # keep numeric order
    def _key(v):
        try: return float(v)
        except Exception: return v
    vals = sorted(vals, key=_key)
    metrics = meta.get("metric_keys", CURVE_METRICS)
    return dict(knob=meta.get("knob", knob), values=vals, metrics=metrics, data=d)


def table(knob, sweep):
    TAB.mkdir(parents=True, exist_ok=True)
    metrics = sweep["metrics"]
    header = [sweep["knob"]] + [PRETTY.get(m, m) for m in metrics]
    rows = []
    for v in sweep["values"]:
        cell = sweep["data"][v]
        rows.append([v] + [f"{cell.get(m, float('nan')):.3f}" for m in metrics])
    # CSV
    with open(TAB / f"tab_param_{knob}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    # LaTeX (best value per metric column in bold)
    best = {}
    for j, m in enumerate(metrics):
        col = [(float(sweep["data"][v].get(m, float("-inf")))) for v in sweep["values"]]
        best[j] = col.index(max(col))
    cols = "l" + "r" * len(metrics)
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{Sensitivity of {PROJECT} to {KNOBS.get(knob,(knob,''))[0]} "
             f"(deployed fusion; best per column in bold).}}",
             f"\\label{{tab:param-{PROJECT}-{knob}}}",
             f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(h.replace("_", r"\_") for h in header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        cells = [r[0]] + [(f"\\textbf{{{r[j+1]}}}" if best.get(j) == i else r[j+1])
                          for j in range(len(metrics))]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"tab_param_{knob}.tex").write_text("\n".join(lines), encoding="utf-8")


def fig_one(ax, knob, sweep):
    xs = sweep["values"]
    for m in CURVE_METRICS:
        if m not in sweep["metrics"]:
            continue
        ys = [sweep["data"][v].get(m) for v in xs]
        ax.plot(range(len(xs)), ys, marker="o", label=PRETTY.get(m, m))
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
    ax.set_xlabel(KNOBS.get(knob, (knob, knob))[1])
    ax.set_ylabel("score"); ax.grid(color="#EEE"); ax.set_axisbelow(True)
    ax.set_title(f"{KNOBS.get(knob, (knob,''))[0]}")


def main():
    sweeps = {k: _load(k) for k in KNOBS}
    present = {k: v for k, v in sweeps.items() if v}
    if not present:
        print(f"[{PROJECT}] no param_experiments/*.json; run run_param_experiments.py first.")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"[{PROJECT}] rendering param sweeps: {list(present)}")

    # per-knob table + single-panel figure
    for knob, sweep in present.items():
        table(knob, sweep)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        fig_one(ax, knob, sweep)
        ax.legend(fontsize=9, frameon=False, ncol=2)
        fig.suptitle(f"Parameter sensitivity: {KNOBS[knob][0]}  [{PROJECT}]",
                     fontsize=12, weight="bold")
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(FIG / f"fig_param_{knob}.{e}", bbox_inches="tight")
        plt.close(fig)

    # combined overview
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (knob, sweep) in zip(axes, present.items()):
        fig_one(ax, knob, sweep)
    axes[-1].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f"Parameter sensitivity of the deployed fusion  [{PROJECT}]",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"fig_param_all.{e}", bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote tables -> {TAB}/tab_param_*.{{tex,csv}}")
    print(f"  wrote figures -> {FIG}/fig_param_*.{{png,pdf}}")


if __name__ == "__main__":
    main()
