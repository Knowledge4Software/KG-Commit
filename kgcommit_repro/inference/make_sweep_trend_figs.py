"""
Trend figures for the M (block) and G (gap) sweeps -- one line per project, x = param
value, y = Macro-F1 (F+G). Parses the already-written D3 sweep .tex tables.
Also builds the App-E best-combo aggregate figure (macro-avg Macro-F1 of each of the
31 fusion combos across projects).
Cache-only (reads .tex tables + fusion pickles). No Neo4j.
"""
import re
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401
from paper_projects import ACTIVE as PROJECTS

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material"
D3 = PM / "discussions" / "D3_gap_M"
APPH = PM / "appendices" / "H_param_sensitivity"
APPE = PM / "appendices" / "E_fusion_combos"
CMAP = plt.get_cmap("tab10")


def parse_sweep(path):
    """Return {param_value: macro_f1} from a d3 sweep .tex table (Macro-F1 = col 4)."""
    if not path.exists():
        return None
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([\d.]+)\s*&\s*[\d.]+\s*&\s*[\d.]+\s*&\s*([\d.]+)", line)
        if m:
            try:
                out[float(m.group(1))] = float(m.group(2))
            except ValueError:
                pass
    return out or None


def trend_fig(kind, outdirs):
    fig, ax = plt.subplots(figsize=(7, 4.4)); any_ = False
    for i, (disp, folder) in enumerate(PROJECTS):
        d = parse_sweep(D3 / f"d3_{kind}sweep__{folder}.tex")
        if not d:
            continue
        xs = sorted(d); ys = [d[x] for x in xs]
        ax.plot(xs, ys, "-o", ms=4, color=CMAP(i % 10), lw=1.4, label=disp); any_ = True
    if not any_:
        plt.close(fig); return
    ax.set_xlabel("Block size $M$" if kind == "M" else "Gap $G$")
    ax.set_ylabel("Macro-F1"); ax.set_title(("Block $M$" if kind == "M" else "Gap $G$"), weight="bold")
    ax.legend(fontsize=7, ncol=3); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for od in outdirs:
        od.mkdir(parents=True, exist_ok=True)
        for e in ("pdf", "png"):
            fig.savefig(od / f"robustness_{kind}.{e}", bbox_inches="tight")
    plt.close(fig)


def best_combo_fig():
    """App-E: macro-avg Macro-F1 of each 31 fusion combo across projects."""
    agg = {}
    for disp, folder in PROJECTS:
        p = OUTP / folder / "final_fusion_results.pkl"
        if not p.exists():
            continue
        part1 = pickle.load(open(p, "rb")).get("part1", {})
        for k, v in part1.items():
            agg.setdefault(k, []).append(v["metrics"]["Macro_F1"])
    if not agg:
        return
    means = {k: float(np.mean(v)) for k, v in agg.items()}
    order = sorted(means, key=means.get)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(order)), [means[k] for k in order], color="#0072B2")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_ylabel("Mean Macro-F1"); ax.set_title("Fusion combinations", weight="bold")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(APPE / f"combo_macroavg.{e}", bbox_inches="tight")
    plt.close(fig)
    best = max(means, key=means.get)
    print(f"  best overall combo: {best} (mean Macro-F1 {means[best]:.3f})")


def main():
    trend_fig("M", [D3, APPH])
    trend_fig("G", [D3, APPH])
    best_combo_fig()
    print("wrote M/G trend figs (D3 + App-H) + App-E best-combo aggregate")


if __name__ == "__main__":
    main()
