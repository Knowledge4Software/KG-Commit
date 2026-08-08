"""
D2 (warm-up K) + Appendix H figures, cache-only from param_experiments/*.json.

D2 (main text):
  - k_robustness: KG-Commit F+G Macro-F1 vs K, one line per project (robustness).
  - k_variance: variance of Macro-F1 over K per project (bar) -- stability.
Appendix H (per project): K, and (if present) M / ROLL sweep trend figures.

Source: outputs/<p>/param_experiments/K.json (and M.json, ROLL.json where present).
K.json format: {"<Kval>": {"Macro_F1":..,"G_Mean":..,"ROC_AUC":..}, "_meta":..}
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
CMAP = plt.get_cmap("tab10")


def load_sweep(folder, name):
    f = OUTP / folder / "param_experiments" / f"{name}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    xs, ys = [], []
    for k in sorted([x for x in d if x != "_meta"], key=lambda s: float(s)):
        v = d[k]
        m = v.get("Macro_F1", v.get("F1"))
        if m is not None:
            xs.append(float(k)); ys.append(m)
    return (xs, ys) if xs else None


def d2_krobust():
    out = PM / "discussions" / "D2_warmup_K"; out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4)); variances = {}
    for i, (disp, folder) in enumerate(PROJECTS):
        s = load_sweep(folder, "K")
        if not s:
            continue
        xs, ys = s
        ax.plot(xs, ys, "-o", ms=4, color=CMAP(i % 10), lw=1.4, label=disp)
        variances[disp] = float(np.var(ys))
    ax.set_xlabel("Warm-up $K$"); ax.set_ylabel("Macro-F1 (F+G)")
    ax.set_title("Warm-up $K$", weight="bold")
    ax.legend(fontsize=7, ncol=3); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(out / f"k_robustness.{e}", bbox_inches="tight")
    plt.close(fig)
    # variance bar
    if variances:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        names = list(variances); vals = [variances[n] for n in names]
        ax.bar(names, vals, color="#0072B2")
        ax.set_ylabel("Variance"); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title("Variance over $K$", weight="bold"); fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(out / f"k_variance.{e}", bbox_inches="tight")
        plt.close(fig)
    return len(variances)


def appendixH_figs():
    out = PM / "appendices" / "H_param_sensitivity"; out.mkdir(parents=True, exist_ok=True)
    for name, xlabel in [("K", "warm-up $K$"), ("M", "block size $M$"), ("ROLL", "window $W$")]:
        fig, ax = plt.subplots(figsize=(7, 4.4)); any_ = False
        for i, (disp, folder) in enumerate(PROJECTS):
            s = load_sweep(folder, name)
            if not s:
                continue
            xs, ys = s; ax.plot(xs, ys, "-o", ms=3, color=CMAP(i % 10), lw=1.3, label=disp); any_ = True
        if not any_:
            plt.close(fig); continue
        ax.set_xlabel(xlabel); ax.set_ylabel("Macro-F1 (F+G)")
        ax.set_title(xlabel, weight="bold")
        ax.legend(fontsize=7, ncol=3); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(out / f"robustness_{name}.{e}", bbox_inches="tight")
        plt.close(fig)


def main():
    n = d2_krobust()
    appendixH_figs()
    print(f"wrote D2 (K robustness+variance, {n} projects) + Appendix H trend figures")


if __name__ == "__main__":
    main()
