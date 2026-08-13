"""
Figures for the parameter-sensitivity (Discussion) experiments -> docs/figures/v4/param/.

Per parameter (K, M, ROLL, l), two figures:
  fig_param_<p>_lines   metric-vs-parameter line plot (Buggy-F1, AUC, G-Mean,
                        Macro-F1) for the deployed fusion F + the best single method
  fig_param_<p>_heat    heatmap of Buggy-F1 across (method x parameter value)

Plus one synthesis figure:
  fig_param_F_robustness  the deployed fusion's 7 metrics vs each parameter (small
                          multiples) -- shows how stable F is across the setup.

Run:  python scalability/make_param_figures.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

FIG = C.ROOT / "docs" / "figures" / "v4" / "param"
FIG.mkdir(parents=True, exist_ok=True)
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.25})

PARAMS = ["K", "M", "ROLL", "l"]
PLABEL = {"K": "warmup fraction K", "M": "label gap M (commits)",
          "ROLL": "rolling window (commits)", "l": "CSTG refresh l (commits)",
          "KM": "warmup fraction K (gap M=200)"}
METHODS = ["RN", "PPR", "LP", "DW", "KGE", "F"]
KEYMETRICS = ["Buggy_F1", "AUC", "G_Mean", "Macro_F1"]
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]


def _load(p):
    fp = C.OUT_PARAM / f"{p}.json"
    return json.load(open(fp)) if fp.exists() else None


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}")


def _grid(d):
    return [str(v) for v in d["_meta"]["grid"]]


def fig_lines(param, d):
    vals = _grid(d)
    x = [float(v) for v in vals]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for k, met in enumerate(KEYMETRICS):
        yF = [d[v]["F"]["whole"][met] for v in vals]
        ax.plot(x, yF, "o-", color=OKABE[k], label=f"F: {met}")
    ax.set_xlabel(PLABEL[param]); ax.set_ylabel("whole-stream score")
    ax.set_title(f"Deployed fusion $F$ vs {PLABEL[param]}")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, f"fig_param_{param}_lines")


def fig_heat(param, d):
    vals = _grid(d)
    Mrows = METHODS
    Z = np.array([[d[v][m]["whole"]["Buggy_F1"] for v in vals] for m in Mrows])
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(vals), 3.6))
    im = ax.imshow(Z, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals)
    ax.set_yticks(range(len(Mrows))); ax.set_yticklabels(Mrows)
    ax.set_xlabel(PLABEL[param]); ax.set_title("Buggy-F1")
    for i in range(len(Mrows)):
        for j in range(len(vals)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if Z[i, j] < Z.max() * 0.75 else "black")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, f"fig_param_{param}_heat")


def fig_F_robustness(data):
    """Small multiples: F's 7 metrics vs each parameter (one panel per parameter)."""
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 3.4),
                             sharey=True)
    for ax, param in zip(axes, PARAMS):
        d = data.get(param)
        if not d:
            ax.set_visible(False); continue
        vals = _grid(d); x = [float(v) for v in vals]
        for k, met in enumerate(M7):
            y = [d[v]["F"]["whole"][met] for v in vals]
            ax.plot(x, y, "-", color=OKABE[k % len(OKABE)], lw=1.4, label=met)
        ax.set_title(PLABEL[param], fontsize=9)
        ax.set_xlabel(param)
    axes[0].set_ylabel("F score")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("Robustness of the deployed fusion $F$ across setup parameters", y=1.02)
    _save(fig, "fig_param_F_robustness")


def fig_km_compare(d):
    """KM: ours (F, PPR) vs baselines (JIT-LR, Naive) across the warmup sweep at
    fixed gap M=200 -- two panels (Buggy-F1 and AUC)."""
    vals = [str(v) for v in d["_meta"]["grid"]]
    x = [float(v) for v in vals]
    series = [("F", OKABE[0], "-", "o"), ("PPR", OKABE[2], "-", "s"),
              ("JIT_LR", OKABE[4], "--", "^"), ("Naive", OKABE[6], ":", "x")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, met, title in [(axes[0], "Buggy_F1", "Buggy-F1"), (axes[1], "AUC", "AUC")]:
        for name, col, ls, mk in series:
            y = [d[v][name]["whole"][met] for v in vals]
            lab = {"F": "F=RN+PPR (ours)", "PPR": "PPR (ours)",
                   "JIT_LR": "JIT-LR (baseline)", "Naive": "Naive (baseline)"}[name]
            ax.plot(x, y, ls, marker=mk, color=col, label=lab)
        ax.set_xlabel("warmup fraction K"); ax.set_ylabel(title)
        ax.set_title(f"{title} vs warmup at fixed gap M=200")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Ours vs labels-only baselines under real-world constraints "
                 "(gap M=200)", y=1.03)
    _save(fig, "fig_param_KM_compare")


def main():
    print("rendering param figures ->", FIG)
    data = {}
    for pm in PARAMS:
        d = _load(pm)
        if not d:
            print(f"  (skip {pm}: no results)"); continue
        data[pm] = d
        fig_lines(pm, d)
        fig_heat(pm, d)
    if data:
        fig_F_robustness(data)
    dkm = _load("KM")
    if dkm:
        fig_km_compare(dkm)
        fig_lines("KM", dkm)   # F's own metrics vs K at gap 200
    print("done.")


if __name__ == "__main__":
    main()
