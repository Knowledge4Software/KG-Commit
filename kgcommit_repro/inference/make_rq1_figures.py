"""
RQ1 figures (cache-only):
  - ROC curve per project: KG-Commit(F+G) + the five baselines + random-guess.
  - Effort-aware comparison figure: Popt / ACC@20 grouped bars, F+G vs baselines.
Both from cached raw per-commit scores (raws in baseline_extra + raw_fusion_scores).
gap only affects the threshold (ROC is threshold-free -> same for both settings), so
ROC is emitted once; effort bars read stored Setting-A values.

Run: KGC_PROJECT=<any> python inference/make_rq1_figures.py
Out: Paper/paper_material/RQ1_performance/{roc__<p>.pdf/png, effort_bars.pdf/png}
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import _kgc_paths  # noqa: F401
import common_window as cw

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ1_performance"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
BASE5 = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
         ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]
COL = {"F+G": "#D55E00", "LR": "#0072B2", "HGB": "#009E73",
       "LApredict": "#CC79A7", "Deeper": "#E69F00", "JITLine": "#7f4fa0"}


def roc_fig(folder, disp):
    bx = OUTP / folder / "baseline_extra_results.pkl"
    rf = OUTP / folder / "raw_fusion_scores.pkl"
    if not (bx.exists() and rf.exists()):
        return False
    B = pickle.load(open(bx, "rb")); F = pickle.load(open(rf, "rb"))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    # F+G
    yv = np.asarray(F["y"], int); pv = np.asarray(F["scores"]["F+G"], float)
    if len(np.unique(yv)) > 1:
        fpr, tpr, _ = roc_curve(yv, pv); a = roc_auc_score(yv, pv)
        ax.plot(fpr, tpr, color=COL["F+G"], lw=2.6, label=f"KG-Commit F+G ({a:.3f})", zorder=5)
    ev0 = cw.common_ev0(folder)                            # common-window clip
    for key, lab in BASE5:
        r = B["raws"].get(key)
        if not r:
            continue
        idx = np.asarray(r["idx"]); yb = np.asarray(r["y"], int); pb = np.asarray(r["pred"], float)
        if ev0 is not None:
            m = idx >= ev0; yb, pb = yb[m], pb[m]           # restrict to [ev0, N)
        if len(np.unique(yb)) > 1:
            fpr, tpr, _ = roc_curve(yb, pb); a = roc_auc_score(yb, pb)
            ax.plot(fpr, tpr, color=COL[lab], lw=1.6, label=f"{lab} ({a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{disp}", weight="bold"); ax.legend(fontsize=7, loc="lower right")
    ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"roc__{folder}.{e}", bbox_inches="tight")
    plt.close(fig); return True


def effort_fig():
    labels = ["F+G"] + [l for _, l in BASE5]
    popt = {l: [] for l in labels}; acc = {l: [] for l in labels}; names = []
    for disp, folder in PROJECTS:
        ef = OUTP / folder / "effort_results.pkl"      # F+G effort (run_effort_eval.py)
        bx = OUTP / folder / "baseline_extra_results.pkl"
        if not (ef.exists() and bx.exists()):
            continue
        E = pickle.load(open(ef, "rb")); B = pickle.load(open(bx, "rb"))
        names.append(disp)
        # F+G effort metrics come from run_effort_eval's "Fusion(F+G)" subset; the
        # final_fusion_results.pkl metrics dict has no Popt/ACC20 keys (was the bug).
        fgm = E.get("models", {}).get("Fusion(F+G)", {})
        popt["F+G"].append(fgm.get("Popt", np.nan)); acc["F+G"].append(fgm.get("ACC20", np.nan))
        for key, lab in BASE5:
            m = B["baselines"].get(key, {})
            popt[lab].append(m.get("Popt", np.nan)); acc[lab].append(m.get("ACC20", np.nan))
    if not names:
        return
    x = np.arange(len(names)); w = 0.14
    for metric, data, fname in [("$P_{opt}$", popt, "effort_popt"), ("ACC@20", acc, "effort_acc20")]:
        fig, ax = plt.subplots(figsize=(max(7, len(names)*1.1), 4))
        for i, l in enumerate(labels):
            ax.bar(x + (i - 2.5) * w, data[l], w, label=l, color=COL[l])
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric); ax.set_title(metric, weight="bold")
        ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.16))
        ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(PM / f"{fname}.{e}", bbox_inches="tight")
        plt.close(fig)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    for disp, folder in PROJECTS:
        ok = roc_fig(folder, disp)
        print(f"  ROC {disp}: {'ok' if ok else '--'}")
    effort_fig()
    print("wrote ROC (per project) + effort bars")


if __name__ == "__main__":
    main()
