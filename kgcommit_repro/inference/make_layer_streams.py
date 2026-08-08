"""
Layer/representation online-stream figures (dual window, both settings), with the
common-start alignment fix so every curve begins at the SAME commit index.

Produces, per project:
  A) RQ3 subgraph streams:   trends = Core+AST, Core+CFG, Core+DFG, Core+PDG
  B) RQ3 layer streams:      trends = Core, Core+AST, Core+AST+CSTG(F), F+G
  C) Appendix-C per-metric:  trends = F+G + the five baselines, for the 6 non-RQ1
                             metrics (Precision, Recall, Buggy_F1, AUC, ACC, + one).
(A)/(B) are all KG-representation fusions on the same eval window, so they align
naturally; (C) includes baselines, so curves are clipped to the common start.

Cache-only. Sources: final_experiments per-graph Fusion traj/traj_smooth (A/B),
final_fusion part2 F/F+G traj (B), baseline_extra trajs/trajs_smooth (C).
Out: RQ3_subgraphs/streams[/g50]/... and appendices/C_perproject_baselines/streams[/g50]/...
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _kgc_paths  # noqa: F401
import common_window as cw
from run_final_experiments import metric_traj, TRAJ_WINDOW, TRAJ_WINDOW_SMOOTH

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material"
from paper_projects import ACTIVE as PROJECTS  # active paper set

BASE5 = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
         ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]
APPC_METRICS = ["Precision", "Recall", "Buggy_F1", "AUC", "ACC", "G_Mean"]
YLAB = {"Macro_F1": "Macro-F1", "G_Mean": "G-Mean", "AUC": "AUC", "Precision": "Precision",
        "Recall": "Recall", "Buggy_F1": "Buggy-F1", "ACC": "Accuracy"}
SUB_COL = {"Core+AST": "#1f77b4", "Core+CFG": "#ff7f0e", "Core+DFG": "#2ca02c", "Core+PDG": "#d62728"}
LAY_COL = {"Core": "#9e9e9e", "Core+AST": "#1f77b4", "Core+AST+CSTG (F)": "#2ca02c", "F+G": "#D55E00"}
BCOL = {"F+G": "#D55E00", "LR": "#0072B2", "HGB": "#009E73",
        "LApredict": "#CC79A7", "Deeper": "#E69F00", "JITLine": "#7f4fa0"}


def _fus_traj(fe, g, version):
    n = fe.get(g, {}).get("Fusion")
    if not isinstance(n, dict):
        return None
    return n.get("traj_smooth" if version == "smoothed" else "traj") or n.get("traj")


def _common_start(trajs, mk, ev0):
    starts = []
    for t in trajs:
        if t and mk in t:
            x = np.asarray(t["idx"]); x = x[x >= (ev0 or 0)]
            if len(x):
                starts.append(x[0])
    return max(starts) if starts else (ev0 or 0)


def _plot(series, mk, title, path, ev0, need_align):
    """series = list of (label, traj, color, lw). Aligns to common start if need_align."""
    cs = _common_start([t for _, t, _, _ in series], mk, ev0) if need_align else (ev0 or 0)
    fig, ax = plt.subplots(figsize=(8, 4)); drawn = False
    for lab, t, col, lw in series:
        if not t or mk not in t:
            continue
        x = np.asarray(t["idx"]); y = np.asarray(t[mk]); m = x >= cs
        ax.plot(x[m], y[m], color=col, lw=lw, label=lab); drawn = True
    if not drawn:
        plt.close(fig); return False
    ax.axvline(cs, color="#bbb", ls=":", lw=0.8)
    ax.set_title(title, weight="bold"); ax.set_xlabel("Commit index")
    ax.set_ylabel(YLAB.get(mk, mk)); ax.legend(fontsize=8, ncol=3, loc="best")
    ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); return True


def do_project(disp, folder, version, gap, subdir):
    fe_p = OUTP / folder / "final_experiments_results.pkl"
    ff_p = OUTP / folder / "final_fusion_results.pkl"
    bx_p = OUTP / folder / "baseline_extra_results.pkl"
    if not (fe_p.exists() and ff_p.exists()):
        return 0
    fe = pickle.load(open(fe_p, "rb")); ff = pickle.load(open(ff_p, "rb"))
    ev0 = cw.common_ev0(folder)
    win = 150 if version == "accurate" else 800
    tag = "Setting A" if gap == 0 else f"Setting B (G={gap})"
    n = 0

    # A) subgraph streams (all KG fusions -> naturally aligned; need_align False)
    sub = {("ast", "Core+AST"), ("cfg", "Core+CFG"), ("dfg", "Core+DFG"), ("pdg", "Core+PDG")}
    for mk in ["Macro_F1", "G_Mean"]:
        series = []
        for g, lab in sorted(sub):
            t = _fus_traj(fe, g, version)
            if t:
                series.append((lab, t, SUB_COL[lab], 1.8))
        if series:
            d = PM / "RQ3_subgraphs" / "streams" / (subdir or "")
            d.mkdir(parents=True, exist_ok=True)
            if _plot(series, mk, f"{disp}",
                     d / f"substream_{mk}__{folder}", ev0, need_align=False):
                n += 1

    # B) layer streams: Core, Core+AST, F, F+G (KG only -> aligned)
    for mk in ["Macro_F1", "G_Mean"]:
        series = []
        for g, lab in [("core", "Core"), ("ast", "Core+AST"), ("final", "Core+AST+CSTG (F)")]:
            t = _fus_traj(fe, g, version)
            if t:
                series.append((lab, t, LAY_COL[lab], 1.6))
        fgt = ff["part2"]["F+G"].get("traj_smooth" if version == "smoothed" else "traj") \
            or ff["part2"]["F+G"].get("traj")
        if fgt:
            series.append(("F+G", fgt, LAY_COL["F+G"], 2.6))
        if series:
            d = PM / "RQ3_subgraphs" / "streams" / (subdir or "")
            d.mkdir(parents=True, exist_ok=True)
            if _plot(series, mk, f"{disp}",
                     d / f"layerstream_{mk}__{folder}", ev0, need_align=False):
                n += 1

    # C) Appendix-C: F+G vs baselines, 6 non-RQ1 metrics (includes baselines -> align)
    if bx_p.exists():
        B = pickle.load(open(bx_p, "rb"))
        btr = B.get("trajs_smooth" if version == "smoothed" else "trajs") or B.get("trajs")
        fgt = ff["part2"]["F+G"].get("traj_smooth" if version == "smoothed" else "traj") \
            or ff["part2"]["F+G"].get("traj")
        for mk in APPC_METRICS:
            series = [("KG-Commit F+G", fgt, BCOL["F+G"], 2.6)]
            for key, lab in BASE5:
                t = btr.get(key) if btr else None
                if t:
                    series.append((lab, t, BCOL[lab], 1.4))
            d = PM / "appendices" / "C_perproject_baselines" / "streams" / (subdir or "")
            d.mkdir(parents=True, exist_ok=True)
            if _plot(series, mk, f"{disp}",
                     d / f"appCstream_{mk}__{folder}", ev0, need_align=True):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gap", type=int, default=0)
    args = ap.parse_args(); subdir = "" if args.gap == 0 else f"g{args.gap}"
    for disp, folder in PROJECTS:
        na = do_project(disp, folder, "accurate", args.gap, subdir)
        ns = do_project(disp, folder, "smoothed", args.gap, subdir)
        print(f"  {disp}: {na} accurate + {ns} smoothed figs")
    print(f"wrote layer/subgraph/appendix-C streams (gap={args.gap}, aligned, dual-window)")


if __name__ == "__main__":
    main()
