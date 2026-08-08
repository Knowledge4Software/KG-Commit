"""
RQ1 online-stream figures (dual window): per project, Macro-F1 and G-Mean over the
chronological stream with trends = KG-Commit(F+G) + the five baselines.
Emits BOTH the accurate (window=150) and smoothed (window=800) versions, for BOTH
Setting A (gap=0, stored trajectories) and Setting B (gap!=0, trajectories recomputed
from cached raw per-commit scores at the gap).

Sources (gap=0): final_fusion_results.pkl part2['F+G']['traj'/'traj_smooth'],
                 baseline_extra_results.pkl trajs / trajs_smooth.
Sources (gap!=0): raw_fusion_scores.pkl (F+G p) + baseline raws {idx,y,pred},
                 recomputed via metric_traj(gap=..) at window 150 and 800.
Out: RQ1_performance/{accurate,smoothed}[/g50]/stream_<metric>__<p>.pdf/png
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
PM = ROOT / "Paper" / "paper_material" / "RQ1_performance"
from paper_projects import ACTIVE as PROJECTS  # active paper set (hdfs/mapreduce dropped)
BASE5 = [("B_LR", "LR"), ("B_HGB", "HGB"), ("B_LAPREDICT", "LApredict"),
         ("B_DEEPER", "Deeper"), ("B_JITLINE", "JITLine")]
COL = {"F+G": "#D55E00", "LR": "#0072B2", "HGB": "#009E73",
       "LApredict": "#CC79A7", "Deeper": "#E69F00", "JITLine": "#7f4fa0"}
METRICS = ["Macro_F1", "G_Mean"]
PP = {"Macro_F1": "Macro-F1", "G_Mean": "G-Mean"}


def _stored_trajs(folder, version):
    """gap=0: return (fg_traj, {baseline: traj}) at the requested window."""
    ff = OUTP / folder / "final_fusion_results.pkl"; bx = OUTP / folder / "baseline_extra_results.pkl"
    if not (ff.exists() and bx.exists()):
        return None, None
    F = pickle.load(open(ff, "rb")); B = pickle.load(open(bx, "rb"))
    tkey = "traj" if version == "accurate" else "traj_smooth"
    btkey = "trajs" if version == "accurate" else "trajs_smooth"
    fg = F["part2"]["F+G"].get(tkey) or F["part2"]["F+G"].get("traj")
    btr = B.get(btkey) or B.get("trajs")
    return fg, btr


def _recomputed_trajs(folder, version, gap):
    """gap!=0: recompute trajectories from raw scores at (window, gap)."""
    rf = OUTP / folder / "raw_fusion_scores.pkl"; bx = OUTP / folder / "baseline_extra_results.pkl"
    if not (rf.exists() and bx.exists()):
        return None, None
    roll = TRAJ_WINDOW if version == "accurate" else TRAJ_WINDOW_SMOOTH
    d = pickle.load(open(rf, "rb")); B = pickle.load(open(bx, "rb"))
    yv = np.asarray(d["y"], int); p = np.asarray(d["scores"]["F+G"], float); ev0 = d["ev0"]
    fg = metric_traj(yv, p, ev0, roll=roll, gap=gap)
    btr = {}
    for key, _ in BASE5:
        r = B.get("raws", {}).get(key)
        if r:
            yb = np.asarray(r["y"], int); pb = np.asarray(r["pred"], float)
            w0 = int(r["idx"][0]) if r.get("idx") else 0
            btr[key] = metric_traj(yb, pb, w0, roll=roll, gap=gap)
    return fg, btr


def make(folder, disp, version, win, gap, subdir):
    fg, btr = (_stored_trajs(folder, version) if gap == 0
               else _recomputed_trajs(folder, version, gap))
    if not fg:
        return False
    outdir = PM / version if not subdir else PM / version / subdir
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    tag = "Setting A" if gap == 0 else f"Setting B (G={gap})"
    ev0 = cw.common_ev0(folder)                            # common-window start
    for mk in METRICS:
        if mk not in fg:
            continue
        # common TRAJECTORY start = the latest first-plotted x among all curves, so
        # every curve (F+G and baselines) begins at exactly the same commit index.
        fg_x = np.asarray(fg["idx"])
        starts = [fg_x[fg_x >= (ev0 or 0)][0]] if len(fg_x[fg_x >= (ev0 or 0)]) else []
        for key, _ in BASE5:
            t = btr.get(key) if btr else None
            if t and mk in t:
                bx_ = np.asarray(t["idx"])
                bx_ = bx_[bx_ >= (ev0 or 0)]
                if len(bx_):
                    starts.append(bx_[0])
        common_start = max(starts) if starts else (ev0 or 0)

        fig, ax = plt.subplots(figsize=(8, 4))
        fy = np.asarray(fg[mk]); fmask = fg_x >= common_start
        ax.plot(fg_x[fmask], fy[fmask], color=COL["F+G"], lw=2.6, label="KG-Commit F+G", zorder=5)
        for key, lab in BASE5:
            t = btr.get(key) if btr else None
            if not (t and mk in t):
                continue
            xi = np.asarray(t["idx"]); yi = np.asarray(t[mk])
            m = xi >= common_start                          # align every curve to common_start
            xi, yi = xi[m], yi[m]
            ax.plot(xi, yi, color=COL[lab], lw=1.4, label=lab)
        ax.axvline(common_start, color="#999", ls=":", lw=1)
        ax.set_title(f"{disp}", weight="bold")
        ax.set_xlabel("Commit index"); ax.set_ylabel(PP[mk])
        ax.legend(fontsize=8, ncol=3, loc="best"); ax.grid(color="#EEE"); ax.set_axisbelow(True)
        fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(outdir / f"stream_{mk}__{folder}.{e}", bbox_inches="tight")
        plt.close(fig); n += 1
    return n > 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gap", type=int, default=0)
    args = ap.parse_args()
    subdir = "" if args.gap == 0 else f"g{args.gap}"
    for disp, folder in PROJECTS:
        a = make(folder, disp, "accurate", 150, args.gap, subdir)
        s = make(folder, disp, "smoothed", 800, args.gap, subdir)
        print(f"  {disp}: accurate={'ok' if a else '--'} smoothed={'ok' if s else '--'}")
    print(f"wrote RQ1 dual-window streams (gap={args.gap})")


if __name__ == "__main__":
    main()
