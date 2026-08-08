"""
Appendix D: per-project 6x6 = 36 online-stream plots.
For each project: 6 metrics (the 7 headline minus the RQ3 one -> we use all 7 minus
Macro_F1 which headlines RQ3; configurable) x 6 graphs (Core,+AST,+CFG,+DFG,+PDG,+CSTG).
Each plot shows the 5 methods (RN/PPR/LP/DW/KGE) + F (per-graph fusion) as trends.
Dual-window (accurate 150 / smoothed 800), separate images.
Cache-only: final_experiments per-graph per-method traj/traj_smooth + Fusion traj.

Run per project: KGC_PROJECT=<p> python inference/make_appendix_D_streams.py
Out: appendices/D_perproject_layers/streams/<p>/{accurate,smoothed}/D_<metric>_<graph>.pdf
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: F401
from config.project_config import OUT, PROJECT  # noqa: E402

PM = Path(__file__).resolve().parent.parent.parent / "Paper" / "paper_material" / \
    "appendices" / "D_perproject_layers" / "streams" / PROJECT
METRICS = ["Precision", "Recall", "Buggy_F1", "G_Mean", "AUC", "ACC"]  # 6 (Macro-F1 -> RQ3)
YLAB = {"Precision": "Precision", "Recall": "Recall", "Buggy_F1": "Buggy-F1",
        "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Accuracy"}
GRAPHS = [("core", "Core"), ("ast", "Core+AST"), ("cfg", "Core+CFG"),
          ("dfg", "Core+DFG"), ("pdg", "Core+PDG"), ("final", "Core+AST+CSTG")]
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
MCOL = {"RN": "#56B4E9", "PPR": "#E69F00", "LP": "#009E73", "DW": "#0072B2",
        "KGE": "#CC79A7", "F": "#D55E00"}


def main():
    p = OUT / "final_experiments_results.pkl"
    if not p.exists():
        print(f"[{PROJECT}] no final_experiments; skip."); return
    fe = pickle.load(open(p, "rb"))
    n = 0
    for version, tkey in [("accurate", "traj"), ("smoothed", "traj_smooth")]:
        outdir = PM / version; outdir.mkdir(parents=True, exist_ok=True)
        for g, glab in GRAPHS:
            node = fe.get(g, {})
            if not node:
                continue
            for mk in METRICS:
                fig, ax = plt.subplots(figsize=(7, 3.8)); drawn = False
                for m in METHODS:
                    t = node.get(m, {}).get(tkey) or node.get(m, {}).get("traj")
                    if t and mk in t:
                        ax.plot(t["idx"], t[mk], color=MCOL[m], lw=1.2, label=m); drawn = True
                fus = node.get("Fusion", {})
                ft = fus.get(tkey) or fus.get("traj")
                if ft and mk in ft:
                    ax.plot(ft["idx"], ft[mk], color=MCOL["F"], lw=2.4, label="F"); drawn = True
                if not drawn:
                    plt.close(fig); continue
                ax.set_title(f"{PROJECT} --- {glab}", weight="bold", fontsize=9)
                ax.set_xlabel("Commit index"); ax.set_ylabel(YLAB[mk])
                ax.legend(fontsize=7, ncol=3); ax.grid(color="#EEE"); ax.set_axisbelow(True)
                fig.tight_layout()
                fig.savefig(outdir / f"D_{mk}_{g}.pdf", bbox_inches="tight")
                fig.savefig(outdir / f"D_{mk}_{g}.png", bbox_inches="tight")
                plt.close(fig); n += 1
    print(f"[{PROJECT}] wrote {n} Appendix-D stream plots (dual-window).")


if __name__ == "__main__":
    main()
