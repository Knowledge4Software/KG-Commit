"""
Per-project KG-vs-baseline comparison: 7 online-stream figures (one per metric)
and a headline comparison table.

Five trends are overlaid on every metric's stream figure, all under the SAME
online prequential protocol, the SAME trajectory resolution, and---crucially---the
SAME model family (the 5 graph methods RN/PPR/LP/DW/KGE, stacked), so they are
directly comparable:

  1. Core                        -- final_experiments['core']['Fusion']  (5-method)
  2. Core+AST                    -- final_experiments['ast']['Fusion']   (5-method)
  3. Core+AST+CSTG (F, no G)     -- final_fusion part2 'F'   (deployed graph fusion)
  4. Core+AST+CSTG (F+G)         -- final_fusion part2 'F+G' (deployed model)
  5. Baseline                    -- baseline_results best learned JIT baseline

NOTE: Core/Core+AST come from final_experiments' per-graph 5-method Fusion (added
so all trends are the same family). Projects whose final_experiments predates that
addition lack ['<graph>']['Fusion'] and must re-run run_final_experiments.py.

Reads (per project, all already on disk after run_experiments + run_baselines):
  subgraph_rq_results.pkl, final_fusion_results.pkl, baseline_results.pkl
Writes:
  figures/v4/baseline/fig_stream_<metric>.{png,pdf}   (7 metric stream figures)
  tables/v4/tab_kg_vs_baseline.{tex,csv}              (end-of-stream comparison)

Run:  KGC_PROJECT=activemq python inference/make_kg_vs_baseline.py
"""
import pickle
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402

FIG = OUT / "figures" / "v4" / "baseline"; FIG.mkdir(parents=True, exist_ok=True)
TAB = OUT / "tables" / "v4"; TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"savefig.dpi": 200, "font.size": 11, "font.family": "DejaVu Sans"})

METRICS = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
PRETTY = {"Precision": "Precision", "Recall": "Recall", "Macro_F1": "Macro-F1",
          "Buggy_F1": "Buggy-F1", "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Accuracy"}

# trend -> (colour, linestyle, linewidth, z)
STYLE = {
    "Core":                (("#9e9e9e"), "--", 1.5, 1),
    "Core+AST":            (("#1f77b4"), "-",  1.8, 2),
    "Core+AST+CSTG (F)":   (("#2ca02c"), "-",  1.8, 2),
    "Core+AST+CSTG (F+G)": (("#d62728"), "-",  2.6, 4),
    "Baseline":            (("#7f4fa0"), ":",  2.0, 3),
}
ORDER = ["Core", "Core+AST", "Core+AST+CSTG (F)", "Core+AST+CSTG (F+G)", "Baseline"]


def _load_trajs():
    """Return {trend: traj-dict} where traj-dict has 'idx' + the 7 metrics."""
    trends = {}
    fe = OUT / "final_experiments_results.pkl"
    ff = OUT / "final_fusion_results.pkl"
    bl = OUT / "baseline_results.pkl"

    if fe.exists():
        e = pickle.load(open(fe, "rb"))
        # Core / Core+AST = the per-graph 5-method Fusion (same family as F/F+G)
        if "core" in e and isinstance(e["core"].get("Fusion"), dict) and e["core"]["Fusion"].get("traj"):
            trends["Core"] = e["core"]["Fusion"]["traj"]
        if "ast" in e and isinstance(e["ast"].get("Fusion"), dict) and e["ast"]["Fusion"].get("traj"):
            trends["Core+AST"] = e["ast"]["Fusion"]["traj"]
    if ff.exists():
        f = pickle.load(open(ff, "rb"))
        p2 = f.get("part2", {})
        if "F" in p2 and p2["F"].get("traj"):
            trends["Core+AST+CSTG (F)"] = p2["F"]["traj"]
        if "F+G" in p2 and p2["F+G"].get("traj"):
            trends["Core+AST+CSTG (F+G)"] = p2["F+G"]["traj"]
    if bl.exists():
        b = pickle.load(open(bl, "rb"))
        if b.get("baseline_traj"):
            trends["Baseline"] = b["baseline_traj"]
    return trends


def fig_streams(trends):
    n = 0
    for mk in METRICS:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        drawn = False
        for name in ORDER:
            t = trends.get(name)
            if not t or mk not in t:
                continue
            col, ls, lw, z = STYLE[name]
            ax.plot(t["idx"], t[mk], color=col, ls=ls, lw=lw, label=name, zorder=z)
            drawn = True
        if not drawn:
            plt.close(fig); continue
        ax.set_title(f"{PRETTY[mk]} over the online stream  [{PROJECT}]",
                     fontsize=12, weight="bold")
        ax.set_xlabel("commit index (chronological)")
        ax.set_ylabel(f"rolling {PRETTY[mk]} (window=150, stride=25)")
        ax.grid(color="#ECECEC"); ax.set_axisbelow(True)
        ax.legend(fontsize=9, frameon=False, ncol=2, loc="best")
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(FIG / f"fig_stream_{mk}.{e}", bbox_inches="tight")
        plt.close(fig); n += 1
    return n


def _final_metrics_by_trend():
    """CUMULATIVE whole-eval metric per trend (the stored scalar leaves), for the
    table -- more stable and standard than the last noisy rolling-window point."""
    rows = {}
    fe = OUT / "final_experiments_results.pkl"
    ff = OUT / "final_fusion_results.pkl"
    bl = OUT / "baseline_results.pkl"
    if fe.exists():
        e = pickle.load(open(fe, "rb"))
        if "core" in e and isinstance(e["core"].get("Fusion"), dict):
            rows["Core"] = e["core"]["Fusion"]["metrics"]
        if "ast" in e and isinstance(e["ast"].get("Fusion"), dict):
            rows["Core+AST"] = e["ast"]["Fusion"]["metrics"]
    if ff.exists():
        f = pickle.load(open(ff, "rb")); p2 = f.get("part2", {})
        if "F" in p2 and p2["F"].get("metrics"):
            rows["Core+AST+CSTG (F)"] = p2["F"]["metrics"]
        if "F+G" in p2 and p2["F+G"].get("metrics"):
            rows["Core+AST+CSTG (F+G)"] = p2["F+G"]["metrics"]
    if bl.exists():
        b = pickle.load(open(bl, "rb"))
        best = b.get("best_baseline")
        if best and best in b["baselines"]:
            rows["Baseline"] = b["baselines"][best]
    # keep only the 7 headline metrics, coercing to float
    return {name: {mk: float(v.get(mk, np.nan)) for mk in METRICS}
            for name, v in rows.items()}


def table_compare():
    rows = _final_metrics_by_trend()
    header = ["representation"] + [PRETTY[m] for m in METRICS]
    body = [[name] + [f"{rows[name][m]:.3f}" if not np.isnan(rows[name][m]) else "--"
                      for m in METRICS] for name in ORDER if name in rows]
    # CSV
    with open(TAB / "tab_kg_vs_baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(body)
    # LaTeX
    cols = "l" + "r" * len(METRICS)
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{KG-Commit representations vs.\\ JIT baseline on {PROJECT} "
             f"(end-of-stream online metrics).}}",
             f"\\label{{tab:kgvsbase-{PROJECT}}}",
             f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(h.replace("_", r"\_") for h in header) + r" \\", r"\midrule"]
    for r in body:
        lines.append(" & ".join(str(x) for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_kg_vs_baseline.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    trends = _load_trajs()
    have = [t for t in ORDER if t in trends]
    print(f"[{PROJECT}] KG-vs-baseline trends available: {have}")
    if len(have) < 2:
        print("  not enough trends (need experiments + baselines run first); skipping.")
        return
    nfig = fig_streams(trends)
    table_compare()
    print(f"  wrote {nfig} stream figures -> {FIG}")
    print(f"  wrote table -> {TAB / 'tab_kg_vs_baseline.tex'}")


if __name__ == "__main__":
    main()
