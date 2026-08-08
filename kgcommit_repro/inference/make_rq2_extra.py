"""
Small RQ2 C2 items:
  - tab_complexity_analytical: authored big-O table (KG-Commit vs baseline families).
  - tab_total_build_time: end-to-end build time per project (sum of timing logs)
    against commit count -> linear-in-N.
  - cost_crossover: cumulative cost over the stream, KG (incremental: bootstrap +
    flat per-commit) vs a recompute-per-commit baseline; from AST timing logs +
    a representative baseline per-commit predict cost.
Cache-only (timing CSVs + label CSV row counts). No Neo4j.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from paper_projects import ACTIVE as PROJECTS

ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = ROOT / "kgcommit_repro" / "logs"
DATA = ROOT / "data" / "apachejit" / "projects"
PM = ROOT / "Paper" / "paper_material" / "RQ2_scalability"


def _ast_times(folder):
    f = LOGS / folder / "ast_timing.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f, encoding="utf-8", errors="ignore")))
    if not rows:
        return None
    tcol = next((c for c in rows[0] if "ms" in c.lower() or "sec" in c.lower() or "time" in c.lower()), None)
    if not tcol:
        return None
    vals = []
    for r in rows:
        try:
            vals.append(float(r[tcol]))
        except (ValueError, TypeError):
            pass
    return np.asarray(vals, float)


def n_commits(folder):
    f = DATA / f"apache_{folder}.csv"
    if not f.exists():
        return None
    return sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1


def complexity_analytical():
    # Columns name the baseline FAMILIES we actually run, so the analytical table and
    # the measured table (tab:crossproject_scalability) describe the same systems.
    L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\caption{RQ2 analytical cost per operation for KG-Commit and the three "
         r"baseline families evaluated in this paper: change-metric learners "
         r"(LR, LApredict, HGB), the token-based JITLine, and the representation-learning "
         r"Deeper. $\delta$ = size of one commit's change; nbhd = the commit's graph "
         r"neighbourhood; $N$ = commits seen so far. The decisive asymmetry is in the "
         r"first two rows: KG-Commit's per-commit work is bounded by the change, and its "
         r"prediction reads state that already exists, whereas every baseline rebuilds a "
         r"representation per commit and refits over a window that grows with $N$. "
         r"Measured counterparts appear in Table~\ref{tab:crossproject_scalability}.}",
         r"\label{tab:complexity_analytical}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Operation & KG-Commit & Change metrics & Token (JITLine) & Repr.\ (Deeper) \\ \midrule",
         r"Per-commit repr.\ update & $O(\delta)$ incremental & $O(1)$ (12 scalars) & $O(|\text{diff}|)$ recomputed & $O(|\text{diff}|)$ recomputed \\",
         r"Per-commit prediction & $O(\text{nbhd})$, sub-ms & $O(1)$ linear/tree & $O(\text{tok})+\text{RF}$ & $O(\text{enc})+\text{LR}$ \\",
         r"Model refit & none for RN/PPR/LP & retrain on $O(N)$ & retrain on $O(N)$ & autoencoder on $O(N)$ \\",
         r"Space & $O(\sum_j \delta^{(j)})$ & $O(N\!\times\!12)$ & $O(\text{vocab}\!\times\!N)$ & $O(\text{model}+N)$ \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (PM / "tab_complexity_analytical.tex").write_text("\n".join(L), encoding="utf-8")


def total_build_time():
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\caption{RQ2 total AST-layer build time vs.\ commit count (linear in stream "
         r"length: each commit is $O(\delta)$).}",
         r"\label{tab:total_build_time}",
         r"\begin{tabular}{lrr}", r"\toprule",
         r"Project & \#Commits & AST build time (s) \\ \midrule"]
    ns, ts = [], []
    for disp, folder in PROJECTS:
        t = _ast_times(folder); nc = n_commits(folder)
        if t is None or nc is None:
            L.append(f"{disp} & {nc or '--'} & -- \\\\"); continue
        # timing units: assume ms if median<100 else s; report seconds total
        tot = float(np.nansum(t))
        tot_s = tot / 1000.0 if np.nanmedian(t) > 5 else tot
        ns.append(nc); ts.append(tot_s)
        L.append(f"{disp} & {nc:,} & {tot_s:,.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / "tab_total_build_time.tex").write_text("\n".join(L), encoding="utf-8")
    # figure: total build time vs commits (should be ~linear)
    if ns:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        ax.scatter(ns, ts, s=45, color="#0072B2")
        for x, y, (disp, _) in zip(ns, ts, [(d, f) for d, f in PROJECTS if n_commits(f) is not None][:len(ns)]):
            ax.annotate(disp, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Commits"); ax.set_ylabel("AST build time (s)")
        ax.set_title("Total build time vs. commits", weight="bold")
        ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(PM / f"total_build_time.{e}", bbox_inches="tight")
        plt.close(fig)


def cost_crossover():
    # representative mid project: kafka
    t = _ast_times("kafka")
    if t is None:
        return
    n = len(t)
    kg_cum = np.cumsum(np.nan_to_num(t))                    # incremental build cost accrues
    # a recompute-per-commit baseline: pays a fixed feature-extraction+predict cost each
    # commit (JITLine tokenise+RF ~ 29 ms model + tokenisation; use a conservative 30 ms).
    per_commit_recompute = 30.0 if np.nanmedian(t) > 5 else 0.030   # match units
    base_cum = np.arange(1, n + 1) * per_commit_recompute
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(n), kg_cum, color="#D55E00", lw=2, label="KG-Commit (incremental)")
    ax.plot(np.arange(n), base_cum, color="#7f4fa0", lw=2, ls="--", label="Recompute-per-commit")
    ax.set_xlabel("Commit index"); ax.set_ylabel("Cumulative cost")
    ax.set_title("Cumulative cost", weight="bold")
    ax.legend(fontsize=8); ax.grid(color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"cost_crossover.{e}", bbox_inches="tight")
    plt.close(fig)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    complexity_analytical(); total_build_time(); cost_crossover()
    print("wrote tab_complexity_analytical, tab_total_build_time (+fig), cost_crossover")


if __name__ == "__main__":
    main()
