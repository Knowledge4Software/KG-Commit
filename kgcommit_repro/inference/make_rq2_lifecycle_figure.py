"""
RQ2 lifecycle figure: what happens to the KG as a project's history streams in.
===============================================================================

One multi-panel figure answering, on a single commit-index axis, every question
a reader has about the graph as it grows:

  (a) SIZE          cumulative nodes and edges per layer -- how big does it get?
  (b) INGEST RATE   per-commit delta edges (ADDS/REMOVES/UPDATES/MOVES) -- is the
                    work per commit growing, or stationary?
  (c) TIME          per-commit ingest wall-clock, rolling median and p95 -- does
                    folding a commit in get slower as the graph gets bigger?
  (d) CHURN         alive vs removed nodes -- the graph is not append-only; how
                    much of it is retired?
  (e) SPACE         estimated resident footprint over the stream.
  (f) LATENCY       prediction latency against graph scale -- the question that
                    matters at deployment: does serving get slower?

The story the figure tells is the paper's central efficiency claim, made
visually: the graph accumulates context indefinitely, while the per-commit cost
of maintaining it and the per-commit cost of reading it both stay flat.

Sources (all cached, no Neo4j):
  outputs/<p>/final_final_run/complexity/growth_arrays.json   per-commit series
  outputs/<p>/final_final_run/complexity/growth.json          summary stats
  outputs/<p>/final_final_run/complexity/kg_profile.json      node/edge census
  outputs/<p>/final_final_run/complexity/prediction_latency.json
  kgcommit_repro/logs/<p>/ast_timing.csv                      per-commit wall_ms

Run:  python inference/make_rq2_lifecycle_figure.py [--project activemq]
Out:  Paper/ResultsDiscussionsDraft/figures/rq2_lifecycle[_<project>].pdf|png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
LOGS = ROOT / "kgcommit_repro" / "logs"
FIGS = ROOT / "Paper" / "ResultsDiscussionsDraft" / "figures"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

# deployed family only: Core+AST+CSTG (CFG/DFG/PDG/SEQ are candidates, not deployed)
LAYERS = [("ast", "AST", "#1D4ED8"), ("cstg", "CSTG", "#15803D")]
ALL_LAYERS = [("ast", "AST", "#1D4ED8"), ("cfg", "CFG", "#B45309"),
              ("dfg", "DFG", "#7C3AED"), ("pdg", "PDG", "#B91C1C"),
              ("cstg", "CSTG", "#15803D")]

# bytes per node / per edge, for the footprint estimate (Neo4j record sizes:
# 15 B node record + 34 B relationship record, plus property overhead). Used
# only to give the axis a physical scale; the shape is what matters.
B_NODE, B_EDGE = 15.0, 34.0


def jload(p, name):
    f = OUTP / p / "final_final_run" / "complexity" / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def timing(p):
    for c in ("ast_timing.csv", "AST_timing.csv"):
        f = LOGS / p / c
        if f.exists():
            t = pd.read_csv(f)
            if "wall_ms" in t:
                return t
    return None


def roll(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=max(5, w // 10))


def build(project):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    GA = jload(project, "growth_arrays.json")
    GS = jload(project, "growth.json")
    PR = jload(project, "kg_profile.json")
    LT = jload(project, "prediction_latency.json")
    TM = timing(project)
    if not (GA and GS and PR and LT):
        print(f"  {project}: missing artifacts -- skipped")
        return None

    n = len(GA["y"])
    x = np.arange(n)
    w = max(50, n // 60)

    fig = plt.figure(figsize=(13.2, 8.6))
    gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.32)
    fig.suptitle(
        f"Life of the knowledge graph as {DISP[project]} streams in "
        f"({n:,} commits)", fontsize=13, y=0.985)

    # ---------------- (a) cumulative size ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for key, lab, c in ALL_LAYERS:
        L = GA["layers"].get(key)
        if not L or "cum_edges" not in L:
            continue
        ax.plot(x, np.asarray(L["cum_edges"]) / 1e6, lw=1.6, color=c,
                label=lab, ls="-" if key in ("ast", "cstg") else ":")
    ax.set_ylabel("cumulative edges (M)")
    ax.set_title("(a) The graph grows without bound", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(alpha=.25, ls=":")

    # ---------------- (b) per-commit ingest work -----------------------
    ax = fig.add_subplot(gs[0, 1])
    tot = np.asarray(GA["layers"]["ast"]["total"], float)
    med = roll(tot, w).median()
    p95 = roll(tot, w).quantile(.95)
    ax.plot(x, med, lw=1.6, color="#1D4ED8", label=f"median (w={w})")
    ax.plot(x, p95, lw=1.0, color="#B45309", ls="--", label="p95")
    z = np.polyfit(x[np.isfinite(med)], med[np.isfinite(med)], 1)
    ax.plot(x, np.polyval(z, x), lw=1.2, color="#B91C1C", ls="-",
            label=f"trend {z[0]:+.3f}/commit")
    ax.set_yscale("log")
    ax.set_ylabel("AST delta edges / commit")
    ratio = np.nanmedian(tot[-n // 5:]) / max(np.nanmedian(tot[:n // 5]), 1e-9)
    ax.set_title(f"(b) work/commit bounded ({ratio:.1f}x)",
                 fontsize=10, loc="left")
    # headroom above the traces, and a horizontal legend pinned to the top so it
    # cannot sit on the p95 curve (which is the highest line on every project)
    ax.set_ylim(top=np.nanmax(p95) * 12)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
              handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
    ax.grid(alpha=.25, ls=":")

    # ---------------- (c) ingest time ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    if TM is not None and len(TM) >= 20:
        v = TM["wall_ms"].values[:n]
        xx = np.arange(len(v))
        ax.plot(xx, roll(v, w).median(), lw=1.6, color="#7C3AED",
                label=f"median (w={w})")
        ax.plot(xx, roll(v, w).quantile(.95), lw=1.0, ls="--",
                color="#B45309", label="p95")
        ax.set_yscale("log")
        ax.set_ylabel("ingest wall-clock (ms)")
        ax.set_ylim(top=np.nanpercentile(v, 99.5) * 12)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
                  handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
        vr = (np.nanmedian(v[-len(v) // 5:])
              / max(np.nanmedian(v[:len(v) // 5]), 1e-9))
        ax.set_title(f"(c) ingest time ({vr:.1f}x)",
                     fontsize=10, loc="left")
    else:
        ax.set_title("(c) ingest time", fontsize=10, loc="left")
    ax.grid(alpha=.25, ls=":")

    # ---------------- (d) alive vs removed -----------------------------
    ax = fig.add_subplot(gs[1, 0])
    L = GA["layers"]["ast"]
    cum_e = np.asarray(L["cum_edges"], float)
    cum_n = np.asarray(L["cum_net_nodes"], float)
    ax.fill_between(x, 0, cum_n / 1e6, color="#1D4ED8", alpha=.75,
                    label="alive (net) nodes")
    ax.fill_between(x, cum_n / 1e6, cum_e / 1e6, color="#B91C1C", alpha=.35,
                    label="retired / delta edges")
    ax.set_ylabel("millions")
    ax.set_xlabel("commit index")
    ax.set_title("(d) the graph is not append-only", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=.25, ls=":")

    # ---------------- (e) footprint ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    tot_n = np.zeros(n)
    tot_e = np.zeros(n)
    for key, _, _ in ALL_LAYERS:
        Lk = GA["layers"].get(key)
        if not Lk:
            continue
        if "cum_net_nodes" in Lk:
            tot_n += np.asarray(Lk["cum_net_nodes"], float)[:n]
        tot_e += np.asarray(Lk["cum_edges"], float)[:n]
    mb = (tot_n * B_NODE + tot_e * B_EDGE) / 1e6
    ax.plot(x, mb, lw=1.8, color="#15803D")
    ax.fill_between(x, 0, mb, color="#15803D", alpha=.18)
    ax.set_ylabel("estimated store (MB)")
    ax.set_xlabel("commit index")
    ax.set_title(f"(e) footprint grows linearly ($\\approx${mb[-1]:,.0f} MB)",
                 fontsize=10, loc="left")
    ax.grid(alpha=.25, ls=":")

    # ---------------- (f) what it costs to READ ------------------------
    ax = fig.add_subplot(gs[1, 2])
    order = ["core", "ast", "cfg", "dfg", "pdg", "final"]
    nn, ll, nm = [], [], []
    for g in order:
        e = LT.get(g)
        if not e:
            continue
        M = e["predict_ms_per_commit"]
        nn.append(e["nnz"]); ll.append(M["RN"]["median"] + M["PPR"]["median"])
        nm.append(e["name"] if "name" in e else g)
    ax.plot(np.array(nn) / 1e3, ll, "o-", color="#1D4ED8", lw=1.6, ms=6)
    # CFG/DFG/PDG land almost on the same point on every project, so fan their
    # labels out vertically instead of letting them overprint each other.
    dys = {"Core": -13, "Core+AST": -13, "Core+CFG": 10,
           "Core+DFG": -15, "Core+PDG": 21, "Core+AST+CSTG": -13}
    for i, (a, b, t) in enumerate(zip(nn, ll, nm)):
        last = (i == len(nn) - 1)
        ax.annotate(t, (a / 1e3, b), textcoords="offset points",
                    xytext=(-7 if last else 6, dys.get(t, 8)),
                    ha="right" if last else "left", fontsize=6.5)
    ax.margins(x=0.12, y=0.22)
    ax.set_xlabel("graph size: nnz (thousands)")
    ax.set_ylabel("predict latency (ms/commit)")
    ax.set_title("(f) reading it stays linear in size", fontsize=10, loc="left")
    ax.grid(alpha=.25, ls=":")

    # ---------------- (g) cross-project stationarity -------------------
    ax = fig.add_subplot(gs[2, :])
    for p in PROJECTS:
        G = jload(p, "growth_arrays.json")
        if not G:
            continue
        t = np.asarray(G["layers"]["ast"]["total"], float)
        m = len(t)
        ww = max(50, m // 60)
        prog = np.linspace(0, 100, m)
        ax.plot(prog, roll(t, ww).median(), lw=1.4,
                alpha=.95 if p == project else .45,
                color="#1D4ED8" if p == project else "#94A3B8",
                label=DISP[p] if p == project else None, zorder=3 if p == project else 1)
    ax.set_yscale("log")
    ax.set_xlabel("progress through the project's history (%)")
    ax.set_ylabel("AST delta edges / commit")
    ax.set_title("(g) per-commit cost stays in a narrow band on all 11 projects",
                 fontsize=10, loc="left")
    ax.grid(alpha=.25, ls=":")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    FIGS.mkdir(parents=True, exist_ok=True)
    stem = f"rq2_lifecycle_{project}"
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="activemq")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    todo = PROJECTS if a.all else [a.project]
    for p in todo:
        s = build(p)
        if s:
            print(f"  wrote figures/{s}.pdf|.png")


if __name__ == "__main__":
    main()
