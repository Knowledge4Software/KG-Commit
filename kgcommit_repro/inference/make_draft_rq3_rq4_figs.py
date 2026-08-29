"""
Per-project RQ3/RQ4 stream figures for the Results draft.
=========================================================

Generates, for EVERY project and for BOTH fusion-selection rules
(per-project chosen F and the overall fixed F = RN+PPR):

  A. substream    subgraph-candidate comparison (Core, +CFG, +DFG, +PDG, +SEQ, +AST)
                  -- which Layer-2 structural subgraph carries the signal
  B. layerstream  layer comparison, full set:
                  Core, Core+AST, Core+AST+CSTG (F), F+G, Switch@200
  C. layerstream3 layer comparison, 3-layer only:
                  Core, Core+AST, Core+AST+CSTG
  D. switchstream RQ4 justification, all on the 3-layer KG:
                  F, F+G, Switch@200

Series A/B/C come from the persisted trajectories; D is rebuilt from the raw
per-commit fusion scores exactly as build_final_final_run.py splices them (F before
commit S, F+G after), so the curve and the reported switch metrics agree.

Out: outputs/<project>/final_final_run/figures/
       fig_substream__<rule>.{pdf,png}
       fig_layerstream__<rule>.{pdf,png}
       fig_layerstream3__<rule>.{pdf,png}
       fig_switchstream__<rule>.{pdf,png}
     plus copies of the Cassandra/Groovy pair into the draft's figures/.

Cache-only. No Neo4j.

Run: python inference/make_draft_rq3_rq4_figs.py
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"
FIGD = DRAFT / "figures"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

SWITCH_S = 200
GAP = 50
STRIDE, WINDOW = 25, 150
MAX_POINTS = 260

SUB = [("V1_none", "Core", "#9e9e9e"),
       ("V2a_cfg", "Core+CFG", "#1f77b4"),
       ("V2b_dfg", "Core+DFG", "#2ca02c"),
       ("V2c_pdg", "Core+PDG", "#9467bd"),
       ("V2d_seq", "Core+SEQ", "#8c564b"),
       ("V3_ast", "Core+AST", "#d62728")]

RULES = {"perproj": ("per-project chosen $F$", "part2", "per_project"),
         "overall": ("overall $F=$ RN+PPR", "part2_overall", "overall")}


# ------------------------------------------------------------- stream helper --
def _resolution(n):
    if n <= MAX_POINTS * STRIDE:
        return STRIDE, WINDOW
    k = int(np.ceil(n / (MAX_POINTS * STRIDE)))
    return STRIDE * k, WINDOW * k


def _macro_f1(y, yhat):
    out = []
    for c in (0, 1):
        tp = np.sum((yhat == c) & (y == c))
        fp = np.sum((yhat == c) & (y != c))
        fn = np.sum((yhat != c) & (y == c))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(out))


def _online_decisions(p, y, init=300, step=150, gap=GAP):
    p = np.asarray(p, float); y = np.asarray(y, int)
    yhat = np.zeros(len(y), int); thr = 0.5
    for i in range(len(y)):
        yhat[i] = int(p[i] >= thr)
        if i + 1 >= init and (i + 1) % step == 0:
            hi = (i + 1) - gap
            if hi >= 2 and len(np.unique(y[:hi])) > 1:
                yy, pp = y[:hi], p[:hi]
                P = int(yy.sum())
                if 0 < P < len(yy):
                    o = np.argsort(-pp); ys = yy[o]
                    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
                    f1 = 2 * (tp / (tp + fp)) * (tp / P) / \
                        ((tp / (tp + fp)) + (tp / P) + 1e-12)
                    thr = float(pp[o][int(np.argmax(f1))])
    return yhat


def trajectory(y, p):
    stride, window = _resolution(len(y))
    yhat = _online_decisions(p, y)
    xs, vs = [], []
    for end in range(window, len(y) + 1, stride):
        sl = slice(end - window, end)
        if len(np.unique(y[sl])) < 2:
            continue
        xs.append(end); vs.append(_macro_f1(y[sl], yhat[sl]))
    return np.array(xs), np.array(vs)


def _switch(raw, s=SWITCH_S):
    pF = np.asarray(raw["scores"]["F"], float)
    pFG = np.asarray(raw["scores"]["F+G"], float)
    q = pFG.copy()
    if s > 0:
        q[:s] = pF[:s]
    return q


# ------------------------------------------------------------------- figures --
def _draw(series, title, ylab, out_stem, vline=None):
    """series: list of (label, x, y, colour, lw)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for lab, x, v, col, lw in series:
        if len(x):
            ax.plot(x, v, label=lab, color=col, lw=lw,
                    zorder=5 if lw > 1.6 else 3)
    if vline is not None:
        ax.axvline(vline, color="k", ls=":", lw=1.0, alpha=.65)
        ax.annotate(f"$S={SWITCH_S}$", xy=(vline, ax.get_ylim()[0]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8.5, alpha=.75)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Commit index", fontsize=9.5)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.grid(alpha=.25, lw=.6)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8.6, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_project(project):
    base = OUTP / project / "final_final_run"
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    SG = pickle.load(open(base / "experiments" / "subgraph_rq_results.pkl", "rb"))
    E = pickle.load(open(base / "experiments"
                         / "final_experiments_results.pkl", "rb"))
    RAW = {"perproj": pickle.load(open(base / "fusion"
                                       / "raw_fusion_scores.pkl", "rb")),
           "overall": pickle.load(open(base / "fusion"
                                       / "raw_fusion_scores_overall.pkl", "rb"))}

    # ---- A. subgraph candidates (identical for both rules: no F involved) ----
    sub = []
    for key, lab, col in SUB:
        node = SG.get(key)
        if not node or "traj7" not in node:
            continue
        t = node["traj7"]
        sub.append((lab, np.asarray(t["idx"]), np.asarray(t["Macro_F1"]),
                    col, 2.2 if key == "V3_ast" else 1.2))

    for rule, (rlab, _, _) in RULES.items():
        if sub:
            _draw(sub, f"{DISP[project]} — Layer-2 subgraph candidates",
                  "Macro-F1 (online, $w{=}150$)",
                  str(figs / f"fig_substream__{rule}"))

    # ---- B/C/D. layer + switch streams, per rule ----------------------------
    for rule, (rlab, p2key, swkey) in RULES.items():
        raw = RAW[rule]
        y = np.asarray(raw["y"], int)
        idx = np.asarray(raw["commit_index"], int)
        x0 = idx[0]

        layers = []
        for lab, g, col, lw in (("Core", "core", "#9e9e9e", 1.3),
                                ("Core+AST", "ast", "#1f77b4", 1.3),
                                ("Core+AST+CSTG ($F$)", "final", "#2ca02c", 1.7)):
            node = E.get(g, {}).get("Fusion")
            if isinstance(node, dict) and "traj" in node:
                t = node["traj"]
                if "Macro_F1" in t:
                    layers.append((lab, np.asarray(t["idx"]),
                                   np.asarray(t["Macro_F1"]), col, lw))

        # F, F+G and the switch, all from the raw per-commit vectors
        xF, vF = trajectory(y, np.asarray(raw["scores"]["F"], float))
        xG, vG = trajectory(y, np.asarray(raw["scores"]["F+G"], float))
        xS, vS = trajectory(y, _switch(raw))

        full = layers + [
            ("$F{+}G$", x0 + xG, vG, "#ff7f0e", 1.7),
            (f"Switch@{SWITCH_S}", x0 + xS, vS, "#d62728", 2.4)]
        _draw(full, f"{DISP[project]} — representation layers ({rlab})",
              "Macro-F1 (online, $w{=}150$)",
              str(figs / f"fig_layerstream__{rule}"), vline=x0 + SWITCH_S)

        if layers:
            _draw(layers, f"{DISP[project]} — three KG layers ({rlab})",
                  "Macro-F1 (online, $w{=}150$)",
                  str(figs / f"fig_layerstream3__{rule}"))

        _draw([("$F$", x0 + xF, vF, "#2ca02c", 1.6),
               ("$F{+}G$", x0 + xG, vG, "#ff7f0e", 1.6),
               (f"Switch@{SWITCH_S}", x0 + xS, vS, "#d62728", 2.4)],
              f"{DISP[project]} — effect of the switch ({rlab})",
              "Macro-F1 (online, $w{=}150$)",
              str(figs / f"fig_switchstream__{rule}"), vline=x0 + SWITCH_S)

    return figs


def main():
    made = {}
    for p in PROJECTS:
        try:
            made[p] = build_project(p)
            print(f"  {p:<11} ok")
        except Exception as e:
            print(f"  {p:<11} FAILED: {type(e).__name__}: {e}")

    # ship the two worked examples into the draft
    FIGD.mkdir(parents=True, exist_ok=True)
    n = 0
    for proj in ("cassandra", "groovy"):
        src = OUTP / proj / "final_final_run" / "figures"
        for stem in ("substream", "layerstream", "layerstream3", "switchstream"):
            for rule in RULES:
                f = src / f"fig_{stem}__{rule}.pdf"
                if f.exists():
                    shutil.copy2(f, FIGD / f"paper_material_RQ3_{stem}_{rule}__{proj}.pdf")
                    n += 1
    print(f"\ncopied {n} example figures -> {FIGD}")


if __name__ == "__main__":
    main()
