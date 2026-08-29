"""
RQ1 panel figures: online streams + ROC curves, KG-Commit vs. the final baselines.
==================================================================================

Produces the two RQ1 figure families that the paper needs, as SINGLE PANELS over
all 11 projects with ONE shared legend (no per-subplot legend repetition).

Final baseline set (6):  LR, HGB, RF, LApredict, DeepJIT, JITLine-online.
Final KG-Commit methodology: adaptive INIT + Switch@200, in two fusion variants
  * overall  -- the fixed overall fusion F = RN+PPR
  * perproj  -- the per-project chosen fusion F

Three comparison formats are emitted for each family:
  overall   KG-Commit (overall F, switch@200)              vs. baselines
  perproj   KG-Commit (per-project F, switch@200)          vs. baselines
  both      BOTH KG-Commit variants as two separate curves vs. baselines

Everything is read from persisted artifacts -- no Neo4j, no re-inference:
  final_final_run/fusion/raw_fusion_scores{,_overall}.pkl   per-commit F and F+G
  final_final_run/baselines/baseline_extra_results.pkl      per-commit baselines
  DeepJIT_baseline_results/<proj>/M200_G50_.../predictions.csv.gz  (5 seeds)

The switch score vector is rebuilt exactly as build_final_final_run.py does it
(F for the first S scored commits, F+G thereafter); this reproduces the published
switch_results.json metrics to 6 decimals.

Alignment: the fusion span (ev0..N) starts later than the baseline span, so every
trend is intersected onto the fusion span by absolute commit index before any
metric or curve is computed. DeepJIT's `stream_index` is the same absolute index.

Out: Paper/paper_material/RQ1_performance/
       fig_rq1_streams__<variant>.{pdf,png}
       fig_rq1_roc__<variant>.{pdf,png}
       rq1_panel_data.json          (plotted values, for re-rendering)

Run: python inference/make_rq1_panels.py
     python inference/make_rq1_panels.py --metric Macro_F1
"""
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DJ_ROOT = OUTP / "DeepJIT_baseline_results"
PM = ROOT / "Paper" / "paper_material" / "RQ1_performance"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase", "jitline": "JITLine"})

SWITCH_S = 200          # the deployed global switch point
GAP = 50                # verification-latency gap G
STRIDE, WINDOW = 25, 150   # trajectory resolution (protocol setting B)

# ---- baselines: internal key -> display label -------------------------------
BASELINES = [
    ("B_LR",             "LR",              "#1f77b4"),
    ("B_HGB",            "HGB",             "#2ca02c"),
    ("B_RF",             "RF",              "#8c564b"),
    ("B_LAPREDICT",      "LApredict",       "#e377c2"),
    ("DEEPJIT",          "DeepJIT",         "#ff7f0e"),
    ("B_JITLINE_ONLINE", "JITLine-online",  "#9467bd"),
]
KG_STYLE = {
    "overall": ("KG-Commit (RN+PPR, S@200)", "#d62728"),
    "perproj": ("KG-Commit (per-project F, S@200)", "#17becf"),
}


# ------------------------------------------------------------------ loading --
def _switch_scores(raw, s=SWITCH_S):
    """F for the first `s` scored commits, then F+G -- the deployed rule."""
    pF = np.asarray(raw["scores"]["F"], float)
    pFG = np.asarray(raw["scores"]["F+G"], float)
    q = pFG.copy()
    if s > 0:
        q[:s] = pF[:s]
    return q


def _deepjit_scores(project):
    """5-seed mean per-commit probability, keyed by commit SHA.

    Keyed by SHA, not by `stream_index`: DeepJIT's stream_index counts rows of the
    label CSV, while the fusion's index counts rows of the KG stream (the CSV
    restricted to commits the graph holds). Those differ on six projects, so
    index-matching would pair up different commits."""
    pdir = DJ_ROOT / project
    if not pdir.is_dir():
        return None
    frames = []
    for run in sorted(pdir.glob("M200_G50_warmup0.05_ep10_seed*_thrGapF1")):
        f = run / "predictions.csv.gz"
        if f.exists():
            frames.append(pd.read_csv(
                f, usecols=["commit_id", "true_label", "prob_buggy"]))
    if not frames:
        return None
    g = pd.concat(frames, ignore_index=True).groupby("commit_id").agg(
        prob=("prob_buggy", "mean"), y=("true_label", "first"))
    return g["prob"].to_dict(), g["y"].to_dict()


def _kg_order(project):
    """The KG evaluation order as commit SHAs.

    The KG stream is the label CSV sorted by author_date and restricted to the
    commits present in the graph; the fusion's `commit_index` indexes into THAT
    sequence. (Note `kg_stream_cache.pkl`'s key order is BUILD order, not
    evaluation order -- using it directly aligns only ~70% of labels.)
    Returns (shas, csv_row_of_sha) so both the fusion and the CSV-indexed
    baselines can be mapped onto a common commit key."""
    import os
    os.environ["KGC_PROJECT"] = project
    for m in [m for m in list(sys.modules)
              if m.startswith(("config", "_kgc_paths"))]:
        del sys.modules[m]
    import _kgc_paths  # noqa: F401
    from config.project_config import CSV_PATH

    cache = OUTP / project / "kg_stream_cache.pkl"
    if not cache.exists():
        return None, None
    obj = pickle.load(open(cache, "rb"))
    shas = set((obj[0] if isinstance(obj, tuple) else obj).keys())

    csv = pd.read_csv(CSV_PATH).sort_values("author_date").reset_index(drop=True)
    csv_sha = csv["commit_id"].to_numpy()
    sub = csv[csv["commit_id"].isin(shas)].reset_index(drop=True)
    return sub["commit_id"].to_numpy(), csv_sha


def load_project(project):
    """Return {label: (idx, y, prob)} on the COMMON (fusion) evaluation span."""
    base = OUTP / project / "final_final_run"
    fp, fo = base / "fusion" / "raw_fusion_scores.pkl", \
        base / "fusion" / "raw_fusion_scores_overall.pkl"
    bp = base / "baselines" / "baseline_extra_results.pkl"
    if not (fp.exists() and fo.exists() and bp.exists()):
        return None

    Rp = pickle.load(open(fp, "rb"))
    Ro = pickle.load(open(fo, "rb"))
    B = pickle.load(open(bp, "rb"))

    kg_shas, csv_shas = _kg_order(project)
    if kg_shas is None:
        return None

    idx = np.asarray(Rp["commit_index"], int)
    y = np.asarray(Rp["y"], int)
    if idx.max() >= len(kg_shas):
        print(f"      ! fusion index exceeds KG stream -- skipped")
        return None
    fus_sha = kg_shas[idx]                      # the commit key for every scored row

    series = {
        "perproj": (idx, y, _switch_scores(Rp)),
        "overall": (idx, y, _switch_scores(Ro)),
    }

    # Every baseline is joined to the fusion span BY COMMIT SHA. The saved
    # baseline `idx` is a row of the date-sorted label CSV (verified exactly:
    # csv.buggy[idx] == saved y, 1.0000 on all 11 projects), while the fusion
    # index counts rows of the KG stream. Those two sequences differ wherever the
    # graph dropped commits, so joining on the SHA is the only correct pairing.
    for key, label, _ in BASELINES:
        if key == "DEEPJIT":
            dj = _deepjit_scores(project)
            if dj is None:
                continue
            prob_map, y_map = dj
            keep = np.array([s in prob_map for s in fus_sha])
            if keep.sum() < 0.5 * len(idx):
                continue
            pr = np.array([prob_map.get(s, np.nan) for s in fus_sha], float)
            by = np.array([y_map.get(s, -1) for s in fus_sha], int)
        else:
            raw = B.get("raws", {}).get(key)
            if raw is None:
                continue
            bidx = np.asarray(raw["idx"], int)
            bsha = csv_shas[bidx]
            pmap = dict(zip(bsha, np.asarray(raw["pred"], float)))
            ymap = dict(zip(bsha, np.asarray(raw["y"], int)))
            keep = np.array([s in pmap for s in fus_sha])
            if keep.sum() < 0.5 * len(idx):
                continue
            pr = np.array([pmap.get(s, np.nan) for s in fus_sha], float)
            by = np.array([ymap.get(s, -1) for s in fus_sha], int)

        # Guard: a silent mismatch should drop the series, not draw a
        # plausible-looking but wrong curve.
        agree = (y[keep] == by[keep]).mean()
        if agree < 0.98:
            print(f"      ! {label}: label agreement {agree:.3f} -- dropped")
            continue
        series[label] = (idx[keep], y[keep], pr[keep])
    return series


# ------------------------------------------------------- rolling trajectory --
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
    """Prequential hard labels at the leakage-free online-tuned threshold.
    Mirrors online_jit.online_decisions so the streams match the reported metrics."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    yhat = np.zeros(len(y), int)
    thr = 0.5
    for i in range(len(y)):
        yhat[i] = int(p[i] >= thr)
        if i + 1 >= init and (i + 1) % step == 0:
            hi = (i + 1) - gap
            if hi >= 2 and len(np.unique(y[:hi])) > 1:
                yy, pp = y[:hi], p[:hi]
                P = int(yy.sum())
                if 0 < P < len(yy):
                    order = np.argsort(-pp)
                    ys = yy[order]
                    tp = np.cumsum(ys)
                    fp = np.cumsum(1 - ys)
                    prec = tp / (tp + fp)
                    rec = tp / P
                    f1 = 2 * prec * rec / (prec + rec + 1e-12)
                    thr = float(pp[order][int(np.argmax(f1))])
    return yhat


MAX_POINTS = 260     # keep long streams legible (camel is 21.5k commits)


def _resolution(n):
    """Protocol resolution (stride 25 / window 150), widened on long streams.

    At stride 25 a 21k-commit project would draw ~850 points per curve; with eight
    curves overlaid the panel becomes a solid block of ink and the comparison is
    unreadable. Where that happens we scale stride and window by the same factor,
    so the curve is smoothed consistently rather than merely subsampled (which
    would keep the noise and just drop points)."""
    if n <= MAX_POINTS * STRIDE:
        return STRIDE, WINDOW
    k = int(np.ceil(n / (MAX_POINTS * STRIDE)))
    return STRIDE * k, WINDOW * k


def trajectory(y, p, stride=None, window=None):
    """Rolling Macro-F1 over the last `window` scored commits, every `stride`."""
    if stride is None or window is None:
        stride, window = _resolution(len(y))
    yhat = _online_decisions(p, y)
    xs, vs = [], []
    for end in range(window, len(y) + 1, stride):
        sl = slice(end - window, end)
        if len(np.unique(y[sl])) < 2:
            continue
        xs.append(end)
        vs.append(_macro_f1(y[sl], yhat[sl]))
    return np.array(xs), np.array(vs)


def aggregate_curve(data, key, n_bins=20):
    """Cross-project mean trajectory on a normalised-progress axis.

    Projects differ 29x in stream length (748 to 21,511 scored commits), so raw
    commit index cannot pool them. Re-expressing each project's trajectory as a
    fraction of its own stream puts all 11 on one axis; the mean over projects
    then answers "does the ordering hold on average, everywhere in the stream?"
    -- which no single-project panel can show. Shaded band = +/-1 s.d. across
    projects."""
    grid = np.linspace(0.05, 1.0, n_bins)
    curves = []
    for proj, S in data.items():
        if key not in S:
            continue
        _, y, p = S[key]
        xs, vs = trajectory(y, p)
        if len(xs) < 3:
            continue
        frac = xs / len(y)
        curves.append(np.interp(grid, frac, vs))
    if not curves:
        return grid, None, None
    A = np.vstack(curves)
    return grid, A.mean(axis=0), A.std(axis=0)

# ------------------------------------------------------------------- panels --
def _variant_series(variant):
    if variant == "overall":
        return [("overall",) + KG_STYLE["overall"]]
    if variant == "perproj":
        return [("perproj",) + KG_STYLE["perproj"]]
    return [("overall",) + KG_STYLE["overall"], ("perproj",) + KG_STYLE["perproj"]]


def panel_streams(data, variant, out_stem):
    kg = _variant_series(variant)
    n = len(PROJECTS)
    ncol, nrow = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    handles, labels = [], []

    for ax, proj in zip(axes, PROJECTS):
        S = data.get(proj)
        if not S:
            ax.set_visible(False)
            continue
        for key, lab, col in kg:
            if key not in S:
                continue
            idx, y, p = S[key]
            xs, vs = trajectory(y, p)
            if len(xs):
                ln, = ax.plot(idx[0] + xs, vs, color=col, lw=2.4, zorder=5,
                              solid_capstyle="round")
                if lab not in labels:
                    handles.append(ln); labels.append(lab)
        for _, lab, col in BASELINES:
            if lab not in S:
                continue
            idx, y, p = S[lab]
            xs, vs = trajectory(y, p)
            if len(xs):
                ln, = ax.plot(idx[0] + xs, vs, color=col, lw=1.1, alpha=.85)
                if lab not in labels:
                    handles.append(ln); labels.append(lab)
        ax.set_title(DISP[proj], fontsize=11, fontweight="bold")
        ax.grid(alpha=.25, lw=.6)
        ax.tick_params(labelsize=8)

    # 12th slot: the pooled cross-project curve (see aggregate_curve)
    if len(axes) > n:
        ax = axes[n]
        for key, lab, col in kg:
            g, mu, sd = aggregate_curve(data, key)
            if mu is None:
                continue
            ax.plot(g * 100, mu, color=col, lw=2.6, zorder=6)
            ax.fill_between(g * 100, mu - sd, mu + sd, color=col,
                            alpha=.12, lw=0, zorder=2)
        for _, lab, col in BASELINES:
            g, mu, sd = aggregate_curve(data, lab)
            if mu is None:
                continue
            ax.plot(g * 100, mu, color=col, lw=1.2, alpha=.9, zorder=4)
        ax.set_title("All 11 projects (pooled)", fontsize=11,
                     fontweight="bold", color="#444")
        ax.set_xlabel("% through each project's stream", fontsize=9)
        ax.grid(alpha=.25, lw=.6)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_linestyle((0, (4, 3))); sp.set_edgecolor("#888")
    for ax in axes[n + 1:]:
        ax.set_visible(False)
    for k, ax in enumerate(axes[:n]):
        if ax.get_visible():
            if k % ncol == 0:
                ax.set_ylabel("Macro-F1", fontsize=9)
            if k >= n - ncol:
                ax.set_xlabel("Commit index", fontsize=9)

    # The streams grid uses all 12 cells (11 projects + the pooled curve), so its
    # legend stays under the figure -- but as one compact row rather than a block.
    order = [labels.index(l) for _, l, _ in kg if l in labels] +             [labels.index(l) for _, l, _ in BASELINES if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=len(order), frameon=False,
               fontsize=9.5, columnspacing=1.4, handlelength=1.8,
               bbox_to_anchor=(.5, -.004))
    fig.suptitle(f"RQ1 — online prequential Macro-F1 "
                 f"(window {WINDOW}, stride {STRIDE})",
                 fontsize=13, fontweight="bold", y=.997)
    fig.tight_layout(rect=[0, .028, 1, .985])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {Path(out_stem).name}.{{pdf,png}}")


def panel_roc(data, variant, out_stem):
    kg = _variant_series(variant)
    n = len(PROJECTS)
    ncol, nrow = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    handles, labels = [], []
    aucs = {}

    for ax, proj in zip(axes, PROJECTS):
        S = data.get(proj)
        if not S:
            ax.set_visible(False)
            continue
        aucs[proj] = {}
        for key, lab, col in kg:
            if key not in S:
                continue
            _, y, p = S[key]
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            a = roc_auc_score(y, p)
            aucs[proj][lab] = float(a)
            ln, = ax.plot(fpr, tpr, color=col, lw=2.4, zorder=5)
            if lab not in labels:
                handles.append(ln); labels.append(lab)
        for _, lab, col in BASELINES:
            if lab not in S:
                continue
            _, y, p = S[lab]
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            a = roc_auc_score(y, p)
            aucs[proj][lab] = float(a)
            ln, = ax.plot(fpr, tpr, color=col, lw=1.1, alpha=.85)
            if lab not in labels:
                handles.append(ln); labels.append(lab)
        ax.plot([0, 1], [0, 1], "k--", lw=.8, alpha=.6)
        ax.set_title(DISP[proj], fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=.25, lw=.6)
        ax.tick_params(labelsize=8)

    # 12th slot: the legend, not a plot.
    #
    # A pooled ROC over all projects' commits was tried here and removed. Pooling
    # is commit-weighted, so it is dominated by the largest projects and ranks the
    # models differently from the unweighted per-project mean AUC -- two
    # defensible numbers that disagree. That is a real finding, but it belongs in
    # the text, not silently inside a figure whose job is the per-project
    # comparison. Using the free cell for the legend also reclaims the strip that
    # was under the whole panel.
    if len(axes) > n:
        ax = axes[n]
        ax.axis("off")
        order = [labels.index(l) for _, l, _ in kg if l in labels] +                 [labels.index(l) for _, l, _ in BASELINES if l in labels]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="center", frameon=False, fontsize=12,
                  handlelength=2.6, labelspacing=1.0, borderpad=1.0)
    for ax in axes[n + 1:]:
        ax.set_visible(False)
    for k, ax in enumerate(axes[:n]):
        if ax.get_visible():
            if k % ncol == 0:
                ax.set_ylabel("True positive rate", fontsize=9)
            if k >= n - ncol:
                ax.set_xlabel("False positive rate", fontsize=9)

    fig.suptitle("RQ1 — ROC over the full evaluation stream",
                 fontsize=13, fontweight="bold", y=.997)
    fig.tight_layout(rect=[0, 0, 1, .985])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {Path(out_stem).name}.{{pdf,png}}")
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="debug: one project only")
    a = ap.parse_args()

    PM.mkdir(parents=True, exist_ok=True)
    projects = [a.project] if a.project else PROJECTS

    print("loading persisted per-commit scores ...")
    data = {}
    for p in projects:
        S = load_project(p)
        if S is None:
            print(f"  skip {p}: missing artifacts")
            continue
        data[p] = S
        have = [l for _, l, _ in BASELINES if l in S]
        print(f"  {p:<11} n_eval={len(S['overall'][1]):>6}  baselines={len(have)}/6"
              + ("" if len(have) == 6 else f"  MISSING={set(l for _,l,_ in BASELINES)-set(have)}"))

    all_auc = {}
    for variant in ("overall", "perproj", "both"):
        print(f"\n[{variant}]")
        panel_streams(data, variant, str(PM / f"fig_rq1_streams__{variant}"))
        all_auc[variant] = panel_roc(data, variant, str(PM / f"fig_rq1_roc__{variant}"))

    json.dump({"auc": all_auc, "switch_S": SWITCH_S, "gap": GAP,
               "stride": STRIDE, "window": WINDOW,
               "baselines": [l for _, l, _ in BASELINES]},
              open(PM / "rq1_panel_data.json", "w"), indent=2)
    print(f"\nsaved -> {PM}")


if __name__ == "__main__":
    main()
