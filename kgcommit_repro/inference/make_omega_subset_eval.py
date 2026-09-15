#!/usr/bin/env python3
"""
D5 subset evaluation: the motivating example's commits (Omega).
===============================================================

Re-evaluates every method on ONLY the commits satisfying Omega of Eq.(orphan) --
a commit that relocates a file which some untouched file still imports -- and
reports the same metric suite the paper reports on all commits, plus the
effort-aware metrics, plus significance tests.

Methods compared, all on the identical Omega subset per project:
  Fusion(F+G) overall      overall fixed F = RN+PPR, plus CSTG
  Fusion(F+G) per-project  per-project chosen F, plus CSTG
  Switch@200 overall       one-time channel switch, overall F
  Switch@200 per-project   one-time channel switch, per-project F
  Best classical baseline  the per-project strongest of LR/RF/HGB

Cache-only. Every score vector already exists on disk; this script only selects
a subset of commits and recomputes metrics over it. No Neo4j, no refit.

Sources
  outputs/import_handling_check/<p>_orphaned_dependant_cases.csv   Omega membership
  outputs/<p>/raw_method_scores.csv                                index -> commit_id
  outputs/<p>/final_final_run/fusion/raw_fusion_scores{,_overall}.pkl   KG scores
  outputs/<p>/baseline_results.pkl -> baseline_raw                 baseline scores
  data/apachejit/projects/apache_<p>.csv                           la/ld for effort

Out: Paper/paper_material/discussions/D5_omega_subset/
       tables/*.csv  figures/*.{pdf,png}  MANIFEST.json
Run: python kgcommit_repro/inference/make_omega_subset_eval.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("KGC_PROJECT", "zookeeper")

import _kgc_paths  # noqa: E402,F401
from online_jit import final_metrics, online_decisions  # noqa: E402
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

try:
    import effort_metrics as em
except ImportError:
    em = None

ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
IMPD = OUTP / "import_handling_check"
AJIT = ROOT / "data" / "apachejit" / "projects"
DEST = ROOT / "Paper" / "paper_material" / "discussions" / "D5_omega_subset"
TABD, FIGD = DEST / "tables", DEST / "figures"

SWITCH_S = 200

# every baseline is reported as its own column
BASELINE_LABEL = {
    "B_LR": "LR",
    "B_RF": "RF",
    "B_HGB": "HGB",
    "B_LAPREDICT": "LApredict",
    "B_DEEPER": "Deeper",
    "B_JITLINE": "JITLine",
    "B_JITLINE_ONLINE": "JITLine (online)",
}

# KG methods first, then baselines, then the degenerate controls
METHOD_ORDER = [
    "Fusion(F+G) overall", "Fusion(F+G) per-project",
    f"Switch@{SWITCH_S} overall", f"Switch@{SWITCH_S} per-project",
    "LR", "RF", "HGB", "LApredict", "Deeper", "JITLine", "JITLine (online)",
    "All-Buggy", "All-Benign", "Base-rate",
]
MIN_EVAL = 10          # a subset smaller than this cannot support metrics
MIN_POS = 3            # ... nor one with almost no positives

METRICS = ["AUC", "PR_AUC", "MCC", "Macro_F1", "Buggy_F1", "G_Mean",
           "Precision", "Recall", "ACC", "Brier"]
EFFORT = ["Popt", "ACC20"]
ALL_METRICS = METRICS + EFFORT

ALIAS = {"AUC": ("AUC", "ROC_AUC"), "ACC": ("ACC", "Acc"),
         "Buggy_F1": ("Buggy_F1", "F1_online", "F1")}


def get(block, key):
    for k in ALIAS.get(key, (key,)):
        if k in block:
            v = block[k]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return None
            return float(v)
    return None


# --------------------------------------------------------------- data loading
def omega_commits(folder):
    f = IMPD / f"{folder}_orphaned_dependant_cases.csv"
    if not f.exists():
        return set()
    return set(pd.read_csv(f)["commit"].astype(str))


def index_map(folder):
    """commit_index -> commit_id, from the per-method score dump."""
    f = OUTP / folder / "raw_method_scores.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    return dict(zip(d.commit_index.astype(int), d.commit_id.astype(str)))


def effort_map(folder):
    """commit_id -> la+ld (churn), the effort proxy used in the paper."""
    f = AJIT / f"apache_{folder.replace('-', '_')}.csv"
    if not f.exists():
        return {}
    try:
        d = pd.read_csv(f, usecols=["commit_id", "la", "ld"])
    except Exception:
        return {}
    return dict(zip(d.commit_id.astype(str),
                    (d.la.fillna(0) + d.ld.fillna(0)).astype(float)))


def switch_vector(idx, sF, sFG, s=SWITCH_S):
    """F before the S-th evaluated commit, F+G after -- the deployed splice."""
    out = np.asarray(sFG, float).copy()
    k = min(s, len(out))
    out[:k] = np.asarray(sF, float)[:k]
    return out


def load_project(folder):
    """All score vectors for one project, aligned on commit_index."""
    imap = index_map(folder)
    if not imap:
        return None
    base = OUTP / folder / "final_final_run" / "fusion"
    vecs, y_ref, idx_ref = {}, None, None
    for rule, fn in (("overall", "raw_fusion_scores_overall.pkl"),
                     ("per-project", "raw_fusion_scores.pkl")):
        p = base / fn
        if not p.exists():
            continue
        d = pickle.load(open(p, "rb"))
        sc = d.get("scores", {})
        if "F" not in sc or "F+G" not in sc:
            continue
        idx = np.asarray(d["commit_index"], int)
        y = np.asarray(d["y"], int)
        vecs[f"Fusion(F+G) {rule}"] = (idx, y, np.asarray(sc["F+G"], float))
        vecs[f"Switch@{SWITCH_S} {rule}"] = (
            idx, y, switch_vector(idx, sc["F"], sc["F+G"]))
        y_ref, idx_ref = y, idx

    # Every baseline separately. baseline_extra_results.pkl persists a per-commit
    # prediction vector for each one under `raws`, so each gets its own column.
    best_name = None
    ex = OUTP / folder / "baseline_extra_results.pkl"
    if ex.exists():
        d = pickle.load(open(ex, "rb"))
        for name, br in (d.get("raws") or {}).items():
            if not br:
                continue
            vecs[BASELINE_LABEL.get(name, name)] = (
                np.asarray(br["idx"], int), np.asarray(br["y"], int),
                np.asarray(br["pred"], float))

    bl = OUTP / folder / "baseline_results.pkl"
    if bl.exists():
        d = pickle.load(open(bl, "rb"))
        best_name = d.get("best_baseline")
        br = d.get("baseline_raw")
        # fall back to the single persisted vector when the extra run is absent
        if br and not any(k in vecs for k in BASELINE_LABEL.values()):
            vecs[BASELINE_LABEL.get(best_name, str(best_name))] = (
                np.asarray(br["idx"], int), np.asarray(br["y"], int),
                np.asarray(br["pred"], float))
        # the degenerate baselines keep no score vector, but they are defined
        # by construction: constant 1, constant 0, and the running base rate
        if br:
            idx = np.asarray(br["idx"], int)
            yv = np.asarray(br["y"], int)
            if "B_ALL1" in d.get("baselines", {}):
                vecs["All-Buggy"] = (idx, yv, np.ones(len(yv), float))
            if "B_ALL0" in d.get("baselines", {}):
                vecs["All-Benign"] = (idx, yv, np.zeros(len(yv), float))
            if "B_RATE" in d.get("baselines", {}):
                rate = np.full(len(yv), float(yv.mean()))
                vecs["Base-rate"] = (idx, yv, rate)

    if not vecs:
        return None
    return vecs, imap, best_name


def threshold_metrics(y, p, yhat):
    """Operating-point metrics from decisions made on the full stream.

    Computed straight from the 2x2 confusion counts rather than via sklearn:
    the bootstrap calls this tens of thousands of times, and sklearn's
    per-call validation dominates the runtime at these subset sizes.
    """
    y = np.asarray(y, int)
    yhat = np.asarray(yhat, int)
    tp = float(np.sum((yhat == 1) & (y == 1)))
    fp = float(np.sum((yhat == 1) & (y == 0)))
    fn = float(np.sum((yhat == 0) & (y == 1)))
    tn = float(np.sum((yhat == 0) & (y == 0)))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1_pos = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    prec0 = tn / (tn + fn) if tn + fn else 0.0
    rec0 = spec
    f1_neg = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0

    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / den) if den > 0 else 0.0

    return {
        "MCC": mcc,
        "Macro_F1": float((f1_pos + f1_neg) / 2.0),
        "Buggy_F1": float(f1_pos),
        "Precision": float(prec),
        "Recall": float(rec),
        "G_Mean": float(np.sqrt(max(rec, 0.0) * max(spec, 0.0))),
        "ACC": float((tp + tn) / len(y)) if len(y) else 0.0,
    }


def evaluate_subset(y, p, effort=None, yhat=None):
    """The paper's metric suite on one commit subset.

    `yhat` carries the hard decisions made on the FULL stream, restricted to
    this subset. That matters: final_metrics would otherwise re-tune the
    threshold using only the subset, but every Omega subset is far smaller than
    the tuner's INIT=300 warm-up, so the threshold would never leave its 0.5
    default and every commit scoring above 0.5 would be called buggy (MCC=0 on
    ActiveMQ, for instance). The deployed system judges these commits at the
    threshold its full history had produced, so that is what is replayed here.
    Ranking metrics (AUC, PR-AUC, Brier) are threshold-free and unaffected.
    """
    out = {}
    m = final_metrics(y, np.clip(p, 0, 1))
    for k in METRICS:
        out[k] = get(m, k)

    if yhat is not None:
        out.update(threshold_metrics(y, np.clip(p, 0, 1), yhat))
    if effort is not None and em is not None and len(effort) == len(y):
        try:
            out["Popt"] = float(em.popt(y, p, effort))
            out["ACC20"] = float(em.recall_at_effort(y, p, effort, 0.20))
        except Exception:
            out["Popt"] = out["ACC20"] = None
    else:
        out["Popt"] = out["ACC20"] = None
    return out


# --------------------------------------------------------------- significance
def _ranking_metric(y, p, metric):
    """AUC / PR-AUC / Brier on one resample, without the full metric suite."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    p = np.clip(np.asarray(p, float), 0, 1)
    try:
        if metric in ("AUC", "ROC_AUC"):
            return float(roc_auc_score(y, p))
        if metric == "PR_AUC":
            return float(average_precision_score(y, p))
        if metric == "Brier":
            return float(np.mean((p - y) ** 2))
    except Exception:
        return None
    return None


def paired_bootstrap(y, pa, pb, yha=None, yhb=None, metric="MCC",
                     n=1000, seed=0):
    """Paired bootstrap over commits: the CI on metric(A) - metric(B).

    Threshold metrics are computed from the full-stream decisions (yha/yhb) for
    the same reason evaluate_subset does: re-tuning inside a resample of a small
    subset would not reproduce the deployed operating point.
    """
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    diffs = []
    ranking = metric in ("AUC", "PR_AUC", "Brier")
    for _ in range(n):
        s = rng.integers(0, n_obs, n_obs)
        ys = y[s]
        if len(np.unique(ys)) < 2:
            continue
        if ranking or yha is None or yhb is None:
            # only the ranking metric is needed here; final_metrics would also
            # re-run the whole threshold-tuning suite on every resample
            a = _ranking_metric(ys, pa[s], metric)
            b = _ranking_metric(ys, pb[s], metric)
        else:
            a = threshold_metrics(ys, np.clip(pa[s], 0, 1), yha[s]).get(metric)
            b = threshold_metrics(ys, np.clip(pb[s], 0, 1), yhb[s]).get(metric)
        if a is not None and b is not None:
            diffs.append(a - b)
    if len(diffs) < 50:
        return None
    d = np.asarray(diffs)
    return {"mean_diff": float(d.mean()),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "n_boot": int(len(d))}


def wilcoxon_across_projects(per_proj, a, b, metric):
    """Wilcoxon signed-rank over the per-project metric pairs."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    pairs = [(per_proj[p][a][metric], per_proj[p][b][metric])
             for p in per_proj
             if per_proj[p].get(a, {}).get(metric) is not None
             and per_proj[p].get(b, {}).get(metric) is not None]
    if len(pairs) < 5:
        return None
    x = np.array([q[0] for q in pairs])
    z = np.array([q[1] for q in pairs])
    if np.allclose(x, z):
        return None
    try:
        stat, p = wilcoxon(x, z)
    except Exception:
        return None
    return {"n_projects": len(pairs), "statistic": float(stat), "p": float(p),
            "median_diff": float(np.median(x - z)),
            "wins": int((x > z).sum()), "losses": int((x < z).sum()),
            "ties": int((x == z).sum())}


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def in_vs_out(folder, imap, om, vecs):
    """The deployed model on Omega commits vs. all its OTHER evaluated commits.

    Same model, same protocol, same project -- only the commit subset differs.
    This is the diagnostic that matters for D5: the graph contains the edges
    that characterise these commits but no deployed method reads them, so a
    negative delta says the model does its worst work on exactly the class the
    layer was built for.
    """
    key = "Fusion(F+G) per-project"
    if key not in vecs:
        key = next((k for k in vecs if k.startswith("Fusion")), None)
    if key is None:
        return None
    idx, y, p = vecs[key]
    mask = np.array([imap.get(int(i), "") in om for i in idx])
    if mask.sum() < MIN_EVAL or (~mask).sum() < MIN_EVAL:
        return None
    if len(np.unique(y[mask])) < 2 or len(np.unique(y[~mask])) < 2:
        return None
    mi = final_metrics(y[mask], np.clip(p[mask], 0, 1))
    mo = final_metrics(y[~mask], np.clip(p[~mask], 0, 1))
    out = {"n_omega": int(mask.sum()), "n_rest": int((~mask).sum()),
           "n_omega_buggy": int(y[mask].sum())}
    for met in ("Macro_F1", "MCC", "AUC"):
        a, b = get(mi, met), get(mo, met)
        out[met] = (a, b, None if (a is None or b is None) else a - b)
    return out


def paired_wilcoxon(pairs):
    """Wilcoxon signed-rank on (on_omega, on_rest) pairs across projects."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    a = np.array([q[0] for q in pairs], float)
    b = np.array([q[1] for q in pairs], float)
    if len(a) < 5 or np.allclose(a, b):
        return None
    try:
        stat, p = wilcoxon(a, b)
    except Exception:
        return None
    return {"n": len(a), "statistic": float(stat), "p": float(p),
            "mean_delta": float((a - b).mean()),
            "median_delta": float(np.median(a - b)),
            "n_negative": int((a < b).sum())}


# --------------------------------------------------------------- aggregation
def add_aggregates(per_proj, methods, metrics, weights):
    """Macro / micro / total rows, computed per (method, metric)."""
    aggs = {}
    for name, mode in (("Average (macro)", "macro"),
                       ("Average (micro)", "micro"),
                       ("Average (total)", "total")):
        node = {}
        for meth in methods:
            row = {}
            for met in metrics:
                num = den = 0.0
                for proj, d in per_proj.items():
                    v = d.get(meth, {}).get(met)
                    if v is None:
                        continue
                    n, pos = weights.get(proj, (0, 0))
                    w = 1.0 if mode == "macro" else (
                        float(n) if mode == "micro" else float(pos))
                    if w <= 0:
                        continue
                    num += w * v
                    den += w
                row[met] = num / den if den else None
            node[meth] = row
        aggs[name] = node
    return aggs


AGG_NAMES = ["Average (macro)", "Average (micro)", "Average (total)"]


def collect():
    """Per-project metrics on the Omega subset, plus the kept score vectors."""
    per_proj, weights, coverage, subsets, excluded, in_out = {}, {}, [], {}, [], {}
    best_baseline = {}

    for disp, folder in PROJECTS:
        loaded = load_project(folder)
        om = omega_commits(folder)
        if loaded is None or not om:
            excluded.append((disp, "no cached scores or no Omega cases"))
            continue
        vecs, imap, best_name = loaded
        if best_name:
            best_baseline[disp] = best_name
        emap = effort_map(folder)

        ref = next(iter(vecs.values()))
        idx_ref, y_ref = ref[0], ref[1]
        mask = np.array([imap.get(int(i), "") in om for i in idx_ref])
        n_sub = int(mask.sum())
        n_pos = int(y_ref[mask].sum()) if n_sub else 0
        coverage.append([disp, len(om), int(len(idx_ref)), n_sub, n_pos,
                         f"{n_pos / n_sub:.4f}" if n_sub else "",
                         f"{y_ref.mean():.4f}"])

        if n_sub < MIN_EVAL or n_pos < MIN_POS or len(np.unique(y_ref[mask])) < 2:
            excluded.append((disp, f"Omega subset too small to evaluate "
                                   f"(n={n_sub}, buggy={n_pos})"))
            continue

        node, keep = {}, {}
        for meth, (idx, y, p) in vecs.items():
            m = np.array([imap.get(int(i), "") in om for i in idx])
            if m.sum() < MIN_EVAL or len(np.unique(y[m])) < 2:
                continue
            # decisions the deployed system actually made, on the full stream
            yhat_full = online_decisions(np.clip(p, 0, 1), y)
            eff = np.array([emap.get(imap.get(int(i), ""), 0.0) for i in idx[m]])
            node[meth] = evaluate_subset(y[m], p[m], eff if emap else None,
                                         yhat=yhat_full[m])
            keep[meth] = (y[m], p[m], yhat_full[m])
        if node:
            per_proj[disp] = node
            subsets[disp] = keep
            weights[disp] = (n_sub, n_pos)
            ivo = in_vs_out(folder, imap, om, vecs)
            if ivo:
                in_out[disp] = ivo

    return per_proj, weights, coverage, subsets, excluded, in_out, best_baseline


def bar_figure(per_proj, aggs, methods, met, projs):
    order = projs + AGG_NAMES
    x = np.arange(len(order), dtype=float)
    for i in range(len(projs), len(order)):
        x[i] += 0.6
    w = 0.86 / max(len(methods), 1)
    cmap = plt.get_cmap("tab20" if len(methods) > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(max(10.0, 1.5 * len(order) + 3.0), 4.6))
    drawn = False
    for j, m in enumerate(methods):
        vals = []
        for o in order:
            src = per_proj.get(o) or aggs.get(o, {})
            vals.append(src.get(m, {}).get(met))
        if all(v is None for v in vals):
            continue
        drawn = True
        ax.bar([x[i] + (j - (len(methods) - 1) / 2) * w for i in range(len(order))],
               [0 if v is None else v for v in vals], w, label=m,
               color=cmap(j % cmap.N), edgecolor="white", linewidth=0.3)
    if not drawn:
        plt.close(fig)
        return None

    ax.axhline(0, color="#444", linewidth=0.9)
    if projs:
        ax.axvline((x[len(projs) - 1] + x[len(projs)]) / 2, color="#999",
                   linestyle=":", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel(met)
    ax.set_title(f"{met} on the motivating example's commits "
                 f"($\\Omega$ subset, {len(projs)} projects)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=7.5, ncol=1, framealpha=0.95,
              loc="center left", bbox_to_anchor=(1.005, 0.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"fig_omega_{met}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return f"fig_omega_{met}"


def lift_figure(coverage, projs):
    """The D5 claim itself: Omega commits are buggy far above the base rate."""
    sub = {c[0]: c[5] for c in coverage}
    allr = {c[0]: c[6] for c in coverage}
    keep = [p for p in projs if sub.get(p) not in (None, "")]
    if not keep:
        return None
    xx = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(max(7.6, 0.95 * len(keep) + 2.4), 4.1))
    ax.bar(xx - 0.2, [float(allr[p]) for p in keep], 0.4,
           label="all evaluated commits", color="#9e9e9e")
    ax.bar(xx + 0.2, [float(sub[p]) for p in keep], 0.4,
           label=r"$\Omega$ commits", color="#d62728")
    ax.set_xticks(xx)
    ax.set_xticklabels(keep, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("bug-inducing rate")
    ax.set_title(r"Bug-inducing rate: $\Omega$ commits vs. all evaluated commits",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"fig_omega_lift.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return "fig_omega_lift"


def main():
    TABD.mkdir(parents=True, exist_ok=True)
    FIGD.mkdir(parents=True, exist_ok=True)

    (per_proj, weights, coverage, subsets, excluded,
     in_out, best_baseline) = collect()
    if not per_proj:
        raise SystemExit("no project had an evaluable Omega subset")

    found = {m for d in per_proj.values() for m in d}
    methods = [m for m in METHOD_ORDER if m in found] + \
              sorted(found - set(METHOD_ORDER))
    projs = sorted(per_proj)
    aggs = add_aggregates(per_proj, methods, ALL_METRICS, weights)

    write_csv(TABD / "omega_coverage.csv",
              ["project", "omega_commits_total", "n_eval_all", "n_eval_omega",
               "n_buggy_omega", "buggy_rate_omega", "buggy_rate_all"], coverage)

    # ---- the deployed model on Omega commits vs. its own other commits -----
    ivo_stats = {}
    if in_out:
        order = sorted(in_out, key=lambda p: (in_out[p]["Macro_F1"][2]
                                              if in_out[p]["Macro_F1"][2] is not None
                                              else 0.0))
        rows = []
        for p in order:
            d = in_out[p]
            cells = [p, d["n_omega"], d["n_omega_buggy"], d["n_rest"]]
            for met in ("Macro_F1", "MCC", "AUC"):
                a, b, dd = d[met]
                cells += ["" if a is None else f"{a:.4f}",
                          "" if b is None else f"{b:.4f}",
                          "" if dd is None else f"{dd:.4f}"]
            rows.append(cells)
        hdr = ["project", "n_omega", "n_omega_buggy", "n_rest"]
        for met in ("Macro_F1", "MCC", "AUC"):
            hdr += [f"{met}_on_omega", f"{met}_on_rest", f"delta_{met}"]
        write_csv(TABD / "omega_in_vs_out.csv", hdr, rows)

        for met in ("Macro_F1", "MCC", "AUC"):
            pairs = [(in_out[p][met][0], in_out[p][met][1]) for p in in_out
                     if in_out[p][met][0] is not None
                     and in_out[p][met][1] is not None]
            r = paired_wilcoxon(pairs)
            if r:
                ivo_stats[met] = r
        if ivo_stats:
            write_csv(TABD / "omega_in_vs_out_significance.csv",
                      ["metric", "n_projects", "mean_delta", "median_delta",
                       "n_negative", "statistic", "p", "significant_05"],
                      [[m, r["n"], f"{r['mean_delta']:.4f}",
                        f"{r['median_delta']:.4f}", r["n_negative"],
                        f"{r['statistic']:.3f}", f"{r['p']:.4f}",
                        "yes" if r["p"] < 0.05 else "no"]
                       for m, r in ivo_stats.items()])

    # one table per metric: projects x methods, with the three aggregate rows
    for met in ALL_METRICS:
        rows = []
        for proj in projs:
            rows.append([proj] + ["" if per_proj[proj].get(m, {}).get(met) is None
                                  else f"{per_proj[proj][m][met]:.6f}"
                                  for m in methods])
        for an in AGG_NAMES:
            rows.append([an] + ["" if aggs[an][m].get(met) is None
                                else f"{aggs[an][m][met]:.6f}" for m in methods])
        # how many projects each column's average actually covers -- coverage
        # differs by method, so two averages are not always over the same set
        rows.append(["n_projects_in_average"] +
                    [str(sum(1 for p in projs
                             if per_proj[p].get(m, {}).get(met) is not None))
                     for m in methods])
        write_csv(TABD / f"omega_{met}.csv", ["project"] + methods, rows)

    # one long table carrying everything
    rows = []
    for proj in projs:
        for m in methods:
            rows.append([proj, m] + ["" if per_proj[proj].get(m, {}).get(k) is None
                                     else f"{per_proj[proj][m][k]:.6f}"
                                     for k in ALL_METRICS])
    for an in AGG_NAMES:
        for m in methods:
            rows.append([an, m] + ["" if aggs[an][m].get(k) is None
                                   else f"{aggs[an][m][k]:.6f}" for k in ALL_METRICS])
    write_csv(TABD / "omega_all_metrics_long.csv",
              ["project", "method"] + ALL_METRICS, rows)

    # ---- significance ------------------------------------------------------
    ref_method = next((m for m in methods
                       if m.startswith("Switch") and "per-project" in m),
                      methods[0])
    others = [m for m in methods if m != ref_method]

    boot_rows = []
    for proj in sorted(subsets):
        for other in others:
            if ref_method not in subsets[proj] or other not in subsets[proj]:
                continue
            ya, pa, yha = subsets[proj][ref_method]
            yb, pb, yhb = subsets[proj][other]
            if len(ya) != len(yb):
                continue
            for met in ("MCC", "Macro_F1", "AUC"):
                r = paired_bootstrap(ya, pa, pb, yha, yhb, metric=met)
                if r:
                    boot_rows.append([
                        proj, ref_method, other, met, f"{r['mean_diff']:.6f}",
                        f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}",
                        f"{r['p_two_sided']:.6f}", r["n_boot"],
                        "yes" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "no"])
    write_csv(TABD / "omega_significance_bootstrap.csv",
              ["project", "method_A", "method_B", "metric", "mean_diff_A_minus_B",
               "ci_lo", "ci_hi", "p_two_sided", "n_boot", "significant_95"],
              boot_rows)

    wil_rows = []
    for other in others:
        for met in ALL_METRICS:
            r = wilcoxon_across_projects(per_proj, ref_method, other, met)
            if r:
                wil_rows.append([ref_method, other, met, r["n_projects"],
                                 f"{r['median_diff']:.6f}", f"{r['statistic']:.3f}",
                                 f"{r['p']:.6f}", r["wins"], r["losses"], r["ties"],
                                 "yes" if r["p"] < 0.05 else "no"])
    write_csv(TABD / "omega_significance_wilcoxon.csv",
              ["method_A", "method_B", "metric", "n_projects", "median_diff",
               "statistic", "p", "wins_A", "losses_A", "ties", "significant_05"],
              wil_rows)

    # ---- figures -----------------------------------------------------------
    # the degenerate controls belong in the tables but would waste three bars
    # per group in every figure; they are constant by construction
    plot_methods = [m for m in methods
                    if m not in ("All-Buggy", "All-Benign", "Base-rate")]

    made = []
    for met in ("MCC", "Macro_F1", "AUC", "Buggy_F1", "G_Mean", "Popt", "ACC20"):
        s = bar_figure(per_proj, aggs, plot_methods, met, projs)
        if s:
            made.append(s)
    s = lift_figure(coverage, [c[0] for c in coverage])
    if s:
        made.append(s)

    (DEST / "MANIFEST.json").write_text(json.dumps({
        "subset": "commits satisfying Omega (Eq. orphan): a commit relocating a "
                  "file that some untouched file still imports",
        "requires_neo4j": False,
        "run": "python kgcommit_repro/inference/make_omega_subset_eval.py",
        "methods": methods,
        "metrics": ALL_METRICS,
        "effort_proxy": "la + ld (churn), from data/apachejit/projects/",
        "reference_method_for_significance": ref_method,
        "projects_evaluated": projs,
        "projects_excluded": [{"project": p, "reason": r} for p, r in excluded],
        "best_classical_baseline_per_project": best_baseline,
        "threshold_note": "Operating-point metrics use the hard decisions the "
                          "deployed system made on the FULL stream, restricted "
                          "to the subset. Re-tuning inside a subset smaller "
                          "than the tuner's INIT=300 warm-up would leave the "
                          "threshold at its 0.5 default and predict every "
                          "commit buggy.",
        "baselines": "Every baseline is a separate column. Per-commit "
                     "prediction vectors come from baseline_extra_results.pkl "
                     "['raws']; the degenerate controls (All-Buggy, All-Benign, "
                     "Base-rate) keep no vector but are constant by "
                     "construction and are reconstructed from the labels.",
        "min_subset_size": MIN_EVAL,
        "min_positives": MIN_POS,
        "weights": {p: {"n_omega_eval": n, "n_omega_buggy": q}
                    for p, (n, q) in weights.items()},
        "significance": {
            "within_project": "paired bootstrap over commits (2000 resamples), "
                              "95% CI on the metric difference",
            "across_projects": "Wilcoxon signed-rank over per-project values"},
        "figures": sorted(made),
        "sources": [
            "outputs/import_handling_check/<p>_orphaned_dependant_cases.csv",
            "outputs/<p>/raw_method_scores.csv",
            "outputs/<p>/final_final_run/fusion/raw_fusion_scores{,_overall}.pkl",
            "outputs/<p>/baseline_results.pkl (baseline_raw)",
            "data/apachejit/projects/apache_<p>.csv"],
    }, indent=2))

    print(f"evaluated {len(projs)} projects: {', '.join(projs)}")
    for p, r in excluded:
        print(f"  excluded {p}: {r}")
    print(f"methods ({len(methods)}): {', '.join(methods)}")
    print(f"\ntables  -> {TABD}\nfigures -> {FIGD}")


if __name__ == "__main__":
    main()
