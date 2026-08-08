"""
Common-evaluation-window helper (fairness fix, cache-only).

KG-Commit's fusion starts scoring at ev0 = W + INIT (INIT=300) because its LR stacking
head needs an init window; the baselines start at W. To compare on identical commits,
every model's aggregate metrics must be computed on the COMMON window [ev0, N).

This module re-scores any model's stored per-commit predictions on [ev0, N) with the
same online-tuned threshold (gap-aware). No Neo4j, no retraining.

Usage:
    from common_window import common_ev0, metrics_common
    ev0 = common_ev0(folder)                          # the shared start commit
    m = metrics_common(y_full, p_full, ev0, N, gap)   # 7-metric dict on [ev0,N)
For baselines whose raws are stored as {idx, y, pred} over their own eval span, use
    metrics_from_raw(raw_dict, ev0, gap)
"""
import pickle
from pathlib import Path
import numpy as np

from online_jit import final_metrics

_OUTP = Path(__file__).resolve().parent.parent.parent / "outputs"


def common_ev0(folder):
    """The common evaluation-window start = KG fusion's ev0 (W+INIT). None if absent."""
    rf = _OUTP / folder / "raw_fusion_scores.pkl"
    if rf.exists():
        return int(pickle.load(open(rf, "rb"))["ev0"])
    ff = _OUTP / folder / "final_fusion_results.pkl"
    if ff.exists():
        meta = pickle.load(open(ff, "rb")).get("meta", {})
        if "W" in meta and "init" in meta:
            return int(meta["W"]) + int(meta["init"])
    return None


def metrics_common(y_full, p_full, ev0, N, gap=0):
    """7-metric dict for a full-length (y,p) restricted to the common window [ev0,N)."""
    y = np.asarray(y_full, int); p = np.asarray(p_full, float)
    ev = np.arange(ev0, N)
    yt = y[ev]; pt = np.clip(np.nan_to_num(p[ev], nan=float(yt.mean())), 0, 1)
    return {k: float(v) for k, v in final_metrics(yt, pt, gap=gap).items()}


def metrics_from_raw(raw, ev0, gap=0):
    """Baseline raws = {idx, y, pred} over its own span; re-score on [ev0, N).
    Returns None if the window has <2 samples or one class."""
    idx = np.asarray(raw["idx"]); y = np.asarray(raw["y"], int); p = np.asarray(raw["pred"], float)
    mask = idx >= ev0
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return None
    return {k: float(v) for k, v in final_metrics(y[mask], p[mask], gap=gap).items()}


def kg_fg_common(folder, name="F+G", gap=0):
    """KG F/F+G metrics on the common window (KG already starts at ev0, so this is its
    natural window; provided for symmetry / gap re-scoring)."""
    rf = _OUTP / folder / "raw_fusion_scores.pkl"
    if not rf.exists():
        return None
    d = pickle.load(open(rf, "rb"))
    if name not in d["scores"]:
        return None
    y = np.asarray(d["y"], int); p = np.asarray(d["scores"][name], float)
    return {k: float(v) for k, v in final_metrics(y, p, gap=gap).items()}
