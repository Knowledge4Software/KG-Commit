"""
Total row for tab:rq4_fg_g50 -- the F -> F+G -> Switch@200 build-up.
====================================================================

That table's columns are pipeline stages, not models, so it needs its own
pooled row. Same convention as the other Total rows: pool DECISIONS, keeping
each project's own online-tuned operating point, rather than re-thresholding a
concatenated score stream (which would impose one global threshold across
projects whose bug rates run 13.6%-61.7%).

Out: outputs/tables/total_rows_rq4.json (+ .tex fragment)
Run: python inference/total_rows_rq4.py
"""
import json
import os
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402
from online_jit import online_decisions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
SWITCH_S = 200


def stages(project):
    """F, F+G and Switch@200 per-commit scores under both fusion rules."""
    ffr = OUTP / project / "final_final_run" / "fusion"
    out = {}
    for rule, fn in (("ov", "raw_fusion_scores_overall.pkl"),
                     ("pp", "raw_fusion_scores.pkl")):
        f = ffr / fn
        if not f.exists():
            return None
        R = pickle.load(open(f, "rb"))
        S = R["scores"]
        Fv = np.asarray(S["F"], float)
        FG = np.asarray(S["F+G"], float)
        sw = FG.copy()
        k = min(SWITCH_S, len(sw))
        sw[:k] = Fv[:k]
        out[f"F_{rule}"] = Fv
        out[f"FG_{rule}"] = FG
        out[f"SW_{rule}"] = sw
        out["y"] = np.asarray(R["y"], int)
    return out


def metrics_from(y, yh):
    tp = float(np.sum((yh == 1) & (y == 1)))
    fp = float(np.sum((yh == 1) & (y == 0)))
    fn = float(np.sum((yh == 0) & (y == 1)))
    tn = float(np.sum((yh == 0) & (y == 0)))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return {"Macro_F1": (f1p + f1n) / 2.0, "G_Mean": float(np.sqrt(rec * spec))}


def main():
    cols = ["F_ov", "FG_ov", "SW_ov", "F_pp", "FG_pp", "SW_pp"]
    ys, dec = [], {c: [] for c in cols}
    n_proj = 0
    for p in PROJECTS:
        st = stages(p)
        if st is None:
            print(f"  {p}: skipped")
            continue
        n_proj += 1
        ys.append(st["y"])
        for c in cols:
            pr = np.clip(np.nan_to_num(st[c], nan=float(np.mean(st["y"]))), 0, 1)
            dec[c].append(online_decisions(pr, st["y"], gap=P.GAP))

    y = np.concatenate(ys)
    res = {"n_commits": int(len(y)), "n_projects": n_proj,
           "pooling": "decisions (per-project thresholds preserved)",
           "stages": {}}
    for c in cols:
        res["stages"][c] = metrics_from(y, np.concatenate(dec[c]))

    print(f"pooled decision stream: {len(y):,} commits, {n_proj} projects\n")
    print(f"{'stage':10}{'Macro-F1':>10}{'G-Mean':>9}")
    for c in cols:
        r = res["stages"][c]
        print(f"{c:10}{r['Macro_F1']:10.4f}{r['G_Mean']:9.4f}")

    d_ov = res["stages"]["SW_ov"]["Macro_F1"] - res["stages"]["F_ov"]["Macro_F1"]
    d_pp = res["stages"]["SW_pp"]["Macro_F1"] - res["stages"]["F_pp"]["Macro_F1"]
    res["delta_switch_minus_F"] = {"overall": d_ov, "per_project": d_pp}
    print(f"\nSwitch@200 - F:  overall {d_ov:+.4f}   per-project {d_pp:+.4f}")

    dst = OUTP / "tables" / "total_rows_rq4.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1), encoding="utf-8")

    # LaTeX row: the table shows 3 metrics per stage; we have 2 threshold ones.
    cells = []
    for c in cols:
        r = res["stages"][c]
        cells += [f"{r['Macro_F1']:.3f}", f"{r['G_Mean']:.3f}", "--"]
    frag = ("% tab:rq4_fg_g50 -- Total = pooled decisions (per-project\n"
            "% thresholds kept). AUC omitted: rank-based, no threshold to pool.\n"
            r"\midrule \textbf{Total} & " + " & ".join(cells) + r" \\")
    (OUTP / "tables" / "total_rows_rq4.tex").write_text(frag, encoding="utf-8")
    print(f"\nwrote {dst.relative_to(ROOT)} (+ .tex)")


if __name__ == "__main__":
    main()
