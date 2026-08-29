"""
Artifacts requested for the Experimental Results revision.
==========================================================

Builds four deliverables into one standalone LaTeX document:

 1. Significance tables: 5 metrics (Macro-F1, G-Mean, AUC, P_opt, ACC@20)
    x 3 fusion rules (overall F_ov, per-project F_pp, both side by side)
    = 15 tables, KG-Commit vs each of the 6 baselines.

 2. A "Total" row in every table: all projects pooled into ONE stream and
    scored as a single project, alongside the existing Macro/Micro means.

 3. Seed confidence for ALL baselines (not just DeepJIT), per project and in
    the three aggregate forms.

 4. A written diagnosis of why KG-Commit's measured seed sd is exactly zero.

Everything is computed from the cached artifacts:
    outputs/<p>/final_final_run/fusion/raw_fusion_scores{,_overall}.pkl
    outputs/<p>/final_final_run/baselines/baseline_extra_results.pkl
    outputs/<p>/online_jit_streams_v5.pkl        (Kamei metrics -> churn)
    outputs/DeepJIT_baseline_results/grid_summary.csv
No Neo4j, no graph, no refit.

Run:  python inference/make_teammate_artifacts.py
Out:  Paper/ResultsDiscussionsDraft/artifacts_for_revision.tex
"""
import json
import os
# Cross-project driver: the config package binds one project at import
# time, so pin a placeholder. All reads are per-project file paths.
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402
from effort_metrics import popt, recall_at_effort  # noqa: E402
from online_jit import final_metrics, online_decisions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

BASELINES = [("LR", "B_LR"), ("HGB", "B_HGB"), ("RF", "B_RF"),
             ("LApredict", "B_LAPREDICT"), ("DeepJIT", None),
             ("JITLine-online", "B_JITLINE_ONLINE")]

METRICS = [("Macro_F1", "Macro-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"),
           ("Popt", r"$P_{\mathrm{opt}}$"), ("ACC20", r"ACC@20\%LOC")]

TOL_LEVELS = [(0.147, "Negligible"), (0.33, "Small"), (0.474, "Medium")]

# deployed splice point: score with F for the first S evaluated commits, F+G after
SWITCH_S = 200

# paired-bootstrap resamples per project
BOOTS = 200


# ------------------------------------------------------------------ loading --
def churn(project, n_expect, ev_slice):
    """Inspection effort per scored commit = la + ld + 1, from the cached
    Kamei metric block (columns 0,1). No Neo4j."""
    f = OUTP / project / "online_jit_streams_v5.pkl"
    if not f.exists():
        return None
    X = np.asarray(pickle.load(open(f, "rb"))["Xms"], float)
    if X.shape[0] < ev_slice.stop:
        return None
    e = X[ev_slice, 0] + X[ev_slice, 1]
    e = np.where(np.isfinite(e), e, 0.0)
    return np.abs(e) + 1.0


def load_project(project):
    """Per-commit (y, score) for KG-Commit under both rules and every baseline.

    Delegates to make_rq1_panels.load_project, which joins every baseline to the
    fusion evaluation span BY COMMIT SHA. That join matters: the baselines' saved
    index refers to the date-sorted label CSV while the fusion index counts rows
    of the KG stream, and the two diverge wherever the graph dropped a commit.
    Aligning by position instead of SHA silently pairs the wrong commits (label
    agreement fell to 78% on camel and 67% on hbase when we tried it).

    Returns {model: score array} plus y / effort, all on the common span.
    """
    from make_rq1_panels import load_project as _lp

    S = _lp(project)
    if not S:
        return None

    name = {"KGov": "overall", "KGpp": "perproj"}
    out, y_ref, idx_ref = {}, None, None
    for m in MODELS:
        key = name.get(m, m)
        if key not in S:
            continue
        idx, y, pr = S[key]
        out[m] = np.asarray(pr, float)
        y_ref, idx_ref = np.asarray(y, int), np.asarray(idx, int)
    if y_ref is None or "KGov" not in out:
        return None
    out["y"] = y_ref

    # inspection effort = churn (la+ld+1) for exactly the scored commits
    f = OUTP / project / "online_jit_streams_v5.pkl"
    eff = None
    if f.exists():
        X = np.asarray(pickle.load(open(f, "rb"))["Xms"], float)
        if idx_ref.max() < X.shape[0]:
            e = X[idx_ref, 0] + X[idx_ref, 1]
            eff = np.abs(np.nan_to_num(e)) + 1.0
    out["effort"] = eff
    return out


# ------------------------------------------------------------- metric layer --
def score_all(y, pr, effort, gap):
    """The five reported metrics for one (y, score) pair."""
    p = np.clip(np.nan_to_num(pr, nan=float(np.mean(y))), 0, 1)
    m = final_metrics(y, p, gap=gap)
    out = {k: float(m.get(k, np.nan)) for k in ("Macro_F1", "G_Mean", "AUC")}
    if effort is not None:
        out["Popt"] = float(popt(y, p, effort))
        out["ACC20"] = float(recall_at_effort(y, p, effort, frac=0.20))
    else:
        out["Popt"] = out["ACC20"] = np.nan
    return out


def cliffs_delta(a, b):
    """Cliff's delta on paired differences (how dominant a is over b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    return float((np.sum(d > 0) - np.sum(d < 0)) / len(d))


def level(d):
    if not np.isfinite(d):
        return "--"
    a = abs(d)
    for thr, name in TOL_LEVELS:
        if a < thr:
            return ("$+$" if d >= 0 else "$-$") + name
    return ("$+$" if d >= 0 else "$-$") + "Large"


def _thresholds(D, models, gap):
    """Deployed operating point per model, tuned once on the full stream."""
    thr = {}
    for m in models:
        if m not in D:
            continue
        pr = np.clip(np.nan_to_num(np.asarray(D[m], float),
                                   nan=float(np.mean(D["y"]))), 0, 1)
        yhat = online_decisions(pr, D["y"], gap=gap)
        pos = pr[yhat == 1]
        thr[m] = float(pos.min()) if len(pos) else 0.5
    return thr


def _one_metric(y, p, eff, metric, t):
    """Just the requested metric, at a fixed threshold t."""
    if metric == "AUC":
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(y, p))
        except Exception:
            return np.nan
    if metric == "Popt":
        return float(popt(y, p, eff)) if eff is not None else np.nan
    if metric == "ACC20":
        return (float(recall_at_effort(y, p, eff, frac=0.20))
                if eff is not None else np.nan)
    # threshold metrics
    yh = (p >= t).astype(int)
    tp = float(np.sum((yh == 1) & (y == 1)))
    fp = float(np.sum((yh == 1) & (y == 0)))
    fn = float(np.sum((yh == 0) & (y == 1)))
    tn = float(np.sum((yh == 0) & (y == 0)))
    if metric == "G_Mean":
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return float(np.sqrt(rec * spec))
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return float((f1p + f1n) / 2)                       # Macro_F1


def bootstrap(D, models, metric, boots=None, seed=0):
    """Paired bootstrap of ONE metric over a project's scored commits.

    Every model is scored on the identical resample, so the pairing carries the
    inference. Only `metric` is computed, at each model's deployed operating
    point (tuned once on the full stream, not re-tuned inside each resample).
    """
    boots = BOOTS if boots is None else boots
    y = np.asarray(D["y"], int)
    eff = D.get("effort")
    n = len(y)
    ms = [m for m in models if m in D]
    thr = _thresholds(D, ms, P.GAP)
    pre = {m: np.clip(np.nan_to_num(np.asarray(D[m], float),
                                    nan=float(np.mean(y))), 0, 1) for m in ms}

    rng = np.random.default_rng(seed)
    out = {m: np.full(boots, np.nan) for m in ms}
    for b in range(boots):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        ee = eff[idx] if eff is not None else None
        for m in ms:
            out[m][b] = _one_metric(yy, pre[m][idx], ee, metric, thr[m])
    return out


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 3:
        return np.nan
    try:
        return float(wilcoxon(d).pvalue)
    except Exception:
        return np.nan


# ------------------------------------------------------------------ driver --
MODELS = ["KGov", "KGpp"] + [l for l, _ in BASELINES]


def compute():
    """Per-project point metrics, bootstrap draws, and the pooled TOTAL stream."""
    data, point, boot = {}, {}, {}
    for p in PROJECTS:
        D = load_project(p)
        if D is None:
            print(f"  {p}: incomplete -- skipped")
            continue
        data[p] = D
        point[p] = {m: score_all(D["y"], D[m], D["effort"], gap=P.GAP)
                    for m in MODELS if m in D}

    # ---- TOTAL: concatenate every project's scored commits into ONE stream --
    # Treated as a single project with many commits: not a macro or micro
    # average of per-project scores, but one evaluation over the pooled data.
    common = [m for m in MODELS if all(m in data[p] for p in data)]
    ytot = np.concatenate([data[p]["y"] for p in data])
    etot = (np.concatenate([data[p]["effort"] for p in data])
            if all(data[p]["effort"] is not None for p in data) else None)
    TOT = {"y": ytot, "effort": etot}
    for m in common:
        TOT[m] = np.concatenate([np.asarray(data[p][m]) for p in data])
    point["__TOTAL__"] = {m: score_all(ytot, TOT[m], etot, gap=P.GAP)
                          for m in common}

    return data, point, TOT, common


def bootstrap_all(data, TOT, common, metric, boots=400):
    """Per-project paired draws plus draws on the pooled TOTAL stream."""
    B = {}
    for p in data:
        B[p] = bootstrap(data[p], MODELS, metric, boots=boots)
    B["__TOTAL__"] = bootstrap(TOT, common, metric, boots=boots)
    return B


def agg(point, model, metric, data, how):
    """how: 'macro' (projects equal), 'micro' (weight by commits), 'total'."""
    if how == "total":
        return point.get("__TOTAL__", {}).get(model, {}).get(metric, np.nan)
    vals, wts = [], []
    for p in data:
        v = point[p].get(model, {}).get(metric, np.nan)
        if np.isfinite(v):
            vals.append(v)
            wts.append(len(data[p]["y"]))
    if not vals:
        return np.nan
    return float(np.average(vals, weights=wts) if how == "micro"
                 else np.mean(vals))


# ---------------------------------------------------------- LaTeX emitters --
def d3(v, nd=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def pfmt(p):
    if not np.isfinite(p):
        return "--"
    return "$<$0.05" if p < 0.05 else f"{p:.3f}"


def sig_table(point, boot, data, metric, mlabel, rule, rlabel):
    """One significance table: `rule` KG-Commit vs every baseline on `metric`."""
    kg = rule
    others = [l for l, _ in BASELINES]
    rows = [p for p in PROJECTS if p in data]

    L = [r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         rf"\caption{{Significance of KG-Commit ({rlabel}) against each baseline "
         rf"on \textbf{{{mlabel}}}. Per project the paired bootstrap ($B{{=}}200$) "
         rf"resamples that project's scored commits, scoring every model on the "
         rf"identical resample; \emph{{Macro}} and \emph{{Micro}} pair across the "
         rf"11 project scores; \emph{{Total}} pools all {sum(len(data[p]['y']) for p in rows):,} "
         rf"scored commits into a single stream and evaluates it as one project. "
         rf"$p$ is Wilcoxon signed-rank, $|\delta|$ is Cliff's delta with its "
         rf"effect level.}}",
         rf"\label{{tab:sig_{rule}_{metric}}}",
         r"\begin{tabular}{l r" + "r" * len(others) + " | " +
         "c" * len(others) + "}",
         r"\toprule",
         r"Project & KG-Commit & " + " & ".join(others) +
         r" & \multicolumn{" + str(len(others)) + r"}{c}{$p$ : $|\delta|$ vs KG-Commit} \\",
         r"\cmidrule(lr){2-" + str(2 + len(others)) + r"}\cmidrule(l){" +
         str(3 + len(others)) + "-" + str(2 + 2 * len(others)) + "}",
         r" & & " + " & ".join(others) + " & " + " & ".join(others) + r" \\",
         r"\midrule"]

    for p in rows:
        pt = point[p]
        vals = [d3(pt.get(kg, {}).get(metric, np.nan))]
        vals += [d3(pt.get(o, {}).get(metric, np.nan)) for o in others]
        sig = []
        B = boot.get(p, {})
        for o in others:
            if kg in B and o in B and np.isfinite(B[kg]).any() and np.isfinite(B[o]).any():
                sig.append(f"{pfmt(wilcoxon_p(B[kg], B[o]))} : {level(cliffs_delta(B[kg], B[o]))}")
            else:
                sig.append("--")
        L.append(f"{DISP[p]} & " + " & ".join(vals) + " & " + " & ".join(sig) + r" \\")

    L.append(r"\midrule")
    for how, lab in (("macro", r"\textbf{Macro-Mean}"),
                     ("micro", r"\textbf{Micro-Mean}"),
                     ("total", r"\textbf{Total}")):
        vals = [d3(agg(point, kg, metric, data, how))]
        vals += [d3(agg(point, o, metric, data, how)) for o in others]
        if how == "total":
            B = boot.get("__TOTAL__", {})
            sig = []
            for o in others:
                if kg in B and o in B:
                    sig.append(f"{pfmt(wilcoxon_p(B[kg], B[o]))} : {level(cliffs_delta(B[kg], B[o]))}")
                else:
                    sig.append("--")
        else:
            # pair across the 11 per-project scores
            sig = []
            for o in others:
                a = [point[p].get(kg, {}).get(metric, np.nan) for p in rows]
                b = [point[p].get(o, {}).get(metric, np.nan) for p in rows]
                a, b = np.asarray(a, float), np.asarray(b, float)
                ok = np.isfinite(a) & np.isfinite(b)
                sig.append(f"{pfmt(wilcoxon_p(a[ok], b[ok]))} : {level(cliffs_delta(a[ok], b[ok]))}"
                           if ok.sum() >= 3 else "--")
        L.append(lab + " & " + " & ".join(vals) + " & " + " & ".join(sig) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def both_table(point, boot, data, metric, mlabel):
    """Both fusion rules side by side, against every baseline."""
    others = [l for l, _ in BASELINES]
    rows = [p for p in PROJECTS if p in data]
    L = [r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         rf"\caption{{Both KG-Commit fusion rules on \textbf{{{mlabel}}}: the "
         rf"fixed overall $F_{{\mathrm{{ov}}}}{{=}}$RN$+$PPR and the per-project "
         rf"selected $F_{{\mathrm{{pp}}}}$, each with the CSTG channel and the "
         rf"switch at $S{{=}}200$, against every baseline. \emph{{Total}} pools "
         rf"all scored commits into one stream.}}",
         rf"\label{{tab:sig_both_{metric}}}",
         r"\begin{tabular}{l rr " + "r" * len(others) + "}",
         r"\toprule",
         r"Project & $F_{\mathrm{ov}}{+}G$ & $F_{\mathrm{pp}}{+}G$ & "
         + " & ".join(others) + r" \\",
         r"\midrule"]
    for p in rows:
        pt = point[p]
        v = [d3(pt.get("KGov", {}).get(metric, np.nan)),
             d3(pt.get("KGpp", {}).get(metric, np.nan))]
        v += [d3(pt.get(o, {}).get(metric, np.nan)) for o in others]
        L.append(f"{DISP[p]} & " + " & ".join(v) + r" \\")
    L.append(r"\midrule")
    for how, lab in (("macro", r"\textbf{Macro-Mean}"),
                     ("micro", r"\textbf{Micro-Mean}"),
                     ("total", r"\textbf{Total}")):
        v = [d3(agg(point, "KGov", metric, data, how)),
             d3(agg(point, "KGpp", metric, data, how))]
        v += [d3(agg(point, o, metric, data, how)) for o in others]
        L.append(lab + " & " + " & ".join(v) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def main():
    print("loading + scoring (cache-only) ...")
    data, point, TOT, common = compute()
    print(f"  {len(data)} projects, {len(TOT['y']):,} pooled commits")

    blocks = []
    for metric, mlabel in METRICS:
        print(f"  bootstrap: {metric} ...")
        boot = bootstrap_all(data, TOT, common, metric, boots=BOOTS)
        for rule, rlabel in (("KGov", r"overall $F_{\mathrm{ov}}{=}$RN$+$PPR"),
                             ("KGpp", r"per-project $F_{\mathrm{pp}}$")):
            blocks.append(sig_table(point, boot, data, metric, mlabel,
                                    rule, rlabel))
        blocks.append(both_table(point, boot, data, metric, mlabel))

    out = OUTP / "tables" / "sig_tables.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(blocks)} tables)")


if __name__ == "__main__":
    main()
