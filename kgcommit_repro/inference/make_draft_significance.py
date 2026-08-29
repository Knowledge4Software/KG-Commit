"""
Significance-test table: KG-Commit vs. each baseline on Macro-F1.
=================================================================

Follows the reporting convention of the target journal (Knowledge-Based Systems):
a per-task Macro-F1 table with, for every baseline, the Wilcoxon signed-rank
p-value and Cliff's delta effect level, plus an Average & Win/Tie/Loss row.

DESIGN
------
The pairing unit is the PROJECT, matching the reference paper (one F1 per task,
Wilcoxon across tasks). Our n = 11 projects. Two consequences worth stating:

  * The Wilcoxon test is computed across the 11 paired project scores. With n=11
    the smallest attainable two-sided p is ~0.001, so we report exact p values
    rather than a "<0.05" bucket wherever they exceed it.
  * Cliff's delta is computed on the same 11 paired values. It is a
    non-parametric effect size on the DIFFERENCES, so it answers "how dominant
    is KG-Commit across projects", not "how large is the gap on any one project".

We deliberately do NOT pool per-window trajectories: the fusion and baseline
trajectories are computed over different numbers of rolling windows (e.g. 29 vs
31 on zookeeper), so pooling them would pair non-corresponding observations and
inflate n by ~270x on an assumption we cannot verify.

Cliff's delta -> effective level thresholds follow the reference paper:
    |d| < 0.147 negligible; < 0.33 small; < 0.474 medium; else large.

Three variants are emitted:
  overall   KG-Commit (fixed F_ov = RN+PPR, S@200)      vs. the 6 baselines
  perproj   KG-Commit (per-project F_pp, S@200)         vs. the 6 baselines
  both      both KG-Commit rules, side by side          vs. the 6 baselines

Cache-only. No Neo4j.

Out: Paper/ResultsDiscussionsDraft/tab_significance_{overall,perproj,both}.tex
Run: python inference/make_draft_significance.py
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

BCOLS = [("LR", "B_LR"), ("HGB", "B_HGB"), ("RF", "B_RF"),
         ("LApredict", "B_LAPREDICT"), ("DeepJIT", None),
         ("JITLine-on", "B_JITLINE_ONLINE")]
BDISP = {"JITLine-on": "JITLine-online"}


def cliffs_delta(a, b):
    """Cliff's delta of a over b (paired vectors treated as two samples)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = sum(1 for x, y in itertools.product(a, b) if x > y)
    lt = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (gt - lt) / (len(a) * len(b))


def level(d):
    """Effective level, with sign, per the reference paper's thresholds."""
    m = abs(d)
    lab = ("Negligible" if m < 0.147 else
           "Small" if m < 0.33 else
           "Medium" if m < 0.474 else "Large")
    return ("$+$" if d >= 0 else "$-$") + lab


def deepjit_macro_f1():
    df = pd.read_csv(OUTP / "DeepJIT_baseline_results" / "grid_summary.csv")

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    df["mf1"] = df.apply(
        lambda r: (f1(r.tp, r.fp, r.fn) + f1(r.tn, r.fn, r.fp)) / 2, axis=1)
    df["ps"] = df["project"].str.replace("apache/", "", regex=False)
    return df.groupby("ps")["mf1"].mean().to_dict()


def load():
    S = json.load(open(OUTP / "final_final_run_summary.json"))
    DJ = deepjit_macro_f1()
    rows = {}
    for p in PROJECTS:
        B = pickle.load(open(OUTP / p / "final_final_run" / "baselines"
                             / "baseline_extra_results.pkl", "rb"))
        d = {"KGov": float(S[p]["switch"]["overall"]["grid"]["200"]["Macro_F1"]),
             "KGpp": float(S[p]["switch"]["per_project"]["grid"]["200"]["Macro_F1"])}
        for lab, key in BCOLS:
            d[lab] = (float(DJ[p]) if key is None
                      else float(B["baselines"][key]["Macro_F1"]))
        rows[p] = d
    return rows


def bootstrap_macro_f1(project, boots=400, seed=0):
    """Per-project bootstrap distribution of Macro-F1 for every model.

    Macro-F1 is a set-level metric, so a paired test needs a distribution rather
    than one number per project. We resample the project's scored commits with
    replacement (a moving-block bootstrap would be needed for a time-series claim;
    here the models are compared on the SAME resample, so the pairing -- not the
    marginal distribution -- carries the inference) and recompute Macro-F1 for
    every model on each resample. All models are scored on the identical commit
    set, joined by commit hash, so resample b means the same commits for all.

    Returns {model: array(boots)} or None if the project cannot be loaded.
    """
    from make_rq1_panels import load_project, _online_decisions

    S = load_project(project)
    if not S:
        return None
    models = ["KGov", "KGpp"] + [l for l, _ in BCOLS]
    name = {"KGov": "overall", "KGpp": "perproj"}
    yhat, y_ref = {}, None
    for m in models:
        key = name.get(m, BDISP.get(m, m))
        if key not in S:
            return None
        _, y, pr = S[key]
        yhat[m] = _online_decisions(pr, y)
        y_ref = y

    rng = np.random.default_rng(seed)
    n = len(y_ref)
    out = {m: np.empty(boots) for m in models}
    for b in range(boots):
        idx = rng.integers(0, n, n)
        yy = y_ref[idx]
        if len(np.unique(yy)) < 2:
            for m in models:
                out[m][b] = np.nan
            continue
        for m in models:
            out[m][b] = _macro_f1_np(yy, yhat[m][idx])
    return out


def _macro_f1_np(y, yh):
    vals = []
    for c in (0, 1):
        tp = np.sum((yh == c) & (y == c))
        fp = np.sum((yh == c) & (y != c))
        fn = np.sum((yh != c) & (y == c))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        vals.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(vals))


def stats_vs(rows, kg, bl):
    """Wilcoxon p and Cliff's delta for kg vs bl across the 11 projects."""
    a = np.array([rows[p][kg] for p in PROJECTS])
    b = np.array([rows[p][bl] for p in PROJECTS])
    if np.allclose(a, b):
        return 1.0, 0.0
    try:
        p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    return p, cliffs_delta(a, b)


def wtl(rows, kg, bl, eps=1e-9):
    w = t = l = 0
    for p in PROJECTS:
        diff = rows[p][kg] - rows[p][bl]
        if abs(diff) <= eps:
            t += 1
        elif diff > 0:
            w += 1
        else:
            l += 1
    return w, t, l


def pfmt(p):
    return r"$<$0.05" if p < 0.05 else f"{p:.3f}"


def paired_boot(a, b):
    """Wilcoxon + Cliff's delta on paired bootstrap draws of Macro-F1."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 5 or np.allclose(a, b):
        return 1.0, 0.0
    try:
        pv = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        pv = 1.0
    d = a - b
    dv = (np.sum(d > 0) - np.sum(d < 0)) / len(d)
    return pv, dv


def build(rows, boot, variant):
    """rows: per-project point Macro-F1. boot: per-project bootstrap draws."""
    KGN = {"KGov": r"$F_{\mathrm{ov}}$", "KGpp": r"$F_{\mathrm{pp}}$"}
    kgs = (["KGov"] if variant == "overall" else
           ["KGpp"] if variant == "perproj" else ["KGov", "KGpp"])
    bl = [l for l, _ in BCOLS]
    f1cols = bl + kgs
    cmps = [(b, k) for b in bl for k in kgs]

    if variant == "overall":
        rule = r"the fixed overall fusion $F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$"
    elif variant == "perproj":
        rule = r"the per-project chosen fusion $F_{\mathrm{pp}}$"
    else:
        rule = (r"both fusion-selection rules --- the fixed "
                r"$F_{\mathrm{ov}}{=}\mathrm{RN{+}PPR}$ and the per-project "
                r"$F_{\mathrm{pp}}$")
    lab = {"overall": "tab:significance", "perproj": "tab:significance_pp",
           "both": "tab:significance_both"}[variant]

    cap = (r"\caption{Macro-F1 of KG-Commit (" + rule + r", with the adaptive "
           r"initial-fit window and the switch to $F{+}G$ at $S{=}200$) against "
           r"the six baselines under the online within-project scenario. For each "
           r"project the best method is in \textbf{bold}. The $p\!:\!|\delta|$ "
           r"block gives the Wilcoxon signed-rank $p$ value and Cliff's delta "
           r"between KG-Commit and each baseline. On a project row the pairing is "
           r"over $400$ bootstrap resamples of that project's scored commits, "
           r"with every model scored on the identical resample; on the mean rows "
           r"it is over the 11 paired project scores. We display $<$0.05 when the "
           r"$p$ value falls below that threshold and the exact value otherwise. "
           r"For Cliff's delta we report the effective level, prefixed by $+$ or "
           r"$-$ to indicate direction. Macro-Mean weights projects equally; "
           r"Micro-Mean weights each by its number of scored commits. "
           r"Win/Tie/Loss counts projects.}")

    L = [r"\begin{table*}[t]", r"\centering",
         r"\scriptsize\setlength{\tabcolsep}{3pt}", cap,
         r"\label{" + lab + "}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(f1cols) + "|" + "c" * len(cmps) + "}",
         r"\toprule",
         r"\multirow{2}{*}{Task} & \multicolumn{" + str(len(f1cols))
         + r"}{c|}{Macro-F1} & \multicolumn{" + str(len(cmps))
         + r"}{c}{$p\!:\!|\delta|$} \\",
         r"\cmidrule(lr){2-" + str(1 + len(f1cols)) + r"}\cmidrule(l){"
         + str(2 + len(f1cols)) + "-" + str(1 + len(f1cols) + len(cmps)) + "}"]

    hdr = [BDISP.get(b, b) for b in bl] + [KGN[k] for k in kgs]
    ch = []
    for b, k in cmps:
        tgt = KGN[k] if len(kgs) > 1 else "KG-Commit"
        ch.append(BDISP.get(b, b) + r" vs.\ " + tgt)
    L.append(" & " + " & ".join(hdr + ch) + r" \\")
    L.append(r"\midrule")

    for p in PROJECTS:
        vals = {m: rows[p][m] for m in f1cols}
        best = max(vals.values())
        cells = [(r"\textbf{" + f"{vals[m]:.3f}" + "}"
                  if abs(vals[m] - best) < 1e-9 else f"{vals[m]:.3f}")
                 for m in f1cols]
        sig = []
        for b, k in cmps:
            d = boot.get(p)
            if not d:
                sig.append("--")
                continue
            pv, dv = paired_boot(d[k], d[b])
            sig.append(pfmt(pv) + r" $:$ " + level(dv))
        L.append(DISP[p] + " & " + " & ".join(cells + sig) + r" \\")

    for name, use_w in ((r"\textbf{Macro-Mean}", False),
                        (r"\textbf{Micro-Mean}", True)):
        w = [rows[p]["n_eval"] for p in PROJECTS] if use_w else None
        avg = {m: float(np.average([rows[p][m] for p in PROJECTS], weights=w))
               for m in f1cols}
        best = max(avg.values())
        cells = [(r"\textbf{" + f"{avg[m]:.3f}" + "}"
                  if abs(avg[m] - best) < 1e-9 else f"{avg[m]:.3f}")
                 for m in f1cols]
        sig = []
        for b, k in cmps:
            pv, dv = stats_vs(rows, k, b)
            sig.append(pfmt(pv) + r" $:$ " + level(dv))
        L.append(r"\midrule " + name + " & " + " & ".join(cells + sig) + r" \\")

    wl = []
    for b, k in cmps:
        a, t, c = wtl(rows, k, b)
        wl.append(f"{a}/{t}/{c}")
    L.append(r"\textbf{Win/Tie/Loss} & "
             + " & ".join([""] * len(f1cols) + wl) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def main():
    rows = load()
    S = json.load(open(OUTP / "final_final_run_summary.json"))
    for p in PROJECTS:
        rows[p]["n_eval"] = int(S[p]["switch"]["overall"]["n_eval"])

    print("bootstrapping per project (400 resamples) ...")
    boot = {}
    for p in PROJECTS:
        d = bootstrap_macro_f1(p)
        boot[p] = d
        print(f"  {p:<11} {'ok' if d else 'SKIPPED'}")

    for v in ("overall", "perproj", "both"):
        out = DRAFT / f"tab_significance_{v}.tex"
        out.write_text(build(rows, boot, v) + "\n", encoding="utf-8")
        print(f"  wrote {out.name}")

    print("\n--- across-project stats (n=11) ---")
    for kg, nm in (("KGov", "overall F"), ("KGpp", "per-project F")):
        print(f"  {nm}:")
        for b, _ in BCOLS:
            pv, dv = stats_vs(rows, kg, b)
            a, t, c = wtl(rows, kg, b)
            print(f"    vs {BDISP.get(b, b):<15} p={pv:<8.4f} d={dv:+.3f} "
                  f"{level(dv).replace('$', ''):<12} W/T/L={a}/{t}/{c}")


if __name__ == "__main__":
    main()
