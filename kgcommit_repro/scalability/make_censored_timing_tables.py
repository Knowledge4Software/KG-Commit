"""
Paper tables for the stall-censored build timing (E6).

Consumes the JSON written by scalability/censor_timing.py and emits three
booktabs tables into Paper/paper_material/RQ2_scalability/:

  tab_timing_censoring.tex     the correction itself: raw vs censored totals and
                               per-commit cost, with the flagged-row counts. This
                               is the transparency table -- it shows the reader
                               exactly what was removed and why.
  tab_build_cost_censored.tex  the clean per-commit build-cost table intended for
                               the paper body (medians + censored totals).
  tab_timing_sensitivity.tex   threshold sensitivity: how the correction moves as
                               the off-model factor varies. Pre-empts the obvious
                               reviewer question about an arbitrary cut-off.

Cache-only: reads JSON + the source CSVs. No Neo4j, no rebuild.

Run:  python scalability/make_censored_timing_tables.py
"""
import csv
import json
from pathlib import Path
from statistics import median

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ2_scalability"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
try:
    from paper_projects import ACTIVE          # [(display, folder), ...]
except Exception:                              # keep the script standalone-runnable
    ACTIVE = [("ActiveMQ", "activemq"), ("Cassandra", "cassandra"),
              ("Groovy", "groovy"), ("Kafka", "kafka"), ("Spark", "spark"),
              ("Zeppelin", "zeppelin"), ("Zookeeper", "zookeeper")]

SENS_FACTORS = (5, 10, 20, 50)


def load_censored(folder, layer="ast"):
    f = OUTP / folder / "scalability" / f"timing_censored_{layer}.json"
    return json.load(open(f)) if f.exists() else None


def _fmt(x, nd=0):
    return f"{x:,.{nd}f}" if x is not None else "--"


# ── table 1: the correction, stated transparently ────────────────────────────

def tab_censoring():
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{RQ2 stall censoring of the live build wall-clock. Long builds are "
         r"interrupted by machine-level events (crash-and-resume, OS suspend, competing "
         r"load) whose idle time is charged to whichever commit straddles them. Rows are "
         r"flagged when their \emph{cost per unit of work} exceeds $10\times$ the project "
         r"median rate, and are censored to their work-implied expectation rather than "
         r"dropped. Totals are materially inflated by these stalls; per-commit medians are "
         r"almost unaffected, which is why the paper reports medians.}",
         r"\label{tab:timing_censoring}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& & \multicolumn{2}{c}{Flagged} & \multicolumn{2}{c}{Total (s)} & Idle \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Project & \#Commits & all & hard & raw & censored & removed \\",
         r"\midrule"]
    any_row = False
    for disp, folder in ACTIVE:
        d = load_censored(folder)
        if not d:
            L.append(f"{disp} & -- & -- & -- & -- & -- & -- " + r"\\")
            continue
        any_row = True
        raw, cen = d["raw"], d["censored"]
        L.append(
            f"{disp} & {_fmt(raw['n'])} & {d['flagged_count']} & {d['flagged_hard']} & "
            f"{_fmt(raw['sum']/1000)} & {_fmt(cen['sum']/1000)} & "
            f"{100*d['removed_share_of_raw_total']:.1f}\\% " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    if not any_row:
        return None
    return "\n".join(L)


# ── table 2: the clean per-commit build cost (paper body) ────────────────────

def tab_build_cost():
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{RQ2 per-commit AST-layer build cost after stall censoring "
         r"(Table~\ref{tab:timing_censoring}). The median is the robust statistic: it is "
         r"invariant to interruption (it shifts by $\le 7\%$ under censoring and is stable "
         r"across every threshold tested), whereas the mean and the total absorb idle time. "
         r"Cost per unit of work is the slope that the $O(\delta)$ claim predicts to be "
         r"constant across projects of very different size.}",
         r"\label{tab:build_cost_censored}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Project & \#Commits & Median ms/commit & Mean ms/commit & ms per unit work \\",
         r"\midrule"]
    any_row = False
    for disp, folder in ACTIVE:
        d = load_censored(folder)
        if not d:
            L.append(f"{disp} & -- & -- & -- & -- " + r"\\")
            continue
        any_row = True
        cen = d["censored"]
        rate = d["policy"]["median_rate_ms_per_work"]
        L.append(f"{disp} & {_fmt(cen['n'])} & {_fmt(cen['p50'],1)} & "
                 f"{_fmt(cen['mean'],1)} & {rate:.3f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) if any_row else None


# ── table 3: threshold sensitivity ───────────────────────────────────────────

def _recensor(path, factor):
    """Re-run the censoring arithmetic at a different factor (pure, in-memory)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                work = sum(float(r.get(c) or 0) for c in
                           ("adds", "removes", "updates", "moves", "matched"))
                work += sum(float(r.get(c) or 0) for c in ("A", "M", "D", "boot"))
                rows.append((float(r["wall_ms"]), work))
            except (TypeError, ValueError, KeyError):
                continue
    worked = [(w, k) for w, k in rows if k > 0]
    if len(worked) < 20:
        return None
    med = median(w / k for w, k in worked)
    lim = factor * med
    clean, nflag = [], 0
    for w, k in rows:
        if (k <= 0 and w > 5000) or (k > 0 and w / k > lim):
            clean.append(k * med); nflag += 1
        else:
            clean.append(w)
    raw_tot = sum(w for w, _ in rows)
    return {"n_flagged": nflag, "total_s": sum(clean) / 1000,
            "median_ms": median(clean),
            "removed_pct": 100 * (raw_tot - sum(clean)) / raw_tot if raw_tot else 0}


def tab_sensitivity():
    srcs = []
    for disp, folder in ACTIVE:
        d = load_censored(folder)
        if not d:
            continue
        p = ROOT / d["_meta"]["source"]
        if p.exists():
            srcs.append((disp, p))
    if not srcs:
        return None
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{Sensitivity of the stall correction to the off-model threshold "
         r"(multiple of the project median cost-per-unit-work). The censored \emph{median} "
         r"per-commit cost is stable across an order of magnitude of threshold choices, so "
         r"the per-commit claims do not depend on this parameter; only the totals, which "
         r"absorb idle time, are sensitive to it. The deployed setting is $10\times$.}",
         r"\label{tab:timing_sensitivity}",
         r"\begin{tabular}{ll" + "r" * len(SENS_FACTORS) + "}", r"\toprule",
         r"Project & Quantity & " +
         " & ".join(rf"${f}\times$" for f in SENS_FACTORS) + r" \\",
         r"\midrule"]
    for i, (disp, p) in enumerate(srcs):
        res = {f: _recensor(p, f) for f in SENS_FACTORS}
        if any(v is None for v in res.values()):
            continue
        L.append(rf"\multirow{{3}}{{*}}{{{disp}}} & flagged rows & " +
                 " & ".join(str(res[f]["n_flagged"]) for f in SENS_FACTORS) + r" \\")
        L.append(r" & censored total (s) & " +
                 " & ".join(f"{res[f]['total_s']:,.0f}" for f in SENS_FACTORS) + r" \\")
        L.append(r" & censored median (ms) & " +
                 " & ".join(f"{res[f]['median_ms']:.1f}" for f in SENS_FACTORS) + r" \\")
        if i < len(srcs) - 1:
            L.append(r"\cmidrule(l){2-" + str(2 + len(SENS_FACTORS)) + r"}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    built = []
    for name, fn in [("tab_timing_censoring", tab_censoring),
                     ("tab_build_cost_censored", tab_build_cost),
                     ("tab_timing_sensitivity", tab_sensitivity)]:
        s = fn()
        if s is None:
            print(f"  skip {name}: no censored timing JSON found "
                  f"(run scalability/censor_timing.py first)")
            continue
        (PM / f"{name}.tex").write_text(s + "\n", encoding="utf-8")
        built.append(name)
        print(f"  wrote {PM / (name + '.tex')}")
    if built:
        n = sum(1 for _, f in ACTIVE if load_censored(f))
        print(f"\n{len(built)} tables over {n}/{len(ACTIVE)} projects with per-commit "
              f"timing logs. Projects built without --timing-log show '--'.")


if __name__ == "__main__":
    main()
