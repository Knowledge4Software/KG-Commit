"""
Stall censoring for per-commit build timing logs (E6).

The per-commit wall-clock logs written by the online build engines
(build_online_kg.py / build_subgraph_online_kg.py, via --timing-log) record REAL
elapsed time. A long-running build is therefore contaminated by intervals in which
the machine was not doing our work at all: a crash-and-resume, an OS suspend, a
laptop lid close, another process saturating the disk. Those intervals are attributed
to whichever commit happened to straddle them, producing wall_ms values that reflect
the interruption rather than the workload.

The detection principle
-----------------------
Do NOT threshold on wall_ms. A genuinely large commit is legitimately slow, and
censoring it would bias the cost model in our own favour -- the opposite of honest.
Instead normalise by the work the row actually performed,

    work(r) = adds + removes + updates + moves + matched + (A + M + D + boot)
    rate(r) = wall_ms(r) / work(r)          [ms per unit of work]

and flag rows whose RATE is far above the population median rate. A heavy commit is
slow *because it did a lot* and sits on the cost model; an interrupted commit is slow
*while doing nothing* and sits far off it. That asymmetry is what separates them.

Robust statistics only (median / MAD), so the estimate of "normal" is itself
unaffected by the outliers we are looking for.

Policy
------
  off-model  rate > FACTOR x median_rate            (default FACTOR = 10)
  hard       rate > HARD_FACTOR x median_rate       (default HARD_FACTOR = 100)

Censoring replaces a flagged row's wall_ms with its work-implied expectation
(work x median_rate) rather than dropping the row, so per-commit counts and the
stream length are preserved and only the idle component is removed.

What this does and does not touch
---------------------------------
This affects ONLY statistics derived from live build wall-clock. It does NOT touch:
  * prediction latency (scalability/prediction_latency.json) -- a controlled
    in-memory replay, never contaminated;
  * build_complexity.json -- a controlled 300-pair replay timed in one clean
    session, likewise never contaminated.
Medians are near-invariant to these stalls by construction; SUMS are not, which is
why totals are the statistic that most needs this correction.

Cache-only: reads CSVs, writes JSON + a flagged-row manifest. No Neo4j, no rebuild.

Run:
    python scalability/censor_timing.py                  # all projects found
    python scalability/censor_timing.py --factor 20      # stricter
    python scalability/censor_timing.py --project kafka
Out:
    outputs/<p>/scalability/timing_censored.json         # clean stats + policy
    outputs/<p>/scalability/timing_flagged.csv           # the audit trail
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"

WORK_COLS = ("adds", "removes", "updates", "moves", "matched")
FILE_COLS = ("A", "M", "D", "boot")
FACTOR = 10.0        # off-model: rate > FACTOR x median rate
HARD_FACTOR = 100.0  # near-certain stall


# ── io ───────────────────────────────────────────────────────────────────────

def find_logs(project=None):
    """Every per-commit timing CSV on disk, as (project, layer, path)."""
    out = []
    for p in sorted(OUTP.glob("*/*timing*.csv")):
        proj = p.parent.name
        if project and proj != project:
            continue
        layer = p.stem.replace("_timing", "").replace("ast_method", "ast")
        out.append((proj, layer or "ast", p))
    return out


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                work = sum(float(r.get(c) or 0) for c in WORK_COLS)
                files = sum(float(r.get(c) or 0) for c in FILE_COLS)
                rows.append({
                    "idx": int(r["idx"]),
                    "sha": r.get("sha", ""),
                    "wall_ms": float(r["wall_ms"]),
                    "work": work + files,
                })
            except (TypeError, ValueError, KeyError):
                continue      # malformed / truncated line (a crash mid-write)
    return rows


# ── statistics ───────────────────────────────────────────────────────────────

def describe(vals):
    if not vals:
        return {}
    v = sorted(vals)
    n = len(v)

    def q(f):
        return v[min(n - 1, max(0, int(f * n)))]

    return {"n": n, "sum": sum(v), "mean": sum(v) / n, "p50": median(v),
            "p95": q(0.95), "max": v[-1]}


def censor(rows, factor=FACTOR, hard_factor=HARD_FACTOR):
    """Split rows into on-model / off-model by cost-per-unit-work, and return the
    censored wall-clock series alongside the audit trail."""
    worked = [r for r in rows if r["work"] > 0]
    if len(worked) < 20:
        return None

    rates = [r["wall_ms"] / r["work"] for r in worked]
    med_rate = median(rates)
    devs = [abs(x - med_rate) for x in rates]
    mad = median(devs) or 1e-12

    lim = factor * med_rate
    hard_lim = hard_factor * med_rate

    flagged, clean_series = [], []
    for r in rows:
        w = r["work"]
        rate = r["wall_ms"] / w if w > 0 else float("inf")
        # a zero-work row costing real time is idle by definition
        is_off = (w <= 0 and r["wall_ms"] > 5000) or (w > 0 and rate > lim)
        if is_off:
            implied = w * med_rate          # what the workload alone predicts
            flagged.append({
                "idx": r["idx"], "sha": r["sha"],
                "wall_ms": round(r["wall_ms"], 1),
                "work": round(w, 1),
                "ms_per_work": round(rate, 2) if w > 0 else None,
                "robust_z": round((rate - med_rate) / (1.4826 * mad), 1) if w > 0 else None,
                "severity": "hard" if (w > 0 and rate > hard_lim) or w <= 0 else "soft",
                "censored_to_ms": round(implied, 1),
                "removed_ms": round(r["wall_ms"] - implied, 1),
            })
            clean_series.append(implied)
        else:
            clean_series.append(r["wall_ms"])

    raw = [r["wall_ms"] for r in rows]
    return {
        "policy": {
            "rule": "rate = wall_ms / work; flag rate > factor x median_rate",
            "work_definition": "adds+removes+updates+moves+matched+A+M+D+boot",
            "factor": factor, "hard_factor": hard_factor,
            "median_rate_ms_per_work": med_rate,
            "mad_rate": mad,
            "zero_work_idle_threshold_ms": 5000,
            "censoring": "flagged wall_ms replaced by work x median_rate "
                         "(row retained; only the idle component is removed)",
        },
        "raw": describe(raw),
        "censored": describe(clean_series),
        "flagged_count": len(flagged),
        "flagged_hard": sum(1 for f in flagged if f["severity"] == "hard"),
        "removed_ms_total": round(sum(f["removed_ms"] for f in flagged), 1),
        "removed_share_of_raw_total": (
            round(sum(f["removed_ms"] for f in flagged) / sum(raw), 4) if sum(raw) else 0.0),
        "_flagged": flagged,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=None, help="restrict to one project folder")
    ap.add_argument("--factor", type=float, default=FACTOR,
                    help=f"off-model multiple of the median rate (default {FACTOR})")
    ap.add_argument("--hard-factor", type=float, default=HARD_FACTOR,
                    help=f"near-certain-stall multiple (default {HARD_FACTOR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; write nothing")
    args = ap.parse_args()

    logs = find_logs(args.project)
    if not logs:
        print("no per-commit timing CSVs found under outputs/*/ "
              "(only projects built with --timing-log have them)")
        return

    for proj, layer, path in logs:
        rows = load(path)
        res = censor(rows, args.factor, args.hard_factor)
        if res is None:
            print(f"{proj}/{layer}: too few usable rows ({len(rows)}) -- skipped")
            continue

        raw, cen = res["raw"], res["censored"]
        infl_mean = (raw["mean"] / cen["mean"] - 1) * 100 if cen["mean"] else 0
        infl_p50 = (raw["p50"] / cen["p50"] - 1) * 100 if cen["p50"] else 0
        print(f"\n=== {proj} / {layer} ===  {raw['n']} commits")
        print(f"  median cost-per-unit-work : {res['policy']['median_rate_ms_per_work']:.3f} ms")
        print(f"  flagged                   : {res['flagged_count']} "
              f"({res['flagged_hard']} hard)")
        print(f"  idle time removed         : {res['removed_ms_total']/1000:.0f}s "
              f"({100*res['removed_share_of_raw_total']:.1f}% of raw total)")
        print(f"  total  raw -> censored    : {raw['sum']/1000:.0f}s -> {cen['sum']/1000:.0f}s")
        print(f"  mean   raw -> censored    : {raw['mean']:.1f} -> {cen['mean']:.1f} ms "
              f"(raw inflated {infl_mean:+.0f}%)")
        print(f"  median raw -> censored    : {raw['p50']:.1f} -> {cen['p50']:.1f} ms "
              f"(raw inflated {infl_p50:+.0f}%)")

        if args.dry_run:
            continue

        d = OUTP / proj / "scalability"
        d.mkdir(parents=True, exist_ok=True)
        flagged = res.pop("_flagged")
        res["_meta"] = {"project": proj, "layer": layer,
                        "source": str(path.relative_to(ROOT)),
                        "scope": "live build wall-clock only; does NOT affect "
                                 "prediction_latency.json or build_complexity.json "
                                 "(both controlled replays)"}
        json.dump(res, open(d / f"timing_censored_{layer}.json", "w"), indent=1)
        with open(d / f"timing_flagged_{layer}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flagged[0].keys()) if flagged
                               else ["idx"])
            w.writeheader()
            w.writerows(flagged)
        print(f"  -> {d/f'timing_censored_{layer}.json'}")
        print(f"  -> {d/f'timing_flagged_{layer}.csv'} (audit trail)")

    if not args.dry_run:
        print("\nReport MEDIAN per-commit cost in the paper where possible: it is "
              "near-invariant to these stalls. If a TOTAL is required, use the "
              "censored total and footnote the policy above.")


if __name__ == "__main__":
    main()
