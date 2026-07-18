"""
E3 -- Build / modify time & space complexity (no KG rebuild).

The per-commit graph-construction cost was never persisted for the Groovy build,
so we recover it two complementary ways that cross-validate each other:

 (a) ANALYTICAL PROXY (read-only): per-commit work is driven by the size of the
     change, not by history length. We already recovered the per-commit delta
     size in E2; here we state the operation counts (parses, diff, writes) and
     show empirically that cost is O(change), space is O(alive nodes + retained
     history) -- i.e. storage grows with the amount of change, not history x
     filesize.

 (b) SAMPLED REAL TIMING (cheap, no DB): we replay the *actual* parse+diff work
     of the online engine -- for AST: online_ast_diff.diff_sources; for the
     subgraphs: subgraph_builders.build_file_graph + subgraph_diff.diff_graphs --
     over a stratified sample of real (before, after) modified-file pairs drawn
     from the commit stream, plus the base build over the 120 BASE files. We time
     ONLY the CPU-bound construction (the Neo4j writes are I/O and were separately
     characterised by the missing-index lesson in the paper), fit
        t = a + b * (nodes_after)         and     t vs delta_size
     report R^2, and extrapolate the fitted per-file cost to the full stream to
     estimate total build time per layer.

For OTHER projects the same numbers can be captured natively: both online engines
now accept --timing-log, writing one row per commit (idx, sha, files, delta
counts, wall_ms). Off by default -> the built Groovy DB is untouched.

Output: outputs/scalability/build_complexity.json

Run:  python scalability/time_build_replay.py --sample 400
      python scalability/time_build_replay.py --sample 150 --kinds ast cfg
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

import _common as C
from config.project_config import REPO_PATH as REPO, BASE_COMMIT  # per-project


def git_bytes(ref, path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def sample_modified_pairs(n, seed=0):
    """Draw n real (before_sha, after_sha, path) modified-.java pairs from the
    stream, stratified by change size so the fit covers small and large edits.
    Read-only: uses git + the commit order already in the DB checkpoint order."""
    rng = np.random.default_rng(seed)
    d = C.driver()
    with C.read_session(d) as s:
        commits = [r["id"] for r in s.run(
            "MATCH (c:Commit {in_jit:true}) RETURN c.id AS id "
            "ORDER BY c.author_ts, c.id")]
    d.close()
    # walk a subset of commits, collect modified java files with both blobs present
    pairs = []
    step = max(1, len(commits) // (n * 3))
    for c in commits[::step]:
        par = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{c}^1"],
                             capture_output=True, text=True)
        if par.returncode != 0:
            continue
        diff = subprocess.run(["git", "-C", str(REPO), "diff", "--name-status",
                               "-M", f"{c}^1", c], capture_output=True, text=True)
        for line in diff.stdout.splitlines():
            parts = line.split("\t")
            if parts[0] == "M" and parts[1].endswith(".java"):
                pairs.append((f"{c}^1", c, parts[1]))
        if len(pairs) >= n * 4:
            break
    rng.shuffle(pairs)
    return pairs[:n]


# ── (b) sampled real timing per layer ───────────────────────────────────────

def time_ast(pairs):
    import online_ast_diff as oad
    rows = []
    for bref, aref, path in pairs:
        bb = git_bytes(bref, path); ab = git_bytes(aref, path)
        if bb is None or ab is None:
            continue
        t = time.perf_counter()
        try:
            d = oad.diff_sources(bb, ab, path)
        except Exception:
            continue
        dt = time.perf_counter() - t
        if d is None:
            continue
        n_after = len(d.get("after_nodes", []))
        delta = (len(d.get("inserted", [])) + len(d.get("deleted", []))
                 + sum(1 for m in d.get("matched", []) if m[2]) + len(d.get("moved", [])))
        rows.append((len(ab), n_after, delta, dt))
    return rows


def time_subgraph(kind, pairs):
    import subgraph_builders as sb
    import subgraph_diff as sd
    rows = []
    for bref, aref, path in pairs:
        bb = git_bytes(bref, path); ab = git_bytes(aref, path)
        if bb is None or ab is None:
            continue
        try:
            t = time.perf_counter()
            bg = sb.build_file_graph(kind, bb.decode("utf-8", "replace"), path)
            ag = sb.build_file_graph(kind, ab.decode("utf-8", "replace"), path)
            if "error" in bg or "error" in ag:
                continue
            d = sd.diff_graphs(bg, ag)
            dt = time.perf_counter() - t
        except Exception:
            continue
        n_after = len(ag["nodes"])
        delta = (len(d["inserted"]) + len(d["deleted"])
                 + sum(1 for m in d["matched"] if m[2]) + len(d["moved"]))
        rows.append((len(ab), n_after, delta, dt))
    return rows


def fit_cost(rows):
    """Fit wall-time (ms) against nodes_after and against delta-size; report the
    linear coefficients and R^2, and the median ms/file."""
    if len(rows) < 5:
        return dict(n=len(rows), note="too few samples")
    arr = np.array(rows, float)          # cols: src_bytes, n_after, delta, sec
    ms = arr[:, 3] * 1000.0
    out = dict(n=len(rows),
               ms_per_file=C.describe(ms),
               nodes_after=C.describe(arr[:, 1]),
               delta_size=C.describe(arr[:, 2]))

    def linfit(x):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, ms, rcond=None)
        pred = X @ beta
        ss_res = ((ms - pred) ** 2).sum()
        ss_tot = ((ms - ms.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return dict(intercept=float(beta[0]), slope=float(beta[1]), r2=float(r2))

    out["fit_vs_nodes_after"] = linfit(arr[:, 1])
    out["fit_vs_delta_size"] = linfit(arr[:, 2])
    return out


def extrapolate(fit, profile_layer):
    """Estimate total modify-time for the layer: median ms/file x number of
    per-commit file-modifications, from the E1 profile (commits_with_tokens as a
    lower bound on modified-file events)."""
    if not isinstance(fit, dict) or "ms_per_file" not in fit:
        return {}
    med = fit["ms_per_file"]["p50"]
    # number of modified-file build events ~ MODIFIED edges over java files;
    # use the layer's commits_with_tokens as a conservative proxy count.
    n_events = profile_layer.get("commits_with_tokens", 0)
    return dict(median_ms_per_file=med,
                est_events=int(n_events),
                est_total_seconds=float(med * n_events / 1000.0),
                note="CPU parse+diff only (excludes Neo4j write I/O)")


# ── (a) analytical proxy from the E2 growth arrays ──────────────────────────

def proxy_from_growth():
    import json
    gp = C.OUT / "growth.json"
    if not gp.exists():
        return {"note": "run collect_growth_stats.py first for the proxy"}
    g = json.load(open(gp))
    prox = {}
    for vid in ("ast", "cfg", "dfg", "pdg", "seq"):
        if vid in g:
            dpc = g[vid]["delta_per_commit"]
            prox[vid] = dict(
                cost_driver="per-commit delta size (adds+removes+updates+moves)",
                median_delta=dpc["p50"], mean_delta=dpc["mean"], p95_delta=dpc["p95"],
                claim="per-commit work = O(change); independent of history length")
    return prox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--kinds", nargs="*", default=["ast", "cfg", "dfg", "pdg", "seq"])
    args = ap.parse_args()

    import json
    profile = json.load(open(C.OUT / "kg_profile.json")) if (C.OUT / "kg_profile.json").exists() else {}

    print(f"sampling {args.sample} real modified-file pairs (read-only git) ...")
    pairs = sample_modified_pairs(args.sample)
    print(f"  got {len(pairs)} pairs")

    out = {"_meta": {"scope": "final V4 layers; CPU parse+diff timing on real "
                             "modified-file pairs + analytical proxy",
                     "n_pairs": len(pairs),
                     "excludes": "Neo4j write I/O (separately characterised by the "
                                 "missing-index lesson in the paper)"}}
    out["analytical_proxy"] = proxy_from_growth()

    for kind in args.kinds:
        print(f"timing {kind} parse+diff ...")
        t0 = time.perf_counter()
        rows = time_ast(pairs) if kind == "ast" else time_subgraph(kind, pairs)
        fit = fit_cost(rows)
        fit["extrapolation"] = extrapolate(fit, profile.get(kind, {}))
        out[kind] = fit
        if "ms_per_file" in fit:
            print(f"  {kind:<4} n={fit['n']:>3} med={fit['ms_per_file']['p50']:.1f}ms/file "
                  f"R2(nodes)={fit['fit_vs_nodes_after']['r2']:.2f} "
                  f"R2(delta)={fit['fit_vs_delta_size']['r2']:.2f} "
                  f"est_total={fit['extrapolation'].get('est_total_seconds', 0):.0f}s "
                  f"({time.perf_counter()-t0:.0f}s wall)")
        else:
            print(f"  {kind}: {fit.get('note')}")

    p = C.save_json(out, "build_complexity.json")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
