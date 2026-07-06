"""
E2 -- Per-commit growth & modification analysis (read-only, no rebuild).

Recovers the online growth trajectory of every final-methodology layer directly
from the built DB, by attributing each delta edge (ADDS/REMOVES/UPDATES/MOVES,
and CSTG MENTIONS) to its source Commit and ordering commits by author_ts (the
true online arrival order). From these per-commit deltas we derive:

  * cumulative nodes/edges vs commit index and vs calendar year (concept drift)
  * the per-commit change-size distribution (adds / removes / updates / moves)
  * buggy-vs-benign change-size separation, with a Mann-Whitney U test and a
    bootstrap CI on the mean-size ratio (validates the paper's ~860 vs ~299 claim)
  * net growth vs churn (removes as a fraction of adds)

Nothing here rebuilds the KG: it is one grouped read-only Cypher pass per layer.

Output: outputs/scalability/growth.json
        (+ compact per-commit arrays for the figures, keyed by layer)

Run:  python scalability/collect_growth_stats.py
"""
import numpy as np
from scipy import stats

import _common as C

# layers whose per-commit growth we track: the final structural family + CSTG.
GROWTH_LAYERS = [("ast", "ASTNode"), ("cfg", "CFGNode"),
                 ("dfg", "DFGNode"), ("pdg", "PDGNode"), ("seq", "SEQNode")]


def commit_order(s):
    """All labelled commits in online arrival order (author_ts, id)."""
    rows = s.run(
        "MATCH (c:Commit {in_jit:true}) "
        "RETURN c.id AS id, c.author_ts AS ts, coalesce(c.buggy,false) AS buggy, "
        "coalesce(c.jit_year, date(datetime({epochSeconds:toInteger(c.author_ts)})).year) "
        "AS year ORDER BY c.author_ts, c.id").data()
    ids = [r["id"] for r in rows]
    idx = {c: i for i, c in enumerate(ids)}
    y = np.array([bool(r["buggy"]) for r in rows])
    year = np.array([int(r["year"]) if r["year"] is not None else -1 for r in rows])
    return ids, idx, y, year


def per_commit_deltas(s, lbl, idx):
    """For one structural layer: per-commit counts of each delta-edge type,
    aligned to the online commit order (0 for commits that touched no node)."""
    n = len(idx)
    counts = {r: np.zeros(n) for r in C.DELTA_RELS}
    for rel in C.DELTA_RELS:
        for r in s.run(
                f"MATCH (c:Commit {{in_jit:true}})-[e:{rel}]->(:{lbl}) "
                f"RETURN c.id AS id, count(e) AS k"):
            i = idx.get(r["id"])
            if i is not None:
                counts[rel][i] = r["k"]
    return counts


def cstg_mentions_per_commit(s, idx):
    n = len(idx)
    m = np.zeros(n)
    for r in s.run(
            "MATCH (c:Commit {in_jit:true})-[e:MENTIONS]->(:Term) "
            "RETURN c.id AS id, count(e) AS k"):
        i = idx.get(r["id"])
        if i is not None:
            m[i] = r["k"]
    return m


def bootstrap_ratio_ci(buggy_sz, benign_sz, iters=5000, seed=0):
    """Bootstrap CI on the ratio of mean change-size (buggy / benign)."""
    rng = np.random.default_rng(seed)
    b = np.asarray(buggy_sz, float); g = np.asarray(benign_sz, float)
    if b.size == 0 or g.size == 0:
        return dict(ratio=float("nan"), lo=float("nan"), hi=float("nan"))
    rs = np.empty(iters)
    for k in range(iters):
        bb = b[rng.integers(0, b.size, b.size)].mean()
        gg = g[rng.integers(0, g.size, g.size)].mean()
        rs[k] = bb / gg if gg > 0 else np.nan
    rs = rs[~np.isnan(rs)]
    return dict(ratio=float(b.mean() / g.mean()) if g.mean() > 0 else float("nan"),
                lo=float(np.percentile(rs, 2.5)), hi=float(np.percentile(rs, 97.5)))


def layer_growth(name, counts, y, year):
    """Assemble the growth summary + compact arrays for one layer."""
    total = sum(counts.values())               # per-commit total delta edges
    adds = counts["ADDS"]; removes = counts["REMOVES"]
    cum_edges = np.cumsum(total)
    # net node growth proxy = ADDS - REMOVES (nodes are soft-removed, kept as history)
    cum_net = np.cumsum(adds - removes)
    touched = total > 0

    buggy_sz = total[y & touched]; benign_sz = total[(~y) & touched]
    mw = stats.mannwhitneyu(buggy_sz, benign_sz, alternative="greater") \
        if buggy_sz.size and benign_sz.size else None
    ratio = bootstrap_ratio_ci(buggy_sz, benign_sz)

    # per-year aggregation (concept-drift overlay)
    years = sorted(set(int(v) for v in year if v > 0))
    by_year = {}
    for yy in years:
        m = year == yy
        by_year[str(yy)] = dict(
            n_commits=int(m.sum()),
            bug_rate=float(y[m].mean()) if m.any() else 0.0,
            delta_edges=float(total[m].sum()),
            mean_delta=float(total[m].mean()) if m.any() else 0.0)

    summary = dict(
        layer=name,
        total_delta_edges=int(total.sum()),
        by_rel={r: int(counts[r].sum()) for r in C.DELTA_RELS},
        churn_ratio=float(removes.sum() / adds.sum()) if adds.sum() else 0.0,
        commits_touching=int(touched.sum()),
        delta_per_commit=C.describe(total[touched]),
        delta_per_buggy=C.describe(buggy_sz),
        delta_per_benign=C.describe(benign_sz),
        buggy_vs_benign=dict(
            mannwhitney_U=float(mw.statistic) if mw else float("nan"),
            mannwhitney_p=float(mw.pvalue) if mw else float("nan"),
            mean_size_ratio=ratio),
        by_year=by_year,
    )
    arrays = dict(
        total=total.astype(int).tolist(),
        cum_edges=cum_edges.astype(float).tolist(),
        cum_net_nodes=cum_net.astype(float).tolist(),
    )
    return summary, arrays


def main():
    ok, status = C.assert_db_complete()
    print(f"DB complete: {ok}  {status}")
    d = C.driver()
    out = {"_meta": {"scope": "final V4 layers + CSTG; per-commit growth in "
                             "author_ts (online) order"}}
    arrays_all = {}
    with C.read_session(d) as s:
        ids, idx, y, year = commit_order(s)
        out["_meta"]["n_commits"] = len(ids)
        out["_meta"]["bug_rate"] = float(y.mean())
        out["_meta"]["years"] = sorted(set(int(v) for v in year if v > 0))
        for vid, lbl in GROWTH_LAYERS:
            print(f"growth: {lbl} ...")
            counts = per_commit_deltas(s, lbl, idx)
            summ, arr = layer_growth(vid, counts, y, year)
            out[vid] = summ
            arrays_all[vid] = arr
            rr = summ["buggy_vs_benign"]["mean_size_ratio"]
            print(f"  {vid:<4} total={summ['total_delta_edges']:>9,} "
                  f"churn={summ['churn_ratio']:.2f} "
                  f"buggy/benign size ratio={rr['ratio']:.2f} "
                  f"[{rr['lo']:.2f},{rr['hi']:.2f}] "
                  f"MW p={summ['buggy_vs_benign']['mannwhitney_p']:.1e}")
        print("growth: CSTG MENTIONS ...")
        men = cstg_mentions_per_commit(s, idx)
        touched = men > 0
        out["cstg"] = dict(
            layer="cstg",
            total_mentions=int(men.sum()),
            commits_touching=int(touched.sum()),
            mentions_per_commit=C.describe(men[touched]),
            mentions_per_buggy=C.describe(men[y & touched]),
            mentions_per_benign=C.describe(men[(~y) & touched]),
        )
        arrays_all["cstg"] = dict(
            total=men.astype(int).tolist(),
            cum_edges=np.cumsum(men).astype(float).tolist())
        print(f"  cstg total MENTIONS={out['cstg']['total_mentions']:,}")
    d.close()

    # compact arrays live in a separate file (kept out of the summary json)
    out["_arrays_file"] = "growth_arrays.json"
    C.save_json({"y": y.astype(int).tolist(), "year": year.astype(int).tolist(),
                 "layers": arrays_all}, "growth_arrays.json")
    p = C.save_json(out, "growth.json")
    print(f"\nsaved -> {p}")

    # sanity anchor: AST buggy/benign ratio should reproduce the paper's ~860/299
    r = out["ast"]["buggy_vs_benign"]["mean_size_ratio"]["ratio"]
    print(f"anchor: AST buggy/benign delta-size ratio = {r:.2f} "
          f"(paper reports ~860/299 = 2.9)")


if __name__ == "__main__":
    main()
