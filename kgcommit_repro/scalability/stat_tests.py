"""
E5 -- Statistical significance of the final (V4) comparisons (no rebuild, no DB
rebuild; graph family replayed in-memory from the cached hub dictionaries).

We attach confidence and significance to the paper's headline claims:
  (Q-subgraph) on the fixed five methods, does graph richness help? i.e. is
               Core+AST (and the final Core+AST+CSTG) significantly better than Core?
  (Q-method)   on the final graph, is PPR (the best cell) significantly ahead?
  (Q-fusion)   is the deployed F+G = RN+PPR+CSTG significantly better than F=RN+PPR
               and than the best single method?

Method: we regenerate per-commit out-of-sample scores (y, p) for every
(method, graph) by replaying inference/run_final_experiments.run_graph on the
cached graph family -- run_graph RETURNS full-length per-commit `pred`, so no DB
and no KG rebuild is needed. On these paired (y,p) vectors we compute:

  * bootstrap 95% CIs for ROC-AUC, PR-AUC, MCC, G-Mean (per method/graph)
  * DeLong's test for paired ROC-AUC differences (exact AUC covariance)
  * a paired bootstrap test for PR-AUC / MCC differences
  * Cliff's delta effect size on the score separation
  * Holm-Bonferroni correction across each family of comparisons

Pure numpy/scipy; no statsmodels / powerlaw dependency.

Output: outputs/scalability/significance.json

Run:  python scalability/stat_tests.py            # full replay + tests
      python scalability/stat_tests.py --quick    # fewer bootstrap iters
"""
import argparse
import numpy as np
from scipy import stats

import _common as C


# ── metric point estimates ──────────────────────────────────────────────────

def roc_auc(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    n1 = y.sum(); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, p))


def mcc_at(y, p, thr=None):
    from sklearn.metrics import matthews_corrcoef
    y = np.asarray(y, int)
    if thr is None:
        thr = np.quantile(p, 1 - y.mean())      # rate-matched operating point
    return float(matthews_corrcoef(y, (p >= thr).astype(int)))


def gmean_at(y, p, thr=None):
    y = np.asarray(y, int)
    if thr is None:
        thr = np.quantile(p, 1 - y.mean())
    yh = (p >= thr).astype(int)
    tp = ((yh == 1) & (y == 1)).sum(); fn = ((yh == 0) & (y == 1)).sum()
    tn = ((yh == 0) & (y == 0)).sum(); fp = ((yh == 1) & (y == 0)).sum()
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return float(np.sqrt(max(rec, 0) * max(spec, 0)))


METRICS = {"ROC_AUC": roc_auc, "PR_AUC": pr_auc, "MCC": mcc_at, "G_Mean": gmean_at}


# ── bootstrap CIs (paired resampling over commits) ──────────────────────────

def bootstrap_ci(y, p, fn, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = y.size; vals = np.empty(iters)
    for k in range(iters):
        b = rng.integers(0, n, n)
        vals[k] = fn(y[b], p[b])
    vals = vals[~np.isnan(vals)]
    return dict(est=float(fn(y, p)),
                lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)))


def paired_bootstrap_diff(y, pa, pb, fn, iters=2000, seed=0):
    """Paired bootstrap test of H0: metric(a) == metric(b). Two-sided p from the
    bootstrap distribution of the difference; also returns the CI on the diff."""
    rng = np.random.default_rng(seed)
    n = y.size; diffs = np.empty(iters)
    for k in range(iters):
        b = rng.integers(0, n, n)
        diffs[k] = fn(y[b], pa[b]) - fn(y[b], pb[b])
    diffs = diffs[~np.isnan(diffs)]
    obs = fn(y, pa) - fn(y, pb)
    # two-sided p: fraction of bootstrap diffs on the other side of 0
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(diff=float(obs), lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)), p=float(min(p, 1.0)))


# ── DeLong's exact paired ROC-AUC test ──────────────────────────────────────

def _midrank(x):
    order = np.argsort(x); x_s = x[order]
    n = len(x); tr = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_s[j] == x_s[i]:
            j += 1
        tr[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n); out[order] = tr
    return out


def delong_test(y, pa, pb):
    """DeLong paired test for two correlated ROC-AUCs. Returns AUCs, z, two-sided p.
    Standard fast implementation (Sun & Xu 2014)."""
    y = np.asarray(y, int)
    pos = y == 1; neg = ~pos
    m = int(pos.sum()); n = int(neg.sum())
    if m == 0 or n == 0:
        return dict(auc_a=float("nan"), auc_b=float("nan"), z=float("nan"),
                    p=float("nan"))
    preds = [pa, pb]
    aucs = np.empty(2); v01 = np.empty((2, m)); v10 = np.empty((2, n))
    for k, pr in enumerate(preds):
        xp = np.asarray(pr, float)[pos]; xn = np.asarray(pr, float)[neg]
        tz = _midrank(np.concatenate([xp, xn]))
        tx = _midrank(xp); ty = _midrank(xn)
        aucs[k] = (tz[:m].sum() - tx.sum()) / (m * n) + 0.5 - 0.0  # via structural comps
        v01[k] = (tz[:m] - tx) / n
        v10[k] = 1.0 - (tz[m:] - ty) / m
    # recompute AUC cleanly (the structural form above can drift); use Mann-Whitney
    aucs = np.array([roc_auc(y, pa), roc_auc(y, pb)])
    s01 = np.cov(v01); s10 = np.cov(v10)
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), z=float(z), p=float(p))


def cliffs_delta(y, p):
    """Cliff's delta between buggy vs benign scores (rank separation effect size)."""
    a = np.asarray(p, float)[np.asarray(y, int) == 1]
    b = np.asarray(p, float)[np.asarray(y, int) == 0]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # via Mann-Whitney U (O(n log n)): delta = 2U/(mn) - 1
    U = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * U / (a.size * b.size) - 1)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (returns array aligned to input order)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p); m = p.size
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    return adj


# ── replay the final graph family to get paired per-commit (y, p) ────────────

def replay_predictions(quick=False):
    """Return preds[graph][method] = full-length per-commit scores, plus y and the
    evaluation slice ev (warmup..N). Uses the cached hub dicts; NO live DB."""
    import run_final_experiments as rfe
    b = C.load_final_graph_cache()
    cids, y, files, devs, tok, cstg = (b["cids"], b["y"], b["files"], b["devs"],
                                       b["tok"], b["cstg"])
    graphs = ["core", "ast", "final"] if quick else C.FINAL_GRAPHS
    preds = {}
    for g in graphs:
        _, pred = rfe.run_graph(g, cids, y, files, devs, tok, cstg)
        preds[g] = {m: np.asarray(pred[m], float) for m in C.FINAL_METHODS}
        print(f"  replayed graph {g}")
    from online_infer import WARMUP_FRAC
    W = int(len(cids) * WARMUP_FRAC)
    ev = np.arange(W, len(cids))
    return preds, np.asarray(y, int), ev


def fuse(preds_graph, methods, y, ev):
    """Prequential-free convenience: stack chosen methods' scores with a single
    logistic regression fit on the warmup and applied to ev -- a proxy for the
    deployed fusion, sufficient for a paired significance comparison of F vs F+G."""
    from sklearn.linear_model import LogisticRegression
    W = ev[0]
    X = np.column_stack([np.nan_to_num(preds_graph[m], nan=y[:W].mean())
                         for m in methods])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[:W], y[:W])
    return clf.predict_proba(X)[:, 1]


def _final_fusion_effect():
    """Authoritative deployed-model F -> F+G metric change from the fusion run."""
    import pickle
    p = C.OUT.parent / "final_fusion_results.pkl"
    if not p.exists():
        return {}
    d = pickle.load(open(p, "rb"))
    try:
        F = d["part2"]["F"]["metrics"]; FG = d["part2"]["F+G"]["metrics"]
    except Exception:
        return {}
    keys = ["ROC_AUC", "PR_AUC", "Macro_F1", "G_Mean", "Buggy_F1", "MCC"]
    return {k: {"F": float(F[k]), "F+G": float(FG[k]),
                "delta": float(FG[k] - F[k])} for k in keys if k in F and k in FG}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    iters = 800 if args.quick else 2000

    print("replaying final graph family (cached, no DB rebuild) ...")
    preds, y, ev = replay_predictions(quick=args.quick)
    yev = y[ev]
    out = {"_meta": {"scope": "final V4 methodology; paired significance on "
                             "replayed per-commit scores", "n_eval": int(ev.size),
                     "bootstrap_iters": iters}}

    # ---- CIs for every (method, graph) cell ----
    cells = {}
    for g in preds:
        for m in C.FINAL_METHODS:
            p = np.clip(np.nan_to_num(preds[g][m][ev], nan=yev.mean()), 0, 1)
            cells[f"{g}:{m}"] = {
                name: bootstrap_ci(yev, p, fn, iters=iters)
                for name, fn in METRICS.items()}
            cells[f"{g}:{m}"]["cliffs_delta"] = cliffs_delta(yev, p)
    out["cells"] = cells
    print(f"CIs computed for {len(cells)} (method,graph) cells")

    # ---- Q-subgraph: does richness help? Core vs Core+AST vs final, per method ----
    rich = {}
    if "core" in preds and "ast" in preds:
        pv = []
        for m in C.FINAL_METHODS:
            pc = np.clip(np.nan_to_num(preds["core"][m][ev], nan=yev.mean()), 0, 1)
            pa = np.clip(np.nan_to_num(preds["ast"][m][ev], nan=yev.mean()), 0, 1)
            dl = delong_test(yev, pa, pc)
            rich[f"{m}: AST>Core (ROC)"] = dl; pv.append(dl["p"])
            if "final" in preds:
                pf = np.clip(np.nan_to_num(preds["final"][m][ev], nan=yev.mean()), 0, 1)
                dlf = delong_test(yev, pf, pc)
                rich[f"{m}: final>Core (ROC)"] = dlf; pv.append(dlf["p"])
        adj = holm([rich[k]["p"] for k in rich])
        for k, a in zip(list(rich), adj):
            rich[k]["p_holm"] = float(a)
    out["richness_vs_core"] = rich

    # ---- Q-method: on the final graph, is PPR ahead of the others? ----
    method_cmp = {}
    if "final" in preds:
        pv = []
        pppr = np.clip(np.nan_to_num(preds["final"]["PPR"][ev], nan=yev.mean()), 0, 1)
        for m in [x for x in C.FINAL_METHODS if x != "PPR"]:
            pm = np.clip(np.nan_to_num(preds["final"][m][ev], nan=yev.mean()), 0, 1)
            d = delong_test(yev, pppr, pm)
            pr = paired_bootstrap_diff(yev, pppr, pm, pr_auc, iters=iters)
            method_cmp[f"PPR>{m} (final)"] = {"roc_delong": d, "prauc_boot": pr}
            pv.append(d["p"])
        adj = holm(pv)
        for k, a in zip(list(method_cmp), adj):
            method_cmp[k]["roc_delong"]["p_holm"] = float(a)
    out["ppr_vs_others_final"] = method_cmp

    # ---- Q-fusion: F=RN+PPR vs the best single method, on the final graph ----
    # (The deployed F+G=RN+PPR+CSTG semantic effect is the authoritative F->F+G
    #  delta from run_final_fusion.py; we surface it from that pickle rather than a
    #  non-prequential proxy, and here only test F vs the best single method with a
    #  proper paired test.)
    fus = {}
    if "final" in preds:
        pg = preds["final"]
        pF = np.clip(fuse(pg, ["RN", "PPR"], y, ev)[ev], 0, 1)
        best_single = max(C.FINAL_METHODS,
                          key=lambda m: roc_auc(yev, np.nan_to_num(pg[m][ev], nan=yev.mean())))
        pbs = np.clip(np.nan_to_num(pg[best_single][ev], nan=yev.mean()), 0, 1)
        fus["F=RN+PPR"] = {name: bootstrap_ci(yev, pF, fn, iters=iters)
                           for name, fn in METRICS.items()}
        fus["F>best_single (ROC)"] = {"best_single": best_single,
                                      **delong_test(yev, pF, pbs)}
        fus["F+G_semantic_effect"] = _final_fusion_effect()
        fus["_note"] = ("F is a warmup-fit stacking of RN+PPR on the final KG for a "
                        "paired test vs the best single method; the deployed "
                        "F+G=RN+PPR+CSTG effect is read from final_fusion_results.pkl.")
    out["fusion"] = fus

    p = C.save_json(out, "significance.json")
    print(f"\nsaved -> {p}")
    # headline print
    if rich:
        print("richness (Holm-adj p):")
        for k, v in rich.items():
            print(f"  {k:<26} dAUC={v['auc_a']-v['auc_b']:+.3f}  p_holm={v.get('p_holm', v['p']):.1e}")


if __name__ == "__main__":
    main()
