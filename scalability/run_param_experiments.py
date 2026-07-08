"""
Parameter sensitivity experiments for the Discussion section of the paper.

Scope (fixed): the final deployed graph *Core+AST+CSTG* only; the five graph-native
inference methods RN/PPR/LP/DW/KGE and the deployed fusion F = RN+PPR; the seven
headline metrics. The CSTG *classifier channel* G is NOT used (F is graph-native).
Each experiment sweeps ONE setup parameter and holds everything else at its default.

Parameters swept:
  K     warmup fraction              -- how much history to fit before scoring
  M     label-availability gap       -- labels of the M commits just before the
                                        block are withheld (they arrive late in
                                        practice); structure is still visible
  ROLL  smoothing / rolling window   -- the trailing window each stream point
                                        averages over (a reporting choice)
  l     online CSTG-hub refresh      -- rebuild the CSTG Term-hub edge weights from
                                        past-only NPMI every l commits (the graph
                                        the five methods walk is refreshed; the G
                                        classifier is still not used)

For every parameter value we report BOTH the whole-stream 7-metric scores (per
method + fusion) and rolling-trajectory summaries (mean / last / area) so the
Discussion can show tables and stream figures.

Reuses the cached final-graph bundle (no KG rebuild, no live DB) and the reference
inference code in inference/. Output: outputs/param_experiments/<param>.json

Run:
  python scalability/run_param_experiments.py --param K
  python scalability/run_param_experiments.py --param all --quick
"""
import argparse
import math
import pickle
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

import _common as C

# reference inference code (final methodology)
import kg_methods as km
import run_final_experiments as rfe
from online_infer import WARMUP_FRAC, BLOCK
from online_jit import final_metrics, online_decisions, _best_threshold

METHODS = C.FINAL_METHODS                 # RN, PPR, LP, DW, KGE
FUSION = ["RN", "PPR"]                     # deployed F (no G, no M)
BASELINES = ["JIT_LR", "Naive"]           # labels-only reference predictors
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]


def load_jit_metrics():
    """The 12 standardised ApacheJIT change metrics per commit, aligned to the
    cached final-graph commit order (from online_jit_streams_v5.pkl)."""
    S = pickle.load(open(C.ROOT / "outputs" / "online_jit_streams_v5.pkl", "rb"))
    return np.asarray(S["Xms"], float)        # (Nc, 12), already standardised on warmup


def baseline_scores(Xms, y, Nc, W, gap=0, block=BLOCK, refit=3):
    """JIT-metrics logistic regression and the naive prior-rate baseline, under the
    SAME warmup K, label gap M and block schedule as our methods: both fit only on
    the labelled past [0, i-gap). They have no graph, so the gap hits them purely
    through fewer training labels -- the honest real-world comparison."""
    from sklearn.linear_model import LogisticRegression
    Xms = np.nan_to_num(np.asarray(Xms, float), nan=0.0, posinf=0.0, neginf=0.0)
    pred = {b: np.full(Nc, np.nan) for b in BASELINES}
    i = max(W, 1); blk = 0; clf = None
    while i < Nc:
        j = min(Nc, i + block); idx = np.arange(i, j)
        lab_end = max(0, i - gap); lab = np.arange(lab_end)
        gr = float(y[lab].mean()) if len(lab) else float(y.mean())
        pred["Naive"][idx] = gr                       # prior bug-rate over labelled past
        if len(lab) and len(set(y[lab])) > 1:
            if blk % refit == 0:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                         solver="lbfgs").fit(Xms[lab], y[lab])
            pred["JIT_LR"][idx] = clf.predict_proba(Xms[idx])[:, 1] if clf else gr
        else:
            pred["JIT_LR"][idx] = gr
        i = j; blk += 1
    return pred

# defaults (mirror run_final_experiments)
DEF_WARMUP = WARMUP_FRAC                   # 0.30
DEF_BLOCK = BLOCK
DEF_ROLL = 800
DEF_REFIT_EMB = rfe.REFIT_EMB
DW_DIM, KGE_DIM = rfe.DW_DIM, rfe.KGE_DIM


# ── graph builders ──────────────────────────────────────────────────────────

def _hub_rows(g, cids, files, devs, tok, cstg):
    """Weighted memberships (i, hubkey, weight, relchar) for the final graph."""
    cidx = {c: i for i, c in enumerate(cids)}
    raw = []
    for c in cids:
        i = cidx[c]
        for f in files.get(c, ()):
            raw.append((i, "F:" + f, 1.0, "F"))
        if c in devs:
            raw.append((i, "D:" + devs[c], 1.0, "D"))
        for t, n in tok["ast"].get(c, ()):
            raw.append((i, "Tast:" + t, float(n), "T"))
        for t, n in cstg.get(c, ()):
            raw.append((i, "S:" + t, float(n), "S"))
    return raw, cidx


def _assemble(raw, Nc, term_scale=None):
    """Build C, P, edges from membership rows; optional per-hub weight scaling for
    CSTG term hubs (used by the l-refresh experiment)."""
    df = defaultdict(int)
    for i, h, w, r in raw:
        df[h] += 1
    hub = {}
    rows, cols, data, edges = [], [], [], []
    for i, h, w, r in raw:
        j = hub.setdefault(h, len(hub))
        wt = w * math.log(1.0 + Nc / df[h])
        if term_scale is not None and r == "S":
            wt *= term_scale.get(h, 1.0)
        rows.append(i); cols.append(j); data.append(wt)
        edges.append((i, rfe.REL[r], j))
    Nh = len(hub); Ntot = Nc + Nh
    Cm = sp.csr_matrix((data, (rows, cols)), shape=(Nc, max(Nh, 1)))
    B = sp.csr_matrix((data, (rows, [c + Nc for c in cols])), shape=(Ntot, Ntot))
    A = B + B.T
    deg = np.asarray(A.sum(0)).ravel(); deg[deg == 0] = 1.0
    P = A.multiply(sp.csr_matrix(1.0 / deg)).tocsr()
    E = np.array(edges) if edges else np.zeros((0, 3), int)
    return Cm, P, Nc, Nh, Ntot, E, hub


# ── the prequential run (parametrized) ──────────────────────────────────────

def run(cids, y, files, devs, tok, cstg,
        warmup=DEF_WARMUP, block=DEF_BLOCK, gap=0, refit_emb=DEF_REFIT_EMB,
        term_scale=None):
    """Prequential run of the five methods on the final graph with a label gap.

    gap M: at each block the graph STRUCTURE uses all past commits [0,i), but only
    the LABELS of commits in [0, i-gap) are revealed -- so seeds/fits/priors use
    `labeled` = past minus the last M commits. Scoring is still on the block.
    Returns pred{method->full-length scores} and Nc, W.
    """
    raw, _ = _hub_rows("final", cids, files, devs, tok, cstg)
    Cm, P, Nc, Nh, Ntot, edges, _ = _assemble(raw, len(cids), term_scale)
    W = int(Nc * warmup)
    pred = {m: np.full(Nc, np.nan) for m in METHODS}
    past_mask = np.zeros(Nc, bool)
    dw_clf = kge_clf = dwE = kgeE = None
    i = max(W, 1); blk = 0
    while i < Nc:
        j = min(Nc, i + block); idx = np.arange(i, j)
        lab_end = max(0, i - gap)                     # labels available up to here
        labeled = np.arange(lab_end)
        gr = float(y[labeled].mean()) if len(labeled) else float(y.mean())
        if len(labeled) and len(set(y[labeled])) > 1:
            pred["RN"][idx] = km.rn_scores(Cm, y, labeled, idx)
            pred["LP"][idx] = km.lp_scores(Cm, y, labeled, idx)
            bs = labeled[y[labeled] == 1]; gs = labeled[y[labeled] == 0]
            rb = km._ppr(P, list(bs), Ntot); rg = km._ppr(P, list(gs), Ntot)
            pred["PPR"][idx] = rb[idx] / (rb[idx] + rg[idx] + 1e-12)
            if blk % refit_emb == 0:
                past_mask[:] = False; past_mask[labeled] = True
                dwE = km.dw_embed(Cm, labeled, DW_DIM)
                dw_clf = km._lr().fit(dwE[labeled], y[labeled])
                kgeE = km.kge_embed(edges, Nc, Nh, past_mask, dim=KGE_DIM)
                kge_clf = km._lr().fit(kgeE[labeled], y[labeled])
            pred["DW"][idx] = dw_clf.predict_proba(dwE[idx])[:, 1] if dw_clf else gr
            pred["KGE"][idx] = kge_clf.predict_proba(kgeE[idx])[:, 1] if kge_clf else gr
        else:
            for m in METHODS:
                pred[m][idx] = gr
        i = j; blk += 1
    return pred, Nc, W


def run_with_l(cids, y, files, devs, tok, cstg, l_refresh, warmup=DEF_WARMUP,
               block=DEF_BLOCK):
    """l-sweep: rebuild CSTG Term-hub weights from PAST-ONLY NPMI every `l` commits.

    A term hub's weight is scaled by the mean NPMI of the term to the other terms
    co-mentioned by past commits -- so a larger l means staler semantic hub weights.
    Structure/labels are full past (gap=0). Still no G classifier.
    """
    raw, cidx = _hub_rows("final", cids, files, devs, tok, cstg)
    Nc = len(cids)
    # precompute per-commit term sets for the online NPMI
    terms_of = {i: [t for t, _ in cstg.get(c, ())] for i, c in enumerate(cids)}
    W = int(Nc * warmup)
    pred = {m: np.full(Nc, np.nan) for m in METHODS}
    past_mask = np.zeros(Nc, bool)
    dw_clf = kge_clf = dwE = kgeE = None
    tt = defaultdict(float); pair = defaultdict(float); seen = 0
    term_scale = {}

    def refresh_scale():
        adj = defaultdict(list)
        N = max(seen, 1)
        for (a, b), c in pair.items():
            if c < 2:
                continue
            pa, pb, pab = tt[a] / N, tt[b] / N, c / N
            if pab <= 0 or pa <= 0 or pb <= 0:
                continue
            npmi = math.log(pab / (pa * pb)) / (-math.log(pab))
            if npmi > 0:
                adj[a].append(npmi); adj[b].append(npmi)
        return {"S:" + t: (1.0 + float(np.mean(v))) for t, v in adj.items()}

    Cm, P, _, Nh, Ntot, edges, _ = _assemble(raw, Nc, term_scale)
    i = max(W, 1); blk = 0
    while i < Nc:
        # grow past-only NPMI counters up to i, refresh scale every l commits
        while seen < i:
            ts = sorted(set(terms_of.get(seen, [])))
            for a in ts:
                tt[a] += 1
            for a in range(len(ts)):
                for b in range(a + 1, len(ts)):
                    pair[(ts[a], ts[b])] += 1
            seen += 1
            if seen % l_refresh == 0:
                term_scale = refresh_scale()
                Cm, P, _, Nh, Ntot, edges, _ = _assemble(raw, Nc, term_scale)
        j = min(Nc, i + block); idx = np.arange(i, j); past = np.arange(i)
        gr = y[past].mean() if len(past) else 0.0
        pred["RN"][idx] = km.rn_scores(Cm, y, past, idx)
        pred["LP"][idx] = km.lp_scores(Cm, y, past, idx)
        bs = past[y[past] == 1]; gs = past[y[past] == 0]
        rb = km._ppr(P, list(bs), Ntot); rg = km._ppr(P, list(gs), Ntot)
        pred["PPR"][idx] = rb[idx] / (rb[idx] + rg[idx] + 1e-12)
        if blk % DEF_REFIT_EMB == 0 and len(set(y[past])) > 1:
            past_mask[:] = False; past_mask[past] = True
            dwE = km.dw_embed(Cm, past, DW_DIM)
            dw_clf = km._lr().fit(dwE[past], y[past])
            kgeE = km.kge_embed(edges, Nc, Nh, past_mask, dim=KGE_DIM)
            kge_clf = km._lr().fit(kgeE[past], y[past])
        pred["DW"][idx] = dw_clf.predict_proba(dwE[idx])[:, 1] if dw_clf else gr
        pred["KGE"][idx] = kge_clf.predict_proba(kgeE[idx])[:, 1] if kge_clf else gr
        i = j; blk += 1
    return pred, Nc, W


# ── fusion + metric assembly ────────────────────────────────────────────────

def fuse_F(pred, y, W, Nc, block=DEF_BLOCK, refit=3):
    """Prequential LR stacking of F = RN+PPR (deployed fusion; no G, no M)."""
    fill = float(y[:W].mean()) if W > 0 else float(y.mean())   # W=0 -> global rate
    Z = np.column_stack([np.nan_to_num(pred[m], nan=fill, posinf=fill, neginf=fill)
                         for m in FUSION])
    Z = np.nan_to_num(Z, nan=fill, posinf=fill, neginf=fill)
    p = np.full(Nc, np.nan); i = W; blk = 0
    clf = None
    while i < Nc:
        j = min(Nc, i + block); idx = np.arange(i, j)
        if clf is not None:
            p[idx] = clf.predict_proba(Z[idx])[:, 1]
        else:
            p[idx] = y[:i].mean() if i else 0.0
        if blk % refit == 0 and len(set(y[W:j])) > 1:
            clf = km._lr().fit(Z[W:j], y[W:j])
        i = j; blk += 1
    return p


def roll_traj(y_ev, p_ev, roll):
    """Rolling trajectory of the 7 metrics (window=roll, step=BLOCK)."""
    yhat = online_decisions(p_ev, y_ev)
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                 recall_score)
    idxs, cur = [], {m: [] for m in M7}
    n = len(y_ev)
    for j in range(BLOCK, n + 1, BLOCK):
        lo = max(0, j - roll)
        ys, ps, yh = y_ev[lo:j], p_ev[lo:j], yhat[lo:j]
        rec = recall_score(ys, yh, pos_label=1, zero_division=0)
        spec = recall_score(ys, yh, pos_label=0, zero_division=0)
        idxs.append(j)
        cur["Precision"].append(precision_score(ys, yh, pos_label=1, zero_division=0))
        cur["Recall"].append(rec)
        cur["Macro_F1"].append(f1_score(ys, yh, average="macro", zero_division=0))
        cur["Buggy_F1"].append(f1_score(ys, yh, pos_label=1, zero_division=0))
        cur["G_Mean"].append(float(np.sqrt(max(rec, 0) * max(spec, 0))))
        cur["AUC"].append(roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else np.nan)
        cur["ACC"].append(float((yh == ys).mean()))
    return idxs, cur


def summarize(pred, y, Nc, W, roll=DEF_ROLL, base_pred=None):
    """Whole-stream 7 metrics + rolling-traj summary for every method, F, and any
    baselines. `base_pred` (optional) adds labels-only reference rows."""
    ev = np.arange(W, Nc); yt = y[ev]
    out = {}
    allscores = dict(pred)
    allscores["F"] = fuse_F(pred, y, W, Nc)
    row_names = list(METHODS) + ["F"]
    if base_pred is not None:
        allscores.update(base_pred); row_names += BASELINES
    for m in row_names:
        p = np.clip(np.nan_to_num(allscores[m][ev], nan=yt.mean()), 0, 1)
        whole = final_metrics(yt, p)
        idxs, tr = roll_traj(yt, p, roll)
        tsumm = {k: dict(mean=float(np.nanmean(tr[k])),
                         last=float(tr[k][-1]) if tr[k] else float("nan"),
                         auc_traj=float(np.nanmean(tr[k])))    # area proxy = mean
                 for k in M7}
        out[m] = dict(whole={k: float(whole[k]) for k in M7},
                      traj=dict(idx=idxs, series=tr, summary=tsumm))
    return out


# ── sweeps ──────────────────────────────────────────────────────────────────

def sweep(param, quick=False):
    b = C.load_final_graph_cache()
    cids, y, files, devs, tok, cstg = (b["cids"], b["y"], b["files"], b["devs"],
                                       b["tok"], b["cstg"])
    grids = {
        "K":    [0.20, 0.30, 0.40, 0.50] if not quick else [0.30, 0.40],
        "M":    [0, 50, 100, 200, 400] if not quick else [0, 100],
        "ROLL": [200, 400, 800, 1600] if not quick else [400, 800],
        "l":    [100, 200, 400, 800] if not quick else [200, 400],
        # KM: warmup sweep (incl. very-low warmup) under a FIXED realistic gap M=200,
        # with labels-only baselines evaluated in the identical setup.
        "KM":   [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50] if not quick else [0.0, 0.10, 0.30],
    }[param]
    km_gap = 200
    with_base = (param == "KM")
    Xms = load_jit_metrics() if with_base else None
    res = {"_meta": dict(param=param, grid=grids, graph="Core+AST+CSTG",
                         methods=METHODS + ["F"] + (BASELINES if with_base else []),
                         fusion="RN+PPR (no G, no M)",
                         fixed_gap=(km_gap if param == "KM" else None),
                         baselines=(BASELINES if with_base else []),
                         defaults=dict(K=DEF_WARMUP, M=0, ROLL=DEF_ROLL, l=400))}
    for v in grids:
        print(f"[{param}={v}] ...", flush=True)
        if param == "K":
            pred, Nc, W = run(cids, y, files, devs, tok, cstg, warmup=v)
            res[str(v)] = summarize(pred, y, Nc, W)
        elif param == "M":
            pred, Nc, W = run(cids, y, files, devs, tok, cstg, gap=v)
            res[str(v)] = summarize(pred, y, Nc, W)
        elif param == "ROLL":
            # ROLL is a reporting window: compute preds once, summarize at each roll
            if "_pred" not in res:
                pred, Nc, W = run(cids, y, files, devs, tok, cstg)
                res["_pred"] = (pred, Nc, W)
            pred, Nc, W = res["_pred"]
            res[str(v)] = summarize(pred, y, Nc, W, roll=v)
        elif param == "l":
            pred, Nc, W = run_with_l(cids, y, files, devs, tok, cstg, l_refresh=v)
            res[str(v)] = summarize(pred, y, Nc, W)
        elif param == "KM":
            pred, Nc, W = run(cids, y, files, devs, tok, cstg, warmup=v, gap=km_gap)
            bpred = baseline_scores(Xms, y, Nc, W, gap=km_gap)
            res[str(v)] = summarize(pred, y, Nc, W, base_pred=bpred)
        bf = res[str(v)]["F"]["whole"]["Buggy_F1"]
        auc = res[str(v)]["F"]["whole"]["AUC"]
        print(f"   F: Buggy-F1={bf:.3f} AUC={auc:.3f}", flush=True)
    res.pop("_pred", None)
    C.OUT_PARAM.mkdir(parents=True, exist_ok=True)
    p = C.save_param_json(res, f"{param}.json")
    print(f"saved -> {p}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", choices=["K", "M", "ROLL", "l", "KM", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    params = ["K", "M", "ROLL", "l", "KM"] if args.param == "all" else [args.param]
    for pm in params:
        sweep(pm, quick=args.quick)


if __name__ == "__main__":
    main()
