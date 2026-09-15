"""
CSTG online GRAPH ablation: the five final graph-inference methods (RN, PPR, LP,
DW, KGE) run fully online (prequential) over the CSTG's own graph structure, one
column per *separated CSTG layer* and one for the full CSTG -- plus the CSTG
feature-classifier channel G as a comparison row. All seven headline metrics
(Precision, Recall, Buggy-F1, Macro-F1, G-Mean, AUC, MCC), same protocol as the
V4 final experiments (inference/run_final_experiments.py, run_final_fusion.py).

This answers "which parts of the CSTG's graph carry the signal, and how do the
graph-native methods compare to the feature-based G channel" -- the graph-native
counterpart to the batch text-feature ablation in collect_cstg_ablation.py.

CSTG hub-graph variants (columns), each sharing the Core file/dev relational layer
so the CSTG contribution is isolated:
  core        Core only (file + developer hubs; the floor)
  mentions    Core + (Commit)-[:MENTIONS]->(Term) term hubs, UNWEIGHTED (w=1)
  twidf       Core + MENTIONS term hubs, weighted by the stored TW-IDF (e.tw)
  cooccurs    twidf + (Term)-[:COOCCURS {npmi}]-(Term) term-term edges, so the
              walks/propagation/embeddings diffuse THROUGH the NPMI semantic graph
  full        cooccurs + term KIND typing on the edge relation (typed hubs) --
              the complete CSTG graph layer

Rows: RN, PPR, LP, DW, KGE (graph-native, online) + G (CSTG feature classifier,
online, from cstg_online_features -- graph-independent, shown once for reference).

Read-only: the CSTG hub data (MENTIONS/tw, COOCCURS/npmi, Term.kind) is pulled
once and cached to outputs/scalability/cstg_hub_cache.pkl; thereafter the whole
ablation replays in-memory (no live DB, no KG rebuild).

Output: outputs/scalability/cstg_graph_ablation.json

Run:  python scalability/cstg_online_graph_ablation.py
      python scalability/cstg_online_graph_ablation.py --quick   # core, twidf, full
"""
import argparse
import math
import pickle
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

import _common as C

VARIANTS = ["core", "mentions", "twidf", "cooccurs", "full"]
VARIANT_NAME = {"core": "Core", "mentions": "+MENTIONS (unweighted)",
                "twidf": "+MENTIONS (TW-IDF)", "cooccurs": "+COOCCURS (NPMI)",
                "full": "Full CSTG (+typed)"}
M7 = ["Precision", "Recall", "Buggy_F1", "Macro_F1", "G_Mean", "AUC", "MCC"]
# base relation ids; the 'full' variant expands the term relation by term KIND so
# the KGE (DistMult) learns a per-type relation embedding (typed semantic edges).
REL = {"F": 1, "D": 2, "S": 3}            # file, dev, cstg-term
KIND_REL = {"code": 3, "nl": 4, "error": 5, "bug": 6, "action": 7}


# ── read-only fetch of the CSTG hub data (cached) ────────────────────────────

def fetch_cstg_hubs(force=False):
    cache = C.OUT / "cstg_hub_cache.pkl"
    if cache.exists() and not force:
        return pickle.load(open(cache, "rb"))
    d = C.driver()
    mentions = defaultdict(list)     # commit -> [(term_id, tw)]
    kinds = {}                       # term_id -> kind
    cooccurs = []                    # [(term_a, term_b, npmi)]
    with C.read_session(d) as s:
        print("  fetching MENTIONS (tw) ...")
        for r in s.run("MATCH (c:Commit {in_jit:true})-[e:MENTIONS]->(t:Term) "
                       "RETURN c.id AS c, t.id AS t, coalesce(e.tw,1.0) AS tw, "
                       "coalesce(t.kind,'nl') AS k"):
            mentions[r["c"]].append((r["t"], float(r["tw"])))
            kinds[r["t"]] = r["k"]
        print("  fetching COOCCURS (npmi) ...")
        for r in s.run("MATCH (a:Term)-[e:COOCCURS]->(b:Term) "
                       "RETURN a.id AS a, b.id AS b, coalesce(e.npmi,0.0) AS w"):
            cooccurs.append((r["a"], r["b"], float(r["w"])))
    d.close()
    bundle = dict(mentions=dict(mentions), kinds=kinds, cooccurs=cooccurs)
    pickle.dump(bundle, open(cache, "wb"))
    return bundle


# ── build the commit-hub graph for one CSTG variant ──────────────────────────

def build_cstg_graph(variant, cids, files, devs, hubs):
    """Return C (commit x hub incidence), P (symmetric transition over commits +
    term hubs, incl. term-term COOCCURS edges for the cooccurs/full variants),
    Nc, Nh, N, and the typed edge list for KGE.

    Mirrors run_final_experiments.build_graph but (i) varies the CSTG term-hub
    weighting/typing per variant and (ii) can add Term-Term COOCCURS edges into
    the transition matrix so the walks diffuse through the NPMI graph."""
    cidx = {c: i for i, c in enumerate(cids)}; Nc = len(cids)
    mentions, kinds, cooccurs = hubs["mentions"], hubs["kinds"], hubs["cooccurs"]

    raw = []                          # (commit_i, hubkey, weight, rel_id)
    for c in cids:
        i = cidx[c]
        for f in files.get(c, ()):
            raw.append((i, "F:" + f, 1.0, REL["F"]))
        if c in devs:
            raw.append((i, "D:" + devs[c], 1.0, REL["D"]))
        if variant != "core":
            for t, tw in mentions.get(c, ()):
                w = 1.0 if variant == "mentions" else float(tw)
                # 'full' expands the term relation by term KIND (typed edges);
                # the other CSTG variants use one generic term relation.
                rel = KIND_REL.get(kinds.get(t, "nl"), REL["S"]) if variant == "full" \
                    else REL["S"]
                raw.append((i, "S:" + t, w, rel))

    # commit-hub incidence with IDF weighting (as in the final experiments)
    df = defaultdict(int)
    for i, h, w, r in raw:
        df[h] += 1
    hub = {}; rows = []; cols = []; data = []; edges = []
    for i, h, w, r in raw:
        j = hub.setdefault(h, len(hub))
        rows.append(i); cols.append(j)
        data.append(w * math.log(1.0 + Nc / df[h]))
        edges.append((i, r, j))
    Nh = len(hub); N = Nc + Nh
    Cm = sp.csr_matrix((data, (rows, cols)), shape=(Nc, max(Nh, 1)))

    # symmetric adjacency over [commits | hubs]
    B = sp.csr_matrix((data, (rows, [c + Nc for c in cols])), shape=(N, N))
    A = B + B.T

    # term-term COOCCURS edges (only where BOTH term hubs exist in this graph).
    # These enter the transition matrix P (so the walk/propagation methods diffuse
    # THROUGH the NPMI semantic graph) but NOT the KGE triple list: kge_embed models
    # (commit, rel, hub) triples only -- its first column must be a commit index, so
    # a term->term edge would be out of range. The diffusion effect is captured by
    # PPR/LP/DW via P; KGE reads the commit-term relations only.
    if variant in ("cooccurs", "full") and cooccurs:
        cr, cc, cd = [], [], []
        for a, b, w in cooccurs:
            ja = hub.get("S:" + a); jb = hub.get("S:" + b)
            if ja is None or jb is None or w <= 0:
                continue
            ua, ub = ja + Nc, jb + Nc
            cr += [ua, ub]; cc += [ub, ua]; cd += [w, w]
        if cr:
            A = A + sp.csr_matrix((cd, (cr, cc)), shape=(N, N))

    deg = np.asarray(A.sum(0)).ravel(); deg[deg == 0] = 1.0
    P = A.multiply(sp.csr_matrix(1.0 / deg)).tocsr()
    E = np.array(edges) if edges else np.zeros((0, 3), int)
    return Cm, P, Nc, Nh, N, E


# ── run the five methods online on one variant, full 7-metric suite ──────────

def run_variant(variant, cids, y, files, devs, hubs, quick=False):
    import kg_methods as km
    from online_infer import WARMUP_FRAC, BLOCK
    from online_jit import final_metrics

    Cm, P, Nc, Nh, N, edges = build_cstg_graph(variant, cids, files, devs, hubs)
    W = int(Nc * WARMUP_FRAC)
    REFIT_EMB, DW_DIM, KGE_DIM = 5, 64, 32
    pred = {m: np.full(Nc, np.nan) for m in C.FINAL_METHODS}
    past_mask = np.zeros(Nc, bool)
    dw_clf = kge_clf = dwE = kgeE = None

    i = W; blk = 0
    while i < Nc:
        j = min(Nc, i + BLOCK); idx = np.arange(i, j); past = np.arange(i)
        gr = y[past].mean() if len(past) else 0.0
        pred["RN"][idx] = km.rn_scores(Cm, y, past, idx)
        pred["LP"][idx] = km.lp_scores(Cm, y, past, idx)
        bs = past[y[past] == 1]; gs = past[y[past] == 0]
        rb = km._ppr(P, list(bs), N); rg = km._ppr(P, list(gs), N)
        pred["PPR"][idx] = rb[idx] / (rb[idx] + rg[idx] + 1e-12)
        if blk % REFIT_EMB == 0:
            past_mask[:] = False; past_mask[past] = True
            dwE = km.dw_embed(Cm, past, DW_DIM)
            dw_clf = km._lr().fit(dwE[past], y[past]) if len(set(y[past])) > 1 else None
            kgeE = km.kge_embed(edges, Nc, Nh, past_mask, dim=KGE_DIM)
            kge_clf = km._lr().fit(kgeE[past], y[past]) if len(set(y[past])) > 1 else None
        pred["DW"][idx]  = dw_clf.predict_proba(dwE[idx])[:, 1] if dw_clf else gr
        pred["KGE"][idx] = kge_clf.predict_proba(kgeE[idx])[:, 1] if kge_clf else gr
        i = j; blk += 1

    ev = np.arange(W, Nc); yt = y[ev]
    out = {}
    for m in C.FINAL_METHODS:
        p = np.clip(np.nan_to_num(pred[m][ev], nan=yt.mean()), 0, 1)
        fm = final_metrics(yt, p)
        out[m] = {k: float(fm[k]) for k in ("Precision", "Recall", "Buggy_F1",
                                            "Macro_F1", "G_Mean", "AUC", "MCC")}
    # PPR rolling online-evaluation trajectory (all 7 metrics; the notebook plots
    # Macro-F1 and G-Mean per CSTG layer). Reuses the V4 metric_traj helper so the
    # x-axis / rolling window match the paper's other stream figures.
    from run_final_experiments import metric_traj
    ppr_p = np.clip(np.nan_to_num(pred["PPR"][ev], nan=yt.mean()), 0, 1)
    ppr_traj = metric_traj(yt, ppr_p, W)
    ppr_traj = {k: (v if isinstance(v, list) else list(v)) for k, v in ppr_traj.items()}
    info = dict(Nh=Nh, nnz=int(Cm.nnz), n_edges=int(edges.shape[0]), ppr_traj=ppr_traj)
    # full-length RN/PPR scores so the deployed fusion F=RN+PPR (and F+G) can be
    # rebuilt per layer for the fusion-stream figure.
    return out, info, {m: pred[m].copy() for m in ("RN", "PPR")}


# ── the G feature-classifier channel, online (graph-independent reference) ────

def run_G(cids, y):
    """Online CSTG feature classifier (the deployed G channel) under the same
    prequential protocol + final_metrics, for comparison with the graph methods."""
    import cstg_online_features as cof
    from online_infer import WARMUP_FRAC, BLOCK
    from online_jit import final_metrics
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    S = cof.build_online_streams(list(cids), np.asarray(y))
    Xtext, prior, typed = S["Xtext"], S["prior"], S["typed"]
    Nc = len(cids); W = int(Nc * WARMUP_FRAC)

    def block_feats(idx):
        dense = np.column_stack([prior[idx], typed[idx]])
        return dense

    preds = np.full(Nc, np.nan)
    i = W; blk = 0
    scaler = StandardScaler().fit(block_feats(np.arange(W)))
    Xd0 = sp.csr_matrix(scaler.transform(block_feats(np.arange(W))))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        sp.hstack([Xd0, Xtext[np.arange(W)]]).tocsr(), y[:W])
    while i < Nc:
        j = min(Nc, i + BLOCK); idx = np.arange(i, j)
        Xd = sp.csr_matrix(scaler.transform(block_feats(idx)))
        Xb = sp.hstack([Xd, Xtext[idx]]).tocsr()
        preds[idx] = clf.predict_proba(Xb)[:, 1]
        if blk % 5 == 0:
            scaler = StandardScaler().fit(block_feats(np.arange(j)))
            Xd0 = sp.csr_matrix(scaler.transform(block_feats(np.arange(j))))
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
                sp.hstack([Xd0, Xtext[np.arange(j)]]).tocsr(), y[:j])
        i = j; blk += 1
    ev = np.arange(W, Nc); yt = y[ev]
    p = np.clip(np.nan_to_num(preds[ev], nan=yt.mean()), 0, 1)
    fm = final_metrics(yt, p)
    from run_final_experiments import metric_traj
    g_traj = metric_traj(yt, p, W)
    g_traj = {k: (v if isinstance(v, list) else list(v)) for k, v in g_traj.items()}
    metrics = {k: float(fm[k]) for k in M7}
    # return the full-length G score too, so the deployed fusion F+G can be rebuilt
    # per CSTG layer, and G's own online-evaluation trajectory for the trend line.
    return metrics, preds.copy(), g_traj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-g", action="store_true")
    args = ap.parse_args()

    b = C.load_final_graph_cache()          # commits/cids/y/files/devs (cached)
    cids, y, files, devs = b["cids"], b["y"], b["files"], b["devs"]
    print("fetching CSTG hub data (read-only, cached) ...")
    hubs = fetch_cstg_hubs()
    print(f"  MENTIONS for {len(hubs['mentions'])} commits, "
          f"{len(hubs['cooccurs'])} COOCCURS edges, "
          f"{len(set(hubs['kinds'].values()))} term kinds")

    variants = ["core", "twidf", "full"] if args.quick else VARIANTS
    out = {"_meta": dict(
        scope="CSTG online graph ablation: 5 graph methods (RN/PPR/LP/DW/KGE) + G, "
              "fully online (prequential), 7 metrics, over separated CSTG hub-graph "
              "layers + full CSTG. Also the DEPLOYED fusion F+G=RN+PPR+CSTG per "
              "layer, and G's own online trajectory, for the fusion-stream figure.",
        metrics=M7, methods=C.FINAL_METHODS, variants=variants,
        variant_names=VARIANT_NAME, warmup_frac=None,
        fusion="F+G = RN+PPR+CSTG (deployed), prequential-LR stacked")}

    # G channel once (layer-independent): metrics, full-length score, trajectory
    g_metrics = g_score = g_traj = None
    if not args.no_g:
        print("=== G channel (CSTG feature classifier, online) ===")
        g_metrics, g_score, g_traj = run_G(cids, y)
        out["G_channel"] = g_metrics
        out["G_traj"] = g_traj
        print(f"  G    BF1={g_metrics['Buggy_F1']:.3f} MacroF1={g_metrics['Macro_F1']:.3f} "
              f"GM={g_metrics['G_Mean']:.3f} AUC={g_metrics['AUC']:.3f}")

    from run_final_experiments import metric_traj, WARMUP_FRAC
    from run_final_fusion import eval_subset, INIT
    Nc = len(cids); W = int(Nc * WARMUP_FRAC)

    for v in variants:
        print(f"\n=== CSTG variant: {VARIANT_NAME[v]} ===")
        res, info, mpred = run_variant(v, cids, y, files, devs, hubs, quick=args.quick)
        out[v] = dict(methods=res, **info)
        for m in C.FINAL_METHODS:
            r = res[m]
            print(f"  {m:<4} BF1={r['Buggy_F1']:.3f} MacroF1={r['Macro_F1']:.3f} "
                  f"GM={r['G_Mean']:.3f} AUC={r['AUC']:.3f} MCC={r['MCC']:.3f}")
        # graph fusion F = RN+PPR (on THIS layer's graph), and the deployed
        # F+G = F + CSTG G channel -- both prequential-LR stacked exactly as
        # run_final_fusion.eval_subset does (same protocol as the paper).
        yA = np.asarray(y)
        scoresF = {"RN": mpred["RN"], "PPR": mpred["PPR"]}
        fm_F, ftr_F, _, _ = eval_subset(scoresF, ["RN", "PPR"], yA, W, Nc, init=INIT)
        out[v]["F_metrics"] = {k: float(fm_F[k]) for k in M7}
        out[v]["F_traj"] = {k: (val if isinstance(val, list) else list(val))
                            for k, val in ftr_F.items()}
        print(f"  F    BF1={fm_F['Buggy_F1']:.3f} MacroF1={fm_F['Macro_F1']:.3f} "
              f"GM={fm_F['G_Mean']:.3f} AUC={fm_F['AUC']:.3f}")
        if g_score is not None:
            scores = {"RN": mpred["RN"], "PPR": mpred["PPR"], "G": g_score}
            fm, ftr, _, _ = eval_subset(scores, ["RN", "PPR", "G"], yA, W, Nc, init=INIT)
            out[v]["fusion_metrics"] = {k: float(fm[k]) for k in M7}
            out[v]["fusion_traj"] = {k: (val if isinstance(val, list) else list(val))
                                     for k, val in ftr.items()}
            print(f"  F+G  BF1={fm['Buggy_F1']:.3f} MacroF1={fm['Macro_F1']:.3f} "
                  f"GM={fm['G_Mean']:.3f} AUC={fm['AUC']:.3f}")

    p = C.save_json(out, "cstg_graph_ablation.json")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
