"""
Fully online / prequential Just-In-Time evaluation.

This realises the project's true protocol: commits arrive one by one in
chronological order, and for EVERY commit we, in order,
    1. PREDICT     its label using only the KG/history known so far;
    2. EVALUATE    -- reveal the true label and update running metrics
                      (cumulative + rolling-window ROC-AUC / PR-AUC / F1 / MCC);
    3. GROW        the KG state -- add this commit's relational/graph evidence
                      (developer/file/global bug-rate counters advance; the
                      heterogeneous graph used by PPR/embeddings includes it);
    4. LEARN       -- update every model with the now-known (features, label)
                      (incremental partial_fit for linear models; periodic
                      expanding-window retraining for trees / embeddings).

We run, in the SAME stream, a set of within-project BASELINES and a set of
KG-native INFERENCE METHODS, so they are compared under one identical online
protocol. The earlier offline/blocked evaluations (run_all.py, online_infer.py,
compare_baselines.py) are left untouched; this is additive.

Notes on faithfulness / leakage:
  * a commit's OWN change (its diff -> JIT metrics, delta-graph tokens) is known
    at arrival, so those features are available at prediction time;
  * historical/relational features (priors, PPR seeds, embedding classifier) use
    ONLY commits strictly before the one being predicted -- enforced by growing
    the state after each prediction and by seeding PPR from the past;
  * expensive graph features (PPR scores, SVD embedding) are refreshed every few
    blocks over the past-only graph -- the standard prequential approximation.

Run:  python inference/online_jit.py
Output: outputs/online_jit_results.pkl  (+ printed final comparison table)
"""
import numpy as np, scipy.sparse as sp, pickle, math
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, brier_score_loss)
import warnings; from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import advanced_infer as ai, online_infer as oi

OUT = Path(__file__).resolve().parent.parent / "outputs"
WARMUP_FRAC = 0.30      # initial fit window (commits 0..W are not scored)
BLOCK       = 50        # learn/refresh granularity (predictions are per-commit)
REFIT_EVERY = 4         # retrain tree / fusion models every N blocks
SVD_REFRESH = 8         # refit the SVD embedding every N blocks
ROLL        = 800       # rolling-metric window (commits)
HASH_DIM    = 2**18

GROUP = {  # method -> ('baseline'|'kg', pretty name)
    "B_LR_metrics":   ("baseline", "LR / JIT metrics (incremental)"),
    "B_RF_metrics":   ("baseline", "RandomForest / JIT metrics"),
    "B_HGB_metrics":  ("baseline", "GradBoost / JIT metrics"),
    "B_naive_rate":   ("baseline", "Naive prior bug-rate"),
    "K_priors_wvRN":  ("kg",       "Relational priors (wvRN)"),
    "K_tfidf":        ("kg",       "Structural TF-IDF (incremental)"),
    "K_ppr":          ("kg",       "Personalized PageRank"),
    "K_svd_emb":      ("kg",       "KG embedding (SVD/LSA)"),
    "K_fusion":       ("kg",       "Fusion (metrics+priors+TFIDF+PPR)"),
}

def _best_threshold(y, p):
    """Threshold on p that maximises buggy-class F1 (exact, over all cut points)."""
    P = int(y.sum())
    if P == 0 or P == len(y): return 0.5
    order = np.argsort(-p); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp); rec = tp / P
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(p[order][int(np.argmax(f1))])

def online_f1(p, y, init=300, step=150):
    """Prequential buggy-F1 with the decision threshold tuned ONLINE on the past
    (re-tuned every `step` commits using only already-seen predictions)."""
    y = np.asarray(y); yhat = np.zeros(len(y), int); thr = 0.5
    for i in range(len(y)):
        yhat[i] = int(p[i] >= thr)
        if i + 1 >= init and (i + 1) % step == 0:
            thr = _best_threshold(y[:i+1], p[:i+1])
    return float(f1_score(y, yhat, zero_division=0))

def cum_metrics(y, p):
    yh = (p >= 0.5).astype(int)
    out = dict(ROC_AUC=float("nan"), PR_AUC=float("nan"))
    if len(np.unique(y)) > 1:
        out["ROC_AUC"] = roc_auc_score(y, p); out["PR_AUC"] = average_precision_score(y, p)
    out.update(F1=f1_score(y, yh, zero_division=0), MCC=matthews_corrcoef(y, yh),
               Brier=brier_score_loss(y, p), Acc=float((yh == y).mean()))
    return out

def final_metrics(y, p):
    """Cumulative metrics + the online-tuned buggy-F1 (for end-of-stream scoring)."""
    cm = cum_metrics(y, p); cm["F1_online"] = online_f1(np.asarray(p), np.asarray(y))
    return cm

def load_stream_data():
    """Load (commits, tokens, files, devs) from a disk cache if present, else
    from Neo4j (and cache it) -- so the streaming experiment can be re-run and
    visualised without a live database."""
    cache = OUT / "kg_stream_cache.pkl"
    if cache.exists():
        print("Loading KG from cache (no Neo4j needed)...")
        return pickle.load(open(cache, "rb"))
    print("Loading KG from Neo4j...")
    data = ai.load_kg()
    pickle.dump(data, open(cache, "wb"))
    return data

# ===========================================================================
#  Feature-ablation support for the Fusion architecture
#  Fusion = {JIT metrics (M), structural TF-IDF (T), relational priors (R),
#            PPR score (P)}. We precompute each stream ONCE (the priors-growth
#  and per-block PPR are identical regardless of which subset is used), then
#  evaluate any subset with a light prequential expanding-window LR.
# ===========================================================================
FEATURES = ["metrics", "tfidf", "priors", "ppr", "text", "cstg"]
FEAT_ABBR = {"metrics": "M", "tfidf": "T", "priors": "R", "ppr": "P", "text": "X",
             "cstg": "G"}   # G = Commit Semantic-Text Graph
DIFFS = OUT.parent / "data/apachejit/apachejit_with_diffs_rebuilt.csv"

def _lr(sparse=False):
    return LogisticRegression(max_iter=1500, class_weight="balanced",
                              solver="liblinear" if sparse else "lbfgs")

def cstg_streams(cids, y):
    """Enhanced ONLINE CSTG streams (graph-of-words-weighted text with add/remove
    polarity, NPMI-propagated past-only prior, and typed mass). See
    cstg_online_features.build_online_streams. Returns (Xtext, prior, typed)."""
    import cstg_online_features as cof
    S = cof.build_online_streams(list(cids), np.asarray(y))
    return S["Xtext"], S["prior"], S["typed"]

def precompute_streams(force=False):
    """Run the leakage-free online growth ONCE and return the feature streams
    (cached). v3 adds the CSTG stream (G), reusing the v2 streams if present."""
    cache_s = OUT / "online_jit_streams_v5.pkl"   # v5: + intent-consistency block
    if cache_s.exists() and not force:
        return pickle.load(open(cache_s, "rb"))
    import cstg_consistency as cc
    v2 = OUT / "online_jit_streams_v2.pkl"
    if v2.exists() and not force:
        S = pickle.load(open(v2, "rb"))
        commits, tokens, files, devs = load_stream_data()
        cids = sorted(commits, key=lambda c: commits[c]["ts"])
        S["Xcstg"], S["cstg_prior"], S["cstg_typed"] = cstg_streams(cids, S["y"])
        S["cstg_consist"], _ = cc.build(commits, tokens, files, cids)
        pickle.dump(S, open(cache_s, "wb")); return S
    commits, tokens, files, devs = load_stream_data()
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y = np.array([commits[c]["buggy"] for c in cids]); N = len(cids)
    Xm = np.array([[commits[c][m] for m in ai.METRICS] for c in cids], float)
    docs = [" ".join((t+" ")*int(min(k,20)) for t,k in tokens.get(c,())) for c in cids]
    Xh = HashingVectorizer(n_features=HASH_DIM, alternate_sign=False,
                           token_pattern=r"[^\s]+").transform(docs)
    # commit-message text stream (natural-language words, lowercased)
    msgs = [str(commits[c].get("message", "")) for c in cids]
    Xtext = HashingVectorizer(n_features=HASH_DIM, alternate_sign=False,
                              ngram_range=(1, 2), lowercase=True).transform(msgs)
    C, P, _, Nn = oi.build_incidence(cids, tokens, files, devs)
    W = int(N*WARMUP_FRAC)
    Xms = StandardScaler().fit(Xm[:W]).transform(Xm)
    gb = gt = 0; devc = defaultdict(lambda: [0,0]); filec = defaultdict(lambda: [0,0]); recent = []
    Xp = np.zeros((N, 5)); ppr_full = np.zeros(N)
    def prior_feats(c):
        g = (gb/gt) if gt else 0.0
        db, dt = devc[devs.get(c)]; dev = (db + g*5)/(dt + 5)
        frs = [(filec[f][0] + g*5)/(filec[f][1] + 5) for f in files.get(c, ())]
        return [g, dev, (np.mean(frs) if frs else g), (max(frs) if frs else g),
                (np.mean(recent[-50:]) if recent else g)]
    def grow(i, c):
        nonlocal gb, gt
        gt += 1; gb += int(y[i]); recent.append(int(y[i]))
        if devs.get(c): devc[devs[c]][1] += 1; devc[devs[c]][0] += int(y[i])
        for f in files.get(c, ()): filec[f][1] += 1; filec[f][0] += int(y[i])
    for i in range(W): Xp[i] = prior_feats(cids[i]); grow(i, cids[i])
    i = W
    while i < N:
        j = min(N, i+BLOCK); idx = np.arange(i, j); past = np.arange(i)
        bs = past[y[past]==1]; gs = past[y[past]==0]
        rb = oi.ppr(P, list(bs), Nn); rg = oi.ppr(P, list(gs), Nn)
        sc = rb[:N]/(rb[:N]+rg[:N]+1e-12); ppr_full[idx] = sc[idx]
        for k in idx: Xp[k] = prior_feats(cids[k]); grow(k, cids[k])
        i = j
    S = dict(y=y, N=N, W=W, Xms=Xms, Xp=Xp, Xh=Xh, ppr_full=ppr_full, Xtext=Xtext)
    S["Xcstg"], S["cstg_prior"], S["cstg_typed"] = cstg_streams(cids, y)
    S["cstg_consist"], _ = cc.build(commits, tokens, files, cids)
    pickle.dump(S, open(cache_s, "wb")); return S

def run_subset(S, metrics=False, tfidf=False, priors=False, ppr=False, text=False,
               cstg=False, block=BLOCK, refit=REFIT_EVERY, roll=ROLL):
    """Prequential expanding-window LR over the selected feature subset."""
    Xms, Xp, Xh, ppr_full, y, W, N = (S["Xms"], S["Xp"], S["Xh"], S["ppr_full"],
                                      S["y"], S["W"], S["N"])
    Xtext = S.get("Xtext"); Xcstg = S.get("Xcstg")
    cstg_prior = S.get("cstg_prior"); cstg_typed = S.get("cstg_typed")
    cstg_consist = S.get("cstg_consist")
    def dense_cols(idx):
        parts = []
        if metrics: parts.append(Xms[idx])
        if priors:  parts.append(Xp[idx])
        if ppr:     parts.append(ppr_full[idx][:, None])
        if cstg and cstg_prior is not None:
            parts.append(cstg_prior[idx][:, None])
            if cstg_typed is not None: parts.append(cstg_typed[idx])
            if cstg_consist is not None: parts.append(cstg_consist[idx])
        return np.hstack(parts) if parts else None
    def build(idx, scaler):
        mats = []
        d = dense_cols(idx)
        if d is not None: mats.append(sp.csr_matrix(scaler.transform(d)))
        if tfidf: mats.append(Xh[idx])
        if text and Xtext is not None: mats.append(Xtext[idx])
        if cstg and Xcstg is not None: mats.append(Xcstg[idx])
        return sp.hstack(mats).tocsr() if mats else None
    def fit_scaler(upto):
        d = dense_cols(np.arange(upto))
        return StandardScaler().fit(d) if d is not None else None
    scaler = fit_scaler(W); clf = _lr(sparse=True).fit(build(np.arange(W), scaler), y[:W])
    preds = np.full(N, np.nan); traj = {"idx": [], "ROC_AUC": [], "PR_AUC": []}
    i = W; blk = 0
    while i < N:
        j = min(N, i+block); idx = np.arange(i, j)
        preds[idx] = clf.predict_proba(build(idx, scaler))[:, 1]
        lo = max(W, j-roll); sl = np.arange(lo, j)
        cm = cum_metrics(y[sl], np.nan_to_num(preds[sl], nan=y[:i].mean()))
        traj["idx"].append(j); traj["ROC_AUC"].append(cm["ROC_AUC"]); traj["PR_AUC"].append(cm["PR_AUC"])
        if blk % refit == 0:
            scaler = fit_scaler(j); clf = _lr(sparse=True).fit(build(np.arange(j), scaler), y[:j])
        i = j; blk += 1
    ev = np.arange(W, N); p = np.nan_to_num(preds[ev], nan=y[ev].mean())
    return dict(cum=final_metrics(y[ev], p), traj=traj, p=p, y=y[ev])

def ablation_all(S):
    """Evaluate all non-empty subsets of the Fusion features (M,T,R,P,X)."""
    import itertools
    res = {}
    for r in range(1, len(FEATURES) + 1):
        for combo in itertools.combinations(FEATURES, r):
            mask = {f: (f in combo) for f in FEATURES}
            label = "+".join(FEAT_ABBR[f] for f in FEATURES if f in combo)
            res[label] = dict(features=list(combo), n=r, **run_subset(S, **mask))
    return res


def main():
    commits, tokens, files, devs = load_stream_data()
    cids = sorted(commits, key=lambda c: commits[c]["ts"])
    y  = np.array([commits[c]["buggy"] for c in cids]); N = len(cids)
    Xm = np.array([[commits[c][m] for m in ai.METRICS] for c in cids], float)
    docs = [" ".join((t+" ")*int(min(k,20)) for t,k in tokens.get(c,())) for c in cids]
    Xh = HashingVectorizer(n_features=HASH_DIM, alternate_sign=False,
                           token_pattern=r"[^\s]+").transform(docs)
    C, P, _, Nn = oi.build_incidence(cids, tokens, files, devs)
    W = int(N*WARMUP_FRAC)
    print(f"{N} commits; warmup={W}; streamed (scored)={N-W}; bug-rate(stream)={y[W:].mean():.3f}\n")

    # ---- incremental relational-prior state (this IS the online KG growth) ----
    gb = gt = 0
    devc = defaultdict(lambda: [0,0]); filec = defaultdict(lambda: [0,0]); recent = []
    Xp = np.zeros((N, 5))
    def prior_feats(c):
        g = (gb/gt) if gt else 0.0
        db, dt = devc[devs.get(c)]; dev = (db + g*5)/(dt + 5)
        frs = [(filec[f][0] + g*5)/(filec[f][1] + 5) for f in files.get(c, ())]
        return [g, dev, (np.mean(frs) if frs else g),
                (max(frs) if frs else g), (np.mean(recent[-50:]) if recent else g)]
    def grow(i, c):                      # add commit c to the streaming state
        nonlocal gb, gt
        gt += 1; gb += int(y[i]); recent.append(int(y[i]))
        if devs.get(c): devc[devs[c]][1] += 1; devc[devs[c]][0] += int(y[i])
        for f in files.get(c, ()): filec[f][1] += 1; filec[f][0] += int(y[i])

    # warmup: realise priors and grow the state over the first W commits
    for i in range(W): Xp[i] = prior_feats(cids[i]); grow(i, cids[i])

    # ---- models ----
    scaler = StandardScaler().fit(Xm[:W]); Xms = scaler.transform(Xm)
    cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y[:W])
    cwd = {0: cw[0], 1: cw[1]}
    sgd_lr = SGDClassifier(loss="log_loss", class_weight=cwd, random_state=0)
    sgd_tf = SGDClassifier(loss="log_loss", class_weight=cwd, random_state=0)
    sgd_lr.partial_fit(Xms[:W], y[:W], classes=[0,1])
    sgd_tf.partial_fit(Xh[:W],  y[:W], classes=[0,1])
    def rfnew():  return RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                                n_jobs=-1, random_state=0)
    def hgbnew(): return HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                        random_state=0)
    def lrnew(sparse=False): return LogisticRegression(max_iter=1500, class_weight="balanced",
                                                       solver="liblinear" if sparse else "lbfgs")
    rf = rfnew().fit(Xms[:W], y[:W]); hgb = hgbnew().fit(Xms[:W], y[:W])
    svd = TruncatedSVD(n_components=64, random_state=0).fit(C[:W])
    E = svd.transform(C); svd_clf = lrnew().fit(E[:W], y[:W])
    # fusion
    def build_fuse(scaler_f, idx, ppr_col):
        dns = scaler_f.transform(np.hstack([Xms[idx], Xp[idx], ppr_col[idx][:,None]]))
        return sp.hstack([sp.csr_matrix(dns), Xh[idx]]).tocsr()
    ppr_full = np.zeros(N)
    scaler_f = StandardScaler().fit(np.hstack([Xms[:W], Xp[:W], ppr_full[:W][:,None]]))
    fus = lrnew(sparse=True).fit(build_fuse(scaler_f, np.arange(W), ppr_full), y[:W])

    preds = {k: np.full(N, np.nan) for k in GROUP}
    traj  = {k: {"idx": [], "ROC_AUC": [], "PR_AUC": []} for k in GROUP}

    i = W; blk = 0
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j); past = np.arange(i)
        # --- refresh graph features over the PAST graph (online growth) ---
        bs = past[y[past]==1]; gs = past[y[past]==0]
        rb = oi.ppr(P, list(bs), Nn); rg = oi.ppr(P, list(gs), Nn)
        ppr_blk = rb[:N]/(rb[:N]+rg[:N]+1e-12)
        ppr_full[idx] = ppr_blk[idx]
        # === per-commit: PREDICT -> EVALUATE(state captured) -> GROW ===
        for k in idx:
            c = cids[k]; Xp[k] = prior_feats(c)            # priors from strict past
        Xfb = build_fuse(scaler_f, idx, ppr_full)
        preds["B_LR_metrics"][idx]  = sgd_lr.predict_proba(Xms[idx])[:,1]
        preds["B_RF_metrics"][idx]  = rf.predict_proba(Xms[idx])[:,1]
        preds["B_HGB_metrics"][idx] = hgb.predict_proba(Xms[idx])[:,1]
        preds["B_naive_rate"][idx]  = Xp[idx,0]
        preds["K_priors_wvRN"][idx] = 0.5*Xp[idx,2]+0.3*Xp[idx,1]+0.2*Xp[idx,0]
        preds["K_tfidf"][idx]       = sgd_tf.predict_proba(Xh[idx])[:,1]
        preds["K_ppr"][idx]         = ppr_blk[idx]
        preds["K_svd_emb"][idx]     = svd_clf.predict_proba(E[idx])[:,1]
        preds["K_fusion"][idx]      = fus.predict_proba(Xfb)[:,1]
        for k in idx: grow(k, cids[k])                     # ADD commits to the KG state
        # --- EVALUATE: rolling-window metrics trajectory ---
        lo = max(W, j-ROLL); sl = np.arange(lo, j)
        for m in GROUP:
            cm = cum_metrics(y[sl], np.nan_to_num(preds[m][sl], nan=y[:i].mean()))
            traj[m]["idx"].append(j); traj[m]["ROC_AUC"].append(cm["ROC_AUC"]); traj[m]["PR_AUC"].append(cm["PR_AUC"])
        # === LEARN: update models with the now-known block ===
        sgd_lr.partial_fit(Xms[idx], y[idx]); sgd_tf.partial_fit(Xh[idx], y[idx])
        if blk % REFIT_EVERY == 0:
            rf = rfnew().fit(Xms[:j], y[:j]); hgb = hgbnew().fit(Xms[:j], y[:j])
            scaler_f = StandardScaler().fit(np.hstack([Xms[:j], Xp[:j], ppr_full[:j][:,None]]))
            fus = lrnew(sparse=True).fit(build_fuse(scaler_f, np.arange(j), ppr_full), y[:j])
        if blk % SVD_REFRESH == 0:
            svd = TruncatedSVD(n_components=64, random_state=0).fit(C[:j]); E = svd.transform(C)
            svd_clf = lrnew().fit(E[:j], y[:j])
        i = j; blk += 1
        if blk % 10 == 0: print(f"  streamed {j}/{N}")

    # ---- final cumulative prequential metrics over the whole stream ----
    ev = np.arange(W, N)
    results = {"meta": dict(N=N, warmup=W, stream_bug=float(y[ev].mean())), "traj": traj, "methods": {}}
    print(f"\n{'method':<34}{'F1_on':>7}{'F1@.5':>7}{'PR':>7}{'ROC':>7}{'MCC':>7}")
    print("-"*68)
    for m in GROUP:
        p = np.nan_to_num(preds[m][ev], nan=y[ev].mean())
        cm = final_metrics(y[ev], p)
        results["methods"][m] = dict(group=GROUP[m][0], name=GROUP[m][1],
                                     y=y[ev], p=p, cum=cm)
        print(f"{GROUP[m][1]:<34}{cm['F1_online']:7.3f}{cm['F1']:7.3f}"
              f"{cm['PR_AUC']:7.3f}{cm['ROC_AUC']:7.3f}{cm['MCC']:7.3f}")
    pickle.dump(results, open(OUT/"online_jit_results.pkl", "wb"))
    print(f"\nsaved -> {OUT/'online_jit_results.pkl'}")

if __name__ == "__main__":
    main()
