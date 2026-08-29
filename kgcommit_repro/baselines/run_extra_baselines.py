"""
EXTRA JIT baselines (LApredict, Deeper, JITLine), evaluated under the EXACT SAME
online prequential protocol as run_baselines.py and the KG models, so their numbers
sit in the same table and on the same stream axis.

New baselines
-------------
  * B_LAPREDICT  Logistic Regression on the single feature "added lines" (la).
                 (Zeng et al. 2021, "deep learning JIT is overrated" -- LApredict.)
  * B_DEEPER     A DBN-style nonlinear feature transform over the 12 change metrics,
                 then Logistic Regression. (Yang et al. 2015, "Deeper".) The DBN is
                 reimplemented pragmatically as a small unsupervised autoencoder /
                 MLP transform (modern reimplementations of Deeper do this); it runs
                 on the 12-d metric vector, so it is CPU-cheap and needs no GPU.
  * B_JITLINE    Random Forest on [12 expert metrics || diff bag-of-token counts].
                 (Pornprasit et al. 2021, JITLine -- prediction part only; the LIME
                 line-localization is out of scope for commit-level metrics.) Token
                 features reuse the project's CSTG diff tokenizer, fit train-only.

Protocol reuse: this script imports prequential_scores / evaluate / load_project /
WARMUP_FRAC / BLOCK / metric helpers directly from run_baselines, so warm-up window,
block size, refit cadence, scaling-on-past, the online-tuned operating point, the 7
metrics and the effort metrics are byte-identical to the existing baselines.

Reads label CSV (+ diff CSV for JITLine) in place -- NO Neo4j, runs anywhere.

Out: outputs/<project>/baseline_extra_results.pkl
Run: KGC_PROJECT=zookeeper python baselines/run_extra_baselines.py
"""
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import CSV_PATH, DIFF_CSV, OUT, PROJECT  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

# reuse the EXACT protocol + metrics from the existing baseline harness
import run_baselines as rb   # noqa: E402
from online_jit import final_metrics                 # noqa: E402
# FINAL RUN: protocol constants from the single source of truth (see protocol.py).
from protocol import WARMUP_FRAC, BLOCK, REFIT_EVERY  # noqa: E402
import effort_metrics as em                          # noqa: E402
from timing_probe import Probe                        # noqa: E402
from run_subgraph_rq import metric_trajectory, TRAJ_STRIDE, TRAJ_WINDOW  # noqa: E402

JIT_COLS = rb.JIT_COLS
METRIC_KEYS = rb.METRIC_KEYS


# ── LApredict: LR on the single "la" feature ────────────────────────────────
def lapredict_scores(df, y, effort, probe=None):
    Xla = df[["la"]].fillna(0.0).to_numpy(float)
    return rb.prequential_scores(Xla, y, lambda: rb._lr(), probe=probe)


# ── Deeper: autoencoder transform of the 12 metrics -> LR ───────────────────
class _DeeperModel:
    """DBN-style: learn an unsupervised nonlinear encoding of the 12 metrics with a
    small autoencoder (MLPRegressor reconstructing its input), then LR on the hidden
    code concatenated with the raw metrics. Fit only on the given (past) window."""
    def __init__(self, hidden=(16, 8)):
        self.hidden = hidden

    def fit(self, X, y):
        # autoencoder: input -> hidden code -> input  (unsupervised, past only)
        self.ae = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu",
                               max_iter=300, random_state=0)
        self.ae.fit(X, X)
        Z = self._encode(X)
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                      solver="lbfgs").fit(np.hstack([X, Z]), y)
        return self

    def _encode(self, X):
        # forward through all but the final (reconstruction) layer to get the code
        H = X
        for W, b in zip(self.ae.coefs_[:-1], self.ae.intercepts_[:-1]):
            H = np.maximum(0.0, H @ W + b)             # relu
        return H

    def predict_proba(self, X):
        Z = self._encode(X)
        return self.clf.predict_proba(np.hstack([X, Z]))


def deeper_scores(df, y, effort, probe=None):
    X = df[JIT_COLS].fillna(0.0).to_numpy(float)
    return rb.prequential_scores(X, y, lambda: _DeeperModel(), probe=probe)


# ── JITLine: RF on [expert metrics || diff bag-of-tokens] ───────────────────
def _load_diff_tokens(cids_order):
    """Return {commit_id: [token,...]} using the CSTG diff tokenizer, aligned to the
    label CSV's chronological commit order. Missing diffs -> empty token list."""
    from inference.cstg import parse_commit_text, commit_terms
    dd = pd.read_csv(DIFF_CSV)
    tok_by = {}
    for cid, txt in zip(dd["commit_id"], dd["diff_text"].fillna("")):
        try:
            tok_by[cid] = commit_terms(parse_commit_text(txt))
        except Exception:
            tok_by[cid] = []
    return [tok_by.get(c, []) for c in cids_order]


def _jitline_features(df, tokens_per_commit, W, top_k=2000):
    """Expert metrics || bag-of-token counts. Vocabulary is fit on the TRAIN window
    only (first W commits) to stay leakage-free, then applied to all commits."""
    from collections import Counter
    vocab_counter = Counter()
    for toks in tokens_per_commit[:W]:
        vocab_counter.update(sorted(set(toks)))
    # Deterministic top-k: Counter.most_common breaks ties in insertion order, and
    # the insertion order of a set of strings varies between interpreter runs (string
    # hash randomisation). At the top_k cut-off that reshuffles tied tokens in and out
    # of the vocabulary, perturbing the feature matrix and hence JITLine's metrics by
    # up to ~0.03 between otherwise identical runs. Sorting by (-count, token) makes
    # the vocabulary a pure function of the training window.
    ranked = sorted(vocab_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    vocab = {t: i for i, (t, _) in enumerate(ranked)}
    V = len(vocab)
    N = len(tokens_per_commit)
    rows, cols, data = [], [], []
    for i, toks in enumerate(tokens_per_commit):
        c = Counter(toks)
        for t, n in c.items():
            j = vocab.get(t)
            if j is not None:
                rows.append(i); cols.append(j); data.append(float(n))
    Tok = sp.csr_matrix((data, (rows, cols)), shape=(N, max(V, 1)))
    Xexp = df[JIT_COLS].fillna(0.0).to_numpy(float)
    return Xexp, Tok


class _JITLineModel:
    """RF on dense[expert] hstacked with sparse[tokens]. Handles the mixed
    dense/sparse by converting the (already reduced) token block to dense at fit."""
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         n_jobs=-1, random_state=0)

    def fit(self, X, y):
        self.rf.fit(X, y); return self

    def predict_proba(self, X):
        return self.rf.predict_proba(X)


def jitline_scores(df, y, cids_order, probe=None):
    """Prequential RF over [expert || tokens]. Uses the SAME warm-up/block/refit as
    prequential_scores but with its own feature matrix (tokens need the combined
    dense matrix). Vocabulary fit on the warm-up window only.

    `probe`: JITLine's deployment cost is dominated by FEATURISATION (diff
    tokenisation + bag-of-tokens vectorisation), not by the model call. We time the
    whole featurisation path and amortise it over the scored commits, so the
    comparison against KG-Commit's pre-materialised read is honest."""
    N = len(y); W = int(N * WARMUP_FRAC)
    n_scored = N - W
    if probe is not None:
        # tokenisation + vectorisation = the per-commit feature build a deployed
        # JITLine must perform for every incoming commit.
        with probe.featurize(n=n_scored):
            toks = _load_diff_tokens(cids_order)
            Xexp, Tok = _jitline_features(df, toks, W)
            Xfull = np.hstack([Xexp, Tok.toarray()])
    else:
        toks = _load_diff_tokens(cids_order)
        Xexp, Tok = _jitline_features(df, toks, W)
        Xfull = np.hstack([Xexp, Tok.toarray()])
    preds = np.full(N, np.nan)
    scaler = StandardScaler(with_mean=False).fit(Xfull[:W])
    def fit(u):
        return _JITLineModel().fit(scaler.transform(Xfull[:u]), y[:u])
    clf = fit(W); i = W; blk = 0
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        if probe is not None:
            with probe.predict(n=len(idx)):
                preds[idx] = clf.predict_proba(scaler.transform(Xfull[idx]))[:, 1]
        else:
            preds[idx] = clf.predict_proba(scaler.transform(Xfull[idx]))[:, 1]
        if blk % REFIT_EVERY == 0:
            if probe is not None:
                with probe.refit():
                    clf = fit(j)
            else:
                clf = fit(j)
        i = j; blk += 1
    return preds, W


def jitline_online_scores(df, y, cids_order, probe=None):
    """JITLine_fully_online -- JITLine with an ONLINE textual model.

    The incumbent `jitline_scores` fits its bag-of-tokens VOCABULARY once, on the
    warm-up window, and never revisits it: only the RF is refit. A vocabulary frozen
    at warm-up cannot represent identifiers, APIs or error types introduced later,
    so on a long history JITLine is scored on an increasingly stale representation.
    KG-Commit's CSTG, by contrast, refreshes its term layer every block.

    This variant removes that asymmetry: the vocabulary AND the vectoriser are refit
    on the expanding past window every REFIT_EVERY blocks, exactly like CSTG, so the
    comparison isolates the representation rather than the update schedule. It is
    strictly more expensive -- re-vectorising the whole past window each refit --
    which is itself an RQ2 result, not a drawback.

    Reported alongside (never instead of) the incumbent, so both the original
    published behaviour and the fair-update behaviour are visible.
    """
    N = len(y); W = int(N * WARMUP_FRAC)
    n_scored = N - W
    toks = _load_diff_tokens(cids_order)
    Xexp = df[JIT_COLS].fillna(0.0).to_numpy(float)

    preds = np.full(N, np.nan)
    clf = None; scaler = None; Xfull = None

    def refit_representation(u):
        """Rebuild vocabulary + matrix from the first `u` commits (past only)."""
        _, Tok = _jitline_features(df, toks, u)
        Xf = np.hstack([Xexp, Tok.toarray()])
        sc = StandardScaler(with_mean=False).fit(Xf[:u])
        model = _JITLineModel().fit(sc.transform(Xf[:u]), y[:u])
        return Xf, sc, model

    if probe is not None:
        with probe.featurize(n=n_scored):
            Xfull, scaler, clf = refit_representation(W)
    else:
        Xfull, scaler, clf = refit_representation(W)

    i = W; blk = 0
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        if probe is not None:
            with probe.predict(n=len(idx)):
                preds[idx] = clf.predict_proba(scaler.transform(Xfull[idx]))[:, 1]
        else:
            preds[idx] = clf.predict_proba(scaler.transform(Xfull[idx]))[:, 1]
        if blk % REFIT_EVERY == 0:
            # refit BOTH the representation and the model on the expanding past
            if probe is not None:
                with probe.refit():
                    Xfull, scaler, clf = refit_representation(j)
            else:
                Xfull, scaler, clf = refit_representation(j)
        i = j; blk += 1
    return preds, W


# ── main ────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH).sort_values("author_date").reset_index(drop=True)
    y = df["buggy"].astype(int).to_numpy()
    effort = (df["la"].fillna(0) + df["ld"].fillna(0)).to_numpy(float) + 1.0
    cids_order = df["commit_id"].tolist()
    N = len(y); W = int(N * WARMUP_FRAC)
    print(f"[{PROJECT}] extra baselines over {N} commits (warm-up {W}, eval {N-W}), "
          f"bug rate {y.mean():.3f}\n")

    results = {}; preds_by = {}; timings = {}

    # also recompute the 3 learned classic baselines HERE so this file holds ALL
    # seven learned baselines' per-commit trajectories in one place (the original
    # baseline_results.pkl stored a trajectory only for the single best baseline).
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xexp = df[JIT_COLS].fillna(0.0).to_numpy(float)
    classic = {
        "B_LR":  lambda: rb._lr(),
        "B_RF":  lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                n_jobs=-1, random_state=0),
        "B_HGB": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    for name, fac in classic.items():
        print(f"  {name} (re-scored for its trajectory) ...")
        pr = Probe()
        p, w = rb.prequential_scores(Xexp, y, fac, probe=pr)
        results[name] = rb.evaluate(y, p, effort, w); preds_by[name] = p
        timings[name] = pr.finalize(n_scored=int(N - w))

    print("  LApredict (LR on la) ...")
    pr = Probe()
    p, w = lapredict_scores(df, y, effort, probe=pr)
    results["B_LAPREDICT"] = rb.evaluate(y, p, effort, w); preds_by["B_LAPREDICT"] = p
    timings["B_LAPREDICT"] = pr.finalize(n_scored=int(N - w))

    # NOTE (final run): Deeper/DeepJIT is NOT part of the final baseline set --
    # it is not being run. The function is retained for reproducibility of earlier
    # results but is skipped here.
    if os.environ.get("KGC_RUN_DEEPER") == "1":
        print("  Deeper (autoencoder transform + LR) ...")
        pr = Probe()
        p, w = deeper_scores(df, y, effort, probe=pr)
        results["B_DEEPER"] = rb.evaluate(y, p, effort, w); preds_by["B_DEEPER"] = p
        timings["B_DEEPER"] = pr.finalize(n_scored=int(N - w))

    print("  JITLine (RF on expert + diff tokens; vocab frozen at warm-up) ...")
    pr = Probe()
    p, w = jitline_scores(df, y, cids_order, probe=pr)
    results["B_JITLINE"] = rb.evaluate(y, p, effort, w); preds_by["B_JITLINE"] = p
    timings["B_JITLINE"] = pr.finalize(n_scored=int(N - w))

    print("  JITLine_fully_online (vocabulary + model refit every block) ...")
    pr = Probe()
    p, w = jitline_online_scores(df, y, cids_order, probe=pr)
    results["B_JITLINE_ONLINE"] = rb.evaluate(y, p, effort, w)
    preds_by["B_JITLINE_ONLINE"] = p
    timings["B_JITLINE_ONLINE"] = pr.finalize(n_scored=int(N - w))

    # per-baseline trajectories (accurate 150 + smoothed 800), same helper/resolution
    ev = np.arange(W, N); yt = y[ev]
    trajs = {}; trajs_smooth = {}; raws = {}
    for name, p in preds_by.items():
        pe = np.nan_to_num(p[ev], nan=float(yt.mean()))
        trajs[name] = metric_trajectory(yt, pe, W)
        trajs_smooth[name] = metric_trajectory(yt, pe, W, roll=800)
        raws[name] = {"idx": (W + np.arange(len(ev))).tolist(),
                      "y": yt.astype(int).tolist(), "pred": pe.astype(float).tolist()}

    out = dict(project=PROJECT, N=N, warmup=W, n_eval=N - W, bug_rate=float(y.mean()),
               metric_keys=METRIC_KEYS, baselines=results, timings=timings,
               trajs=trajs, trajs_smooth=trajs_smooth, raws=raws,
               traj_stride=TRAJ_STRIDE, traj_window=TRAJ_WINDOW)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / "baseline_extra_results.pkl", "wb"))

    hdr = f"{'baseline':<12}{'PR_AUC':>8}{'Buggy_F1':>9}{'Macro_F1':>9}{'G_Mean':>8}{'AUC':>7}{'Popt':>7}{'ACC20':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for name, m in results.items():
        print(f"{name:<12}{m['PR_AUC']:8.3f}{m['Buggy_F1']:9.3f}{m['Macro_F1']:9.3f}"
              f"{m['G_Mean']:8.3f}{m['AUC']:7.3f}{m['Popt']:7.3f}{m['ACC20']:7.3f}")
    print(f"\nsaved -> {OUT / 'baseline_extra_results.pkl'}")


if __name__ == "__main__":
    main()
