"""
Deployment-cost instrumentation for the JIT baselines.

KG-Commit's efficiency claim is not "our model call is faster" -- a 12-feature
tabular classifier's predict_proba is trivially fast. The claim is about the TOTAL
per-commit cost at deployment, which for a baseline is

    featurize(commit)  +  predict(commit)  +  amortised refit

and for KG-Commit is a bounded read of state that already exists (the graph having
been grown incrementally, at O(delta) per commit). Reporting only `predict` would
flatter the baselines and hide exactly the cost the paper is about; reporting only
a total would hide that the model call itself is cheap for everyone. So we time the
three components SEPARATELY and let the table show all three.

Definitions (all per commit, in milliseconds):
  featurize_ms  building this commit's feature vector from raw inputs. For the
                change-metric baselines this is a table lookup (~0); for JITLine it
                is diff tokenisation + bag-of-tokens vectorisation, which is where
                its real deployment cost lives.
  predict_ms    the fitted model's scoring call alone.
  refit_ms      total refit wall-clock over the stream, divided by the number of
                scored commits (i.e. amortised). KG-Commit has no analogue: it never
                retrains a model over the history.
  total_ms      featurize + predict + refit  (the deployment-cost number)

Usage:
    from timing_probe import Probe
    pr = Probe()
    with pr.featurize(): X = build_features(...)
    with pr.refit():     clf = Model().fit(...)
    with pr.predict(n=len(idx)): p = clf.predict_proba(X[idx])
    pr.finalize(n_scored)   -> dict of the four numbers above
"""
import time
from contextlib import contextmanager


class Probe:
    """Accumulates wall-clock for the three deployment-cost components."""

    def __init__(self):
        self._feat_s = 0.0
        self._pred_s = 0.0
        self._refit_s = 0.0
        self._pred_n = 0          # commits actually scored under timing
        self._n_refits = 0
        self._n_featurized = 0

    @contextmanager
    def featurize(self, n=None):
        t0 = time.perf_counter()
        yield
        self._feat_s += time.perf_counter() - t0
        if n:
            self._n_featurized += n

    @contextmanager
    def predict(self, n=1):
        t0 = time.perf_counter()
        yield
        self._pred_s += time.perf_counter() - t0
        self._pred_n += n

    @contextmanager
    def refit(self):
        t0 = time.perf_counter()
        yield
        self._refit_s += time.perf_counter() - t0
        self._n_refits += 1

    def finalize(self, n_scored=None):
        n = n_scored or self._pred_n or 1
        feat = self._feat_s * 1000.0 / n
        pred = self._pred_s * 1000.0 / max(self._pred_n, 1)
        refit = self._refit_s * 1000.0 / n
        return {
            "featurize_ms_per_commit": feat,
            "predict_ms_per_commit": pred,
            "refit_ms_per_commit_amortised": refit,
            "total_ms_per_commit": feat + pred + refit,
            "_raw": {
                "featurize_s": self._feat_s,
                "predict_s": self._pred_s,
                "refit_s": self._refit_s,
                "n_scored": n,
                "n_predict_calls": self._pred_n,
                "n_refits": self._n_refits,
            },
        }


TIMING_KEYS = ("featurize_ms_per_commit", "predict_ms_per_commit",
               "refit_ms_per_commit_amortised", "total_ms_per_commit")
