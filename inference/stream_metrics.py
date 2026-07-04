"""
Imbalanced-classification metrics for JIT defect prediction, with online
operating-point thresholding and rolling (stream) trends.

Beyond ROC/PR-AUC (threshold-free), the deployment-relevant operating-point
metrics are:
  * buggy_f1  : F1 of the positive (buggy) class
  * macro_f1  : mean of buggy-F1 and benign-F1 (balances both classes)
  * gmean     : sqrt(TPR * TNR)  -- the classic imbalanced metric
  * bal_acc   : (TPR + TNR)/2

All are computed at the ONLINE-tuned threshold (re-tuned on strictly-past
predictions every `step` commits -> leakage-free), exactly like F1_online.
Everything works from a method's stored (y, p) arrays in commit-arrival order,
so no model re-run is needed.
"""
import numpy as np


def _best_thr(y, p):
    """Threshold maximising buggy-F1 over all cut points (past-only when called
    on a prefix)."""
    y = np.asarray(y); p = np.asarray(p)
    P = int(y.sum())
    if P == 0 or P == len(y):
        return 0.5
    order = np.argsort(-p); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp); rec = tp / P
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(p[order][int(np.argmax(f1))])


def online_yhat(y, p, init=300, step=150):
    """Hard predictions using an ONLINE-tuned threshold (retuned on the past
    every `step`), i.e. the realistic operating point."""
    y = np.asarray(y); p = np.asarray(p)
    yhat = np.zeros(len(y), int); thr = 0.5
    for i in range(len(y)):
        yhat[i] = int(p[i] >= thr)
        if i + 1 >= init and (i + 1) % step == 0:
            thr = _best_thr(y[:i + 1], p[:i + 1])
    return yhat


def metrics_at(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    tpr = tp / (tp + fn + 1e-12); tnr = tn / (tn + fp + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    buggy_f1 = 2 * prec * tpr / (prec + tpr + 1e-12)
    prec0 = tn / (tn + fn + 1e-12)
    benign_f1 = 2 * prec0 * tnr / (prec0 + tnr + 1e-12)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0
    return dict(precision=prec, recall=tpr, specificity=tnr,
                buggy_f1=buggy_f1, benign_f1=benign_f1,
                macro_f1=0.5 * (buggy_f1 + benign_f1),
                gmean=float(np.sqrt(max(tpr, 0) * max(tnr, 0))),
                bal_acc=0.5 * (tpr + tnr), mcc=float(mcc), tpr=tpr, tnr=tnr)


def final_metrics(y, p):
    """Whole-stream operating-point metrics at the online-tuned threshold."""
    return metrics_at(y, online_yhat(y, p))


def rolling(y, p, window=800, step=50):
    """Rolling (stream) trend of the operating-point metrics as commits arrive,
    using the online threshold schedule. Returns dict of aligned lists."""
    y = np.asarray(y); p = np.asarray(p)
    yhat = online_yhat(y, p)
    keys = ("precision", "recall", "buggy_f1", "macro_f1", "gmean", "bal_acc", "mcc")
    n = len(y); out = {"idx": []}; out.update({k: [] for k in keys})
    for j in range(window, n + 1, step):
        s = slice(j - window, j)
        m = metrics_at(y[s], yhat[s])
        out["idx"].append(j)
        for k in keys:
            out[k].append(m[k])
    return out
