"""
Recency / bug-cache features -- the orthogonal, light, online signal.

The relational priors (R) capture a file/dev's CUMULATIVE bug-rate. They do NOT
capture RECENCY: how long since a touched file (or the author) was last in a
buggy commit, how many recent buggy touches, whether this is a quick re-fix.
Recency is a distinct, strongly-predictive JIT signal (Kim et al. bug-cache,
FSE'07; Rahman & Devanbu, ICSE'11; Hassan entropy).

Computed incrementally in stream order: each commit's features come from a
per-file / per-dev state built ONLY from strictly earlier commits (labels
revealed after scoring) -> O(files) per commit, past-only, leakage-free, online.
No pairwise/global structure -> light and scalable.

Public: build(commits, files, devs, cids, y) -> (X dense [n, K], names)
"""
import math
from collections import defaultdict, deque
import numpy as np

DAY = 86400.0
TAU = 30 * DAY          # recency decay time-constant
WIN = 90 * DAY          # "recent" window
NAMES = ["file_bug_recency_max", "file_bug_recency_mean", "n_recent_buggy_files",
         "frac_recent_buggy_files", "file_churn_recency_max", "dev_bug_recency",
         "dev_recent_bugrate", "global_recent_bugrate", "commit_gap_log", "is_refix"]


def build(commits, files, devs, cids, y):
    n = len(cids); K = len(NAMES)
    X = np.zeros((n, K), float)
    last_bug_ts = {}          # file -> ts of most recent PAST buggy touch
    last_touch_ts = {}        # file -> ts of most recent PAST touch (any)
    dev_last_bug = {}         # dev  -> ts of most recent PAST buggy commit
    dev_recent = defaultdict(lambda: deque(maxlen=20))   # dev -> recent labels
    recent = deque(maxlen=100)                            # global recent labels
    prev_ts = None
    for i, c in enumerate(cids):
        t = float(commits[c].get("ts", 0) or 0)
        F = list(files.get(c, ())); d = devs.get(c)
        yi = int(y[i])
        # ---- features from strictly PAST state ----
        recs, churn, nrec = [], [], 0
        for f in F:
            lb = last_bug_ts.get(f)
            recs.append(math.exp(-(t - lb) / TAU) if lb is not None else 0.0)
            if lb is not None and (t - lb) <= WIN:
                nrec += 1
            lt = last_touch_ts.get(f)
            churn.append(math.exp(-(t - lt) / TAU) if lt is not None else 0.0)
        X[i, 0] = max(recs) if recs else 0.0
        X[i, 1] = float(np.mean(recs)) if recs else 0.0
        X[i, 2] = nrec
        X[i, 3] = nrec / len(F) if F else 0.0
        X[i, 4] = max(churn) if churn else 0.0
        dlb = dev_last_bug.get(d)
        X[i, 5] = math.exp(-(t - dlb) / TAU) if dlb is not None else 0.0
        X[i, 6] = float(np.mean(dev_recent[d])) if dev_recent[d] else 0.0
        X[i, 7] = float(np.mean(recent)) if recent else 0.0
        X[i, 8] = math.log1p((t - prev_ts) / 3600.0) if prev_ts is not None else 0.0
        X[i, 9] = 1.0 if any(last_bug_ts.get(f) is not None and (t - last_bug_ts[f]) <= WIN for f in F) else 0.0
        # ---- grow state AFTER scoring (past-only) ----
        recent.append(yi)
        if d is not None: dev_recent[d].append(yi)
        for f in F: last_touch_ts[f] = t
        if yi:
            for f in F: last_bug_ts[f] = t
            if d is not None: dev_last_bug[d] = t
        prev_ts = t
    return X, NAMES
