"""
Effort-aware JIT defect-prediction metrics (CPU, no dependencies beyond numpy).

The JIT-SDP literature evaluates predictors not only by threshold-free ranking
(ROC/PR-AUC) but by how many bugs an inspector finds per unit of *inspection
effort*, where effort of a commit is its churn (lines added+deleted). We provide
the two standard measures used by Kamei et al. and follow-ups:

  * popt(y, score, effort)      -> normalised P_opt in [0,1] (higher better)
  * recall_at_effort(...)       -> ACC@k%LOC: recall of buggy commits after
                                   inspecting commits (ranked by risk) until k%
                                   of total effort is spent (a.k.a. Recall@20%).
  * effort_at_recall(...)       -> PCI: % effort needed to reach r% recall.

Ranking: by default commits are ordered by predicted *defect density*
(score / effort) which is the effort-aware ordering; pass density=False to rank
by raw score. Ties and zero-effort commits are handled explicitly.

These are pure functions over (y, score, effort) arrays so they can be dropped
into any evaluation harness (compare_baselines.py, online_jit.py) without Neo4j.
"""
import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _order(score, effort, density):
    """Descending inspection order. density -> score/effort (effort-aware)."""
    score = np.asarray(score, float)
    effort = np.asarray(effort, float)
    eff = np.where(effort <= 0, 1.0, effort)          # avoid /0; min effort = 1 LOC
    key = score / eff if density else score
    # stable order: higher key first; break ties by smaller effort (cheaper first)
    return np.lexsort((eff, -key))


def _cumulative_curve(y, effort, order):
    """Return (x, yv): cumulative fraction of effort vs fraction of bugs found,
    starting at (0,0), along the given inspection order."""
    y = np.asarray(y, float)[order]
    e = np.asarray(effort, float)[order]
    e = np.where(e <= 0, 1.0, e)
    cx = np.concatenate([[0.0], np.cumsum(e)])
    cy = np.concatenate([[0.0], np.cumsum(y)])
    tot_e = cx[-1] if cx[-1] > 0 else 1.0
    tot_y = cy[-1] if cy[-1] > 0 else 1.0
    return cx / tot_e, cy / tot_y


def _area(x, yv):
    """Area under a monotone-x step/linear curve via the trapezoid rule."""
    return float(_trapz(yv, x))


def popt(y, score, effort, density=True):
    """Normalised P_opt (Kamei et al.): 1 - (A_opt - A_model)/(A_opt - A_worst).

    A_opt   : bugs-vs-effort curve when commits are inspected in the *ideal*
              order (buggy first, then by least effort).
    A_worst : the reverse (benign/most-effort first).
    A_model : the predicted order. Returns a value in [0,1]; 0.5 is random.
    """
    y = np.asarray(y, float)
    effort = np.asarray(effort, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    eff = np.where(effort <= 0, 1.0, effort)
    # optimal: buggy first, within class least-effort first
    opt = np.lexsort((eff, -y))
    wst = opt[::-1]
    a_model = _area(*_cumulative_curve(y, effort, _order(score, effort, density)))
    a_opt = _area(*_cumulative_curve(y, effort, opt))
    a_wst = _area(*_cumulative_curve(y, effort, wst))
    denom = (a_opt - a_wst)
    if abs(denom) < 1e-12:
        return float("nan")
    return float(1.0 - (a_opt - a_model) / denom)


def recall_at_effort(y, score, effort, frac=0.20, density=True):
    """ACC@frac: fraction of ALL buggy commits found after inspecting commits
    (in risk order) until `frac` of total effort is spent."""
    y = np.asarray(y, float)
    order = _order(score, effort, density)
    ys = y[order]
    e = np.asarray(effort, float)[order]
    e = np.where(e <= 0, 1.0, e)
    budget = frac * e.sum()
    spent = np.cumsum(e)
    take = spent <= budget
    if not take.any():
        take[0] = True                                # always allow the first commit
    found = ys[take].sum()
    total = y.sum()
    return float(found / total) if total > 0 else float("nan")


def effort_at_recall(y, score, effort, recall=0.20, density=True):
    """PCI@recall: fraction of total effort spent to reach `recall` of all bugs."""
    y = np.asarray(y, float)
    order = _order(score, effort, density)
    ys = y[order]
    e = np.asarray(effort, float)[order]
    e = np.where(e <= 0, 1.0, e)
    total_bugs = y.sum()
    if total_bugs == 0:
        return float("nan")
    target = recall * total_bugs
    found = np.cumsum(ys)
    idx = np.searchsorted(found, target)
    idx = min(idx, len(e) - 1)
    return float(e[:idx + 1].sum() / e.sum())


def all_effort_metrics(y, score, effort, density=True):
    return dict(
        Popt=popt(y, score, effort, density),
        ACC_20=recall_at_effort(y, score, effort, 0.20, density),
        Effort_20=effort_at_recall(y, score, effort, 0.20, density),
    )


if __name__ == "__main__":
    # self-test on synthetic data: a perfect ranker should score Popt~1 and
    # find (almost) all bugs cheaply; a random ranker should sit near 0.5 / prior.
    rng = np.random.default_rng(0)
    n = 4000
    effort = rng.integers(1, 400, n).astype(float)
    y = (rng.random(n) < 0.2).astype(int)
    perfect = y + rng.random(n) * 1e-6                 # near-perfect score
    randsc = rng.random(n)
    worst = 1 - y + rng.random(n) * 1e-6               # anti-correlated
    for name, s in [("perfect", perfect), ("random", randsc), ("worst", worst)]:
        m = all_effort_metrics(y, s, effort)
        print(f"{name:8} Popt={m['Popt']:.3f}  ACC@20%={m['ACC_20']:.3f}  "
              f"Effort@20%rec={m['Effort_20']:.3f}")
    print("\nexpected: perfect Popt~1 / high ACC ; random Popt~0.5 / ACC~0.2 ; worst Popt~0")
