"""
Fusion-head ablation: re-score the deployed F+G model with different stacking
classifiers (LR = current, RF, GBoost) to remove the LR-vs-RF confound when comparing
against JITLine (which uses a RandomForest). Everything else is held identical: the
same chosen method combination F, the same CSTG channel G, the same online prequential
protocol (warm-up, block, refit cadence, past-only fit, online-tuned threshold), the
same seven metrics. ONLY the fusion head changes.

Runs entirely from cached artifacts (raw_method_scores.pkl + online_jit_streams_v5.pkl
+ final_fusion_results.pkl) -- NO Neo4j.

Out: outputs/<project>/fusion_head_ablation.pkl
     { head -> {metric: value} } for head in {LR, RF, GBoost}, plus meta.
Run: KGC_PROJECT=kafka python inference/fusion_head_ablation.py
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from online_jit import final_metrics                # noqa: E402
from online_infer import BLOCK                       # noqa: E402
import run_final_fusion as rff                       # noqa: E402
import run_final_experiments as rfe                  # noqa: E402

M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
INIT = rff.INIT


def _head(kind):
    if kind == "LR":
        return LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")
    if kind == "RF":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      n_jobs=-1, random_state=0)
    if kind == "GBoost":
        return HistGradientBoostingClassifier(random_state=0)
    raise ValueError(kind)


def eval_subset_head(scores, subset, y, W, N, head, init=INIT, refit=3):
    """Exactly run_final_fusion.eval_subset, but the stacking classifier is `head`.
    Size-1 subsets use the raw score (no head); size>1 stack with the chosen head."""
    ev0 = W + init
    ev = np.arange(ev0, N)
    if len(subset) == 1:
        p_all = scores[subset[0]]
    else:
        Z = np.column_stack([np.nan_to_num(scores[m], nan=y[:W].mean()) for m in subset])
        p_all = np.full(N, np.nan)
        i = ev0
        blk = 0
        clf = _head(head).fit(Z[W:ev0], y[W:ev0]) if len(set(y[W:ev0])) > 1 else None
        while i < N:
            j = min(N, i + BLOCK)
            idx = np.arange(i, j)
            p_all[idx] = clf.predict_proba(Z[idx])[:, 1] if clf else y[:i].mean()
            if blk % refit == 0 and len(set(y[W:j])) > 1:
                clf = _head(head).fit(Z[W:j], y[W:j])
            i = j
            blk += 1
    p = np.clip(np.nan_to_num(p_all[ev], nan=y[ev].mean()), 0, 1)
    return {k: float(v) for k, v in final_metrics(y[ev], p).items() if k in M7 or True}


def rebuild_scores():
    """Reconstruct the {method+G} score dict WITHOUT Neo4j, from caches."""
    rm = pickle.load(open(OUT / "raw_method_scores.pkl", "rb"))
    y = np.asarray(rm["y"]); N = rm["N"]; W = rm["warmup"]
    scores = {m: np.asarray(rm["scores"]["final"][m], dtype=float) for m in rfe.METHODS}
    S = pickle.load(open(OUT / "online_jit_streams_v5.pkl", "rb"))
    if not np.array_equal(np.asarray(S["y"]), y):
        raise SystemExit("streams_v5 not aligned to raw_method_scores (stale cache); "
                         "skip this project until re-run.")
    Gfeat = sp.hstack([sp.csr_matrix(np.hstack([S["cstg_prior"][:, None], S["cstg_typed"],
                                                S["cstg_consist"]])), S["Xcstg"]]).tocsr()
    scores["G"] = rff.channel_score(Gfeat, y, W, N, sparse=True)
    return scores, y, W, N


def main():
    ff = pickle.load(open(OUT / "final_fusion_results.pkl", "rb"))
    F = ff["F"]                                  # chosen method combination
    subset = F + ["G"]                           # deployed F+G
    scores, y, W, N = rebuild_scores()

    out = {"meta": dict(project=PROJECT, F=F, subset=subset, W=W, N=N)}
    print(f"[{PROJECT}] F={'+'.join(F)}  ->  F+G head ablation")
    for head in ["LR", "RF", "GBoost"]:
        m = eval_subset_head(scores, subset, y, W, N, head)
        out[head] = {k: m[k] for k in M7}
        print(f"  {head:7} MacroF1={m['Macro_F1']:.3f} G-Mean={m['G_Mean']:.3f} "
              f"AUC={m['AUC']:.3f} BuggyF1={m['Buggy_F1']:.3f}")

    pickle.dump(out, open(OUT / "fusion_head_ablation.pkl", "wb"))
    print(f"saved -> {OUT/'fusion_head_ablation.pkl'}")


if __name__ == "__main__":
    main()
