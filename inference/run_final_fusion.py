"""
Fusion analysis on the final Core+AST+CSTG graph, for the final_experiments
notebook. Two parts:

 Part 1 -- all 2^5-1 = 31 combinations of the five graph-inference methods
   (RN, PPR, LP, DW, KGE) fused by prequential logistic-regression stacking;
   every combo scored on all seven metrics and ranked. The final fusion is chosen
   to balance LOAD (fewer methods) and PERFORMANCE (Macro-F1 and G-Mean): the
   smallest combo within a small tolerance of the best (Macro-F1+G-Mean)/2.

 Part 2 -- append the CSTG channel G and the non-graph JIT metrics M (and G+M) to
   the chosen fusion F, comparing F / F+G / F+M / F+G+M.

Produces the per-metric stream figures (Part 1: 7 panels, 6 trends = chosen fusion
+ 5 singles; Part 2: 7 panels, 4 trends) and saves everything for the notebook.

Out: outputs/final_fusion_results.pkl
     outputs/figures/v4/final/fig_final_fusion_part{1,2}.{png,pdf}

Run:  python inference/run_final_fusion.py
"""
import pickle, itertools
from pathlib import Path
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from online_jit import final_metrics
from online_infer import BLOCK
import run_final_experiments as rfe

OUT = Path(__file__).resolve().parent.parent / "outputs"
FIG = OUT / "figures" / "v4" / "final"; FIG.mkdir(parents=True, exist_ok=True)
METHODS = rfe.METHODS                         # RN, PPR, LP, DW, KGE
M7 = [("Precision", "Prec."), ("Recall", "Rec."), ("Macro_F1", "Macro-F1"),
      ("Buggy_F1", "Buggy-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"), ("ACC", "Acc.")]
INIT = 300; TOL = 0.005


def _lr():
    return LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")


def channel_score(X, y, W, N, sparse=False):
    """Prequential LR score of one feature block (valid over [W,N])."""
    pred = np.full(N, np.nan); i = W; blk = 0; clf = None
    solver = "liblinear" if sparse else "lbfgs"
    def fit(u):
        c = LogisticRegression(max_iter=1500, class_weight="balanced", solver=solver)
        return c.fit(X[:u], y[:u]) if len(set(y[:u])) > 1 else None
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        if blk % 2 == 0:
            clf = fit(i)
        pred[idx] = clf.predict_proba(X[idx])[:, 1] if clf is not None else y[:i].mean()
        i = j; blk += 1
    return pred


def eval_subset(scores, subset, y, W, N, init=INIT, refit=3):
    """Metrics + rolling trajectory + preds for a fused subset over [W+init, N].
    Size-1 subsets use the raw method score; size>1 use prequential LR stacking."""
    ev0 = W + init; ev = np.arange(ev0, N)
    if len(subset) == 1:
        p_all = scores[subset[0]]
    else:
        Z = np.column_stack([np.nan_to_num(scores[m], nan=y[:W].mean()) for m in subset])
        p_all = np.full(N, np.nan); i = ev0; blk = 0
        clf = _lr().fit(Z[W:ev0], y[W:ev0]) if len(set(y[W:ev0])) > 1 else None
        while i < N:
            j = min(N, i + BLOCK); idx = np.arange(i, j)
            p_all[idx] = clf.predict_proba(Z[idx])[:, 1] if clf else y[:i].mean()
            if blk % refit == 0 and len(set(y[W:j])) > 1:
                clf = _lr().fit(Z[W:j], y[W:j])
            i = j; blk += 1
    p = np.clip(np.nan_to_num(p_all[ev], nan=y[ev].mean()), 0, 1)
    return final_metrics(y[ev], p), rfe.metric_traj(y[ev], p, ev0), p


def main():
    commits, cids, y, files, devs, tok, cstg = rfe.load_all()
    N = len(cids); W = int(N * rfe.WARMUP_FRAC)
    print(f"final graph: 5 method scores (W={W}) ...")
    _, preds = rfe.run_graph("final", cids, y, files, devs, tok, cstg)
    scores = {m: preds[m] for m in METHODS}

    # M (JIT metrics) and G (CSTG) prequential channel scores, aligned to cids/W
    S = pickle.load(open(OUT / "online_jit_streams_v5.pkl", "rb"))
    assert np.array_equal(np.asarray(S["y"]), y), "streams_v5 not aligned to cids"
    Xms = StandardScaler().fit(S["Xms"][:W]).transform(S["Xms"])
    scores["M"] = channel_score(Xms, y, W, N)
    Gfeat = sp.hstack([sp.csr_matrix(np.hstack([S["cstg_prior"][:, None], S["cstg_typed"],
                                                S["cstg_consist"]])), S["Xcstg"]]).tocsr()
    scores["G"] = channel_score(Gfeat, y, W, N, sparse=True)

    # ---- Part 1: all 31 combinations of the five methods ----
    print("Part 1: 31 fusions ...")
    part1 = {}
    for r in range(1, 6):
        for combo in itertools.combinations(METHODS, r):
            key = "+".join(combo)
            m, tr, _ = eval_subset(scores, list(combo), y, W, N)
            part1[key] = dict(methods=list(combo), n=r, metrics=m, traj=tr)
    # choose: best (Macro-F1+G-Mean)/2, prefer fewer methods within TOL
    def bal(v): return 0.5 * (v["metrics"]["Macro_F1"] + v["metrics"]["G_Mean"])
    best_bal = max(bal(v) for v in part1.values())
    cands = [(k, v) for k, v in part1.items() if bal(v) >= best_bal - TOL]
    chosen_key, chosen = min(cands, key=lambda kv: (kv[1]["n"], -bal(kv[1])))
    print(f"  chosen fusion F = {chosen_key}  "
          f"(Macro-F1={chosen['metrics']['Macro_F1']:.3f}, G-Mean={chosen['metrics']['G_Mean']:.3f})")

    # ---- Part 2: F, F+G, F+M, F+G+M ----
    print("Part 2: F (+G/+M/+G+M) ...")
    F = chosen["methods"]
    part2 = {}
    for name, extra in [("F", []), ("F+G", ["G"]), ("F+M", ["M"]), ("F+G+M", ["G", "M"])]:
        m, tr, _ = eval_subset(scores, F + extra, y, W, N)
        part2[name] = dict(methods=F + extra, metrics=m, traj=tr)

    pickle.dump(dict(part1=part1, chosen=chosen_key, part2=part2, F=F,
                     meta=dict(W=W, N=N, init=INIT)), open(OUT / "final_fusion_results.pkl", "wb"))
    make_figures(part1, chosen_key, part2, scores, y, W, N)
    print(f"saved -> {OUT/'final_fusion_results.pkl'} + figures")


# ── stream figures ───────────────────────────────────────────────────────────

def _grid(panels_series, suptitle, fname, legend_title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
    for ax, (mk, lbl) in zip(axes, M7):
        for lab, (xs, ys, color, lw, ls) in panels_series[mk].items():
            ax.plot(xs, ys, color=color, lw=lw, ls=ls, label=lab)
        ax.set_title(lbl, fontsize=11, weight="bold"); ax.grid(color="#EEE")
        ax.set_axisbelow(True); ax.set_xlabel("commit index", fontsize=8)
        ax.tick_params(labelsize=8)
    h, l = axes[0].get_legend_handles_labels()
    axes[-1].axis("off"); axes[-1].legend(h, l, loc="center", ncol=1, fontsize=11,
                                          title=legend_title, frameon=False)
    fig.suptitle(suptitle, fontsize=13, weight="bold", y=1.01); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{e}", bbox_inches="tight")
    plt.close(fig); print(f"  wrote {fname}")


def make_figures(part1, chosen_key, part2, scores, y, W, N):
    MCOL = {"RN": "#56B4E9", "PPR": "#E69F00", "LP": "#009E73", "DW": "#0072B2", "KGE": "#CC79A7"}
    # Part 1: chosen fusion (bold) + 5 singles
    p1 = {mk: {} for mk, _ in M7}
    ct = part1[chosen_key]["traj"]
    for mk, _ in M7:
        p1[mk][f"F = {chosen_key}"] = (ct["idx"], ct[mk], "#D55E00", 2.8, "-")
        for m in METHODS:
            t = part1[m]["traj"]
            p1[mk][m] = (t["idx"], t[mk], MCOL[m], 1.4, "--" if m not in chosen_key.split("+") else "-")
    _grid(p1, f"Part 1 — chosen fusion F ({chosen_key}) vs the five single methods",
          "fig_final_fusion_part1", "trend")
    # Part 2: F, F+G, F+M, F+G+M
    C2 = {"F": "#0072B2", "F+G": "#009E73", "F+M": "#E69F00", "F+G+M": "#D55E00"}
    p2 = {mk: {} for mk, _ in M7}
    for mk, _ in M7:
        for name in ["F", "F+G", "F+M", "F+G+M"]:
            t = part2[name]["traj"]
            p2[mk][name] = (t["idx"], t[mk], C2[name], 2.6 if name == "F+G+M" else 1.8, "-")
    _grid(p2, "Part 2 — adding CSTG (G) and JIT metrics (M) to the chosen fusion F",
          "fig_final_fusion_part2", "fusion")


if __name__ == "__main__":
    main()
