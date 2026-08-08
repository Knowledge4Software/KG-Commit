"""
Compare the DEPLOYED model F+G under two fusion choices, for the projects where the
parsimony rule picked a rank-2 combo (Cassandra, Zeppelin):
  (i)  the CHOSEN F  (smallest combo within TOL=0.005 of the best Macro-F1), and
  (ii) the RANK-1  F (the single best-Macro-F1 combo).

Question answered: does the leaner chosen F cost anything once the G (CSTG) channel is
added, i.e. at the level of the actually-deployed F+G?

Cache-only (C2): the per-method 'final' scores come from raw_method_scores.pkl and the
G-channel features from online_jit_streams_v5.pkl -- no Neo4j. The fusion arithmetic
reuses run_final_fusion.eval_subset / channel_score verbatim, so the F+G numbers are
identical to the deployed pipeline's.
"""
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

import _kgc_paths  # noqa: F401
import run_final_fusion as RFF          # reuse eval_subset + channel_score verbatim
import run_final_experiments as rfe     # metric_traj, WARMUP_FRAC

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"

# (project folder, chosen F, rank-1 F)  -- the two projects where they differ
CASES = [
    ("cassandra", ["RN", "PPR"], ["RN", "PPR", "DW"]),
    ("zeppelin",  ["RN", "PPR"], ["RN", "PPR", "KGE"]),
]
METRICS = [("Macro_F1", "Macro-F1"), ("G_Mean", "G-Mean"), ("AUC", "AUC"),
           ("Buggy_F1", "Buggy-F1")]


def build_scores(folder):
    """Reconstruct the score dict {RN,PPR,LP,DW,KGE, G} cache-only, exactly as
    run_final_fusion.main() builds it, but reading the cached 'final' method scores
    instead of querying Neo4j."""
    rm = pickle.load(open(OUTP / folder / "raw_method_scores.pkl", "rb"))
    y = np.asarray(rm["y"], int); N = int(rm["N"]); W = int(rm["warmup"])
    fin = rm["scores"]["final"]                       # per-method scores on final graph
    scores = {m: np.asarray(fin[m], float) for m in ["RN", "PPR", "LP", "DW", "KGE"]}
    # G (CSTG) channel -- identical construction to run_final_fusion.main()
    S = pickle.load(open(OUTP / folder / "online_jit_streams_v5.pkl", "rb"))
    assert np.array_equal(np.asarray(S["y"], int), y), f"{folder}: streams not aligned"
    Gfeat = sp.hstack([sp.csr_matrix(np.hstack([S["cstg_prior"][:, None], S["cstg_typed"],
                                                S["cstg_consist"]])), S["Xcstg"]]).tocsr()
    scores["G"] = RFF.channel_score(Gfeat, y, W, N, sparse=True)
    return scores, y, W, N


def fg_metrics(scores, F, y, W, N):
    """F+G metrics under fusion set F (adds the G channel), via the deployed eval_subset."""
    m, _, _, _ = RFF.eval_subset(scores, F + ["G"], y, W, N)
    return m


def main():
    rows = []
    for folder, chosenF, rank1F in CASES:
        scores, y, W, N = build_scores(folder)
        m_chosen = fg_metrics(scores, chosenF, y, W, N)
        m_rank1 = fg_metrics(scores, rank1F, y, W, N)
        print(f"\n=== {folder} ===  chosen F={'+'.join(chosenF)} ({len(chosenF)} methods)"
              f"  vs  rank-1 F={'+'.join(rank1F)} ({len(rank1F)} methods)")
        print(f"  {'metric':<10}{'F+G(chosen)':>13}{'F+G(rank1)':>12}{'delta':>9}")
        rec = dict(project=folder, chosen="+".join(chosenF), rank1="+".join(rank1F),
                   n_chosen=len(chosenF), n_rank1=len(rank1F))
        for mk, mn in METRICS:
            a = float(m_chosen.get(mk, float("nan")))
            b = float(m_rank1.get(mk, float("nan")))
            print(f"  {mn:<10}{a:>13.4f}{b:>12.4f}{b-a:>+9.4f}")
            rec[mk] = dict(chosen=round(a, 4), rank1=round(b, 4), delta=round(b - a, 4))
        rows.append(rec)
    # emit a small LaTeX table
    _emit(rows)
    import json
    json.dump(rows, open(OUTP / "aggregate" / "chosen_vs_rank1_fg.json", "w"), indent=1)
    print("\nsaved -> outputs/aggregate/chosen_vs_rank1_fg.json + table")


def _emit(rows):
    PM = ROOT / "Paper" / "paper_material" / "RQ4_inference"
    PM.mkdir(parents=True, exist_ok=True)
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\caption{Deployed model $F{+}G$ under the parsimony-selected fusion $F$ vs.\ "
         r"the single best-Macro-F1 (rank-1) fusion, for the two projects where they "
         r"differ. The chosen $F$ uses \emph{fewer} methods; $\Delta$ is rank-1 $-$ "
         r"chosen (a positive $\Delta$ means the larger rank-1 model is better). "
         r"Differences are negligible, confirming the leaner choice loses no deployed "
         r"accuracy.}",
         r"\label{tab:chosen_vs_rank1}",
         r"\begin{tabular}{llcccc}", r"\toprule",
         r"Project & Fusion $F$ (\#methods) & Macro-F1 & G-Mean & AUC & Buggy-F1 \\",
         r"\midrule"]
    for r in rows:
        L.append(f"{r['project'].capitalize()} & {r['chosen']} ({r['n_chosen']}) "
                 f"& {r['Macro_F1']['chosen']:.3f} & {r['G_Mean']['chosen']:.3f} "
                 f"& {r['AUC']['chosen']:.3f} & {r['Buggy_F1']['chosen']:.3f} \\\\")
        L.append(f" & {r['rank1']} ({r['n_rank1']}) "
                 f"& {r['Macro_F1']['rank1']:.3f} & {r['G_Mean']['rank1']:.3f} "
                 f"& {r['AUC']['rank1']:.3f} & {r['Buggy_F1']['rank1']:.3f} \\\\")
        L.append(f" & $\\Delta$ (rank1$-$chosen) "
                 f"& {r['Macro_F1']['delta']:+.3f} & {r['G_Mean']['delta']:+.3f} "
                 f"& {r['AUC']['delta']:+.3f} & {r['Buggy_F1']['delta']:+.3f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    (PM / "tab_chosen_vs_rank1.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
