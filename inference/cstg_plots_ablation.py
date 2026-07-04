"""
Lean ablation for the V3 comparison/trajectory plots: only the ~12 subsets the two
figures need (singles M,T,R,P,X,G + the fusion subsets), each with trajectories.
Much faster than the full 63-subset ablation. Saves outputs/online_jit_ablation_v3.pkl
in the same {label: {cum, traj}} format the plot cells expect.

Run:  python inference/cstg_plots_ablation.py
"""
import pickle
from pathlib import Path
import online_jit as oj

OUT = Path(__file__).resolve().parent.parent / "outputs"

NEEDED = {
    "M": dict(metrics=1), "T": dict(tfidf=1), "R": dict(priors=1),
    "P": dict(ppr=1), "X": dict(text=1), "G": dict(cstg=1),
    "M+T+R": dict(metrics=1, tfidf=1, priors=1),
    "M+T+R+P": dict(metrics=1, tfidf=1, priors=1, ppr=1),
    "M+T+R+P+X": dict(metrics=1, tfidf=1, priors=1, ppr=1, text=1),
    "M+T+R+G": dict(metrics=1, tfidf=1, priors=1, cstg=1),
    "M+T+R+P+G": dict(metrics=1, tfidf=1, priors=1, ppr=1, cstg=1),
    "M+T+R+P+X+G": dict(metrics=1, tfidf=1, priors=1, ppr=1, text=1, cstg=1),
}


def main():
    S = oj.precompute_streams()
    print(f"streams: N={S['N']} warmup={S['W']}\n{'subset':<16}{'F1_on':>7}{'PR':>7}{'ROC':>7}")
    print("-" * 37)
    ABL = {}
    for lab, mask in NEEDED.items():
        r = oj.run_subset(S, **mask)
        ABL[lab] = dict(cum=r["cum"], traj=r["traj"], p=r["p"], y=r["y"])
        c = r["cum"]
        print(f"{lab:<16}{c['F1_online']:7.3f}{c['PR_AUC']:7.3f}{c['ROC_AUC']:7.3f}")
    pickle.dump(ABL, open(OUT / "online_jit_ablation_v3.pkl", "wb"))
    print(f"\nsaved {len(ABL)} subsets -> outputs/online_jit_ablation_v3.pkl")


if __name__ == "__main__":
    main()
