"""
The FINAL methodology, online: the full commit KG (core + AST + AST-delta +
relational) fused with the CSTG, under the prequential (predict-then-grow)
protocol. Every learner is online-compatible (expanding-window LR on strictly
past data; no RF/tree that cannot be updated online).

Reports the key subsets and the full 63-subset ablation over
{M=metrics, T=structural-TFIDF, R=relational-priors, P=PPR, X=commit-text,
 G=CSTG}. Saves outputs/online_jit_ablation_v3.pkl for the notebook.

Run:  python inference/online_final.py
"""
import pickle, sys
from pathlib import Path
import online_jit as oj

OUT = Path(__file__).resolve().parent.parent / "outputs"

def main():
    full = "--full" in sys.argv        # --full = run the slow 63-subset ablation too
    S = oj.precompute_streams()
    print(f"streams ready: N={S['N']} warmup={S['W']} "
          f"CSTG(text={S['Xcstg'].shape}, prior={S['cstg_prior'].shape})\n")

    key = [
        ("M (JIT metrics)",                 dict(metrics=1)),
        ("M+T+R+P (V2 KG fusion, no CSTG)", dict(metrics=1, tfidf=1, priors=1, ppr=1)),
        ("G (CSTG alone)",                  dict(cstg=1)),
        ("M+G",                             dict(metrics=1, cstg=1)),
        ("M+T+R+G",                         dict(metrics=1, tfidf=1, priors=1, cstg=1)),
        ("M+T+R+P+G (FINAL: full KG+CSTG)", dict(metrics=1, tfidf=1, priors=1, ppr=1, cstg=1)),
    ]
    print(f"{'subset':<34}{'F1_on':>7}{'PR':>7}{'ROC':>7}{'F1@.5':>7}")
    print("-" * 62)
    for name, mask in key:
        c = oj.run_subset(S, **mask)["cum"]
        print(f"{name:<34}{c['F1_online']:7.3f}{c['PR_AUC']:7.3f}{c['ROC_AUC']:7.3f}{c['F1']:7.3f}")

    if not full:
        print("\n(skipping slow 63-subset ablation; pass --full to run it)")
        return
    print("\nfull ablation (top 10 by F1_online) ...")
    ABL = oj.ablation_all(S)
    pickle.dump(ABL, open(OUT / "online_jit_ablation_v3.pkl", "wb"))
    top = sorted(ABL.items(), key=lambda kv: -kv[1]["cum"]["F1_online"])[:10]
    for lab, v in top:
        c = v["cum"]
        print(f"  {lab:<16}{c['F1_online']:7.3f}  PR={c['PR_AUC']:.3f}  ROC={c['ROC_AUC']:.3f}")
    print("\nsaved -> outputs/online_jit_ablation_v3.pkl")


if __name__ == "__main__":
    main()
