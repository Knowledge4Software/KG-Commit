"""
Appendix F: CSTG component ablation (cache-only, no Neo4j).
The CSTG channel G is built from four feature blocks:
   prior   = NPMI-propagated bug-risk prior           (S['cstg_prior'])
   typed   = defect-semantic typed term mass          (S['cstg_typed'])
   consist = message<->code consistency features      (S['cstg_consist'])
   gow     = graph-of-words TW-IDF term vectors        (S['Xcstg'])
We score G with each block removed (leave-one-out) and with each block alone, under
the same online protocol, to show each block's marginal contribution. Then we report
the deployed F+G at each ablation.

Writes per project: appendices/F_cstg_ablation/appF_cstg__<project>.tex (+ a trend fig).
Requires: online_jit_streams_v5.pkl, raw_method_scores.pkl, final_fusion_results.pkl.
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: F401
from config.project_config import OUT, PROJECT  # noqa: E402
import run_final_fusion as rff  # noqa: E402
import run_final_experiments as rfe  # noqa: E402
from online_jit import final_metrics  # noqa: E402

M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
PP = {"Precision": "Prec.", "Recall": "Rec.", "Macro_F1": "Macro-F1", "Buggy_F1": "Buggy-F1",
      "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Acc."}
BLOCKS = ["prior", "typed", "consist", "gow"]
PM = Path(__file__).resolve().parent.parent.parent / "Paper" / "paper_material" / "appendices" / "F_cstg_ablation"


def build_blocks(S):
    prior = sp.csr_matrix(np.asarray(S["cstg_prior"]).reshape(-1, 1))
    typed = sp.csr_matrix(np.asarray(S["cstg_typed"]))
    consist = sp.csr_matrix(np.asarray(S["cstg_consist"]))
    gow = S["Xcstg"].tocsr() if sp.issparse(S["Xcstg"]) else sp.csr_matrix(S["Xcstg"])
    return {"prior": prior, "typed": typed, "consist": consist, "gow": gow}


def g_from(blocks, keys, y, W, N):
    mats = [blocks[k] for k in keys if k in blocks and blocks[k].shape[1] > 0]
    if not mats:
        return None
    X = sp.hstack(mats).tocsr()
    return rff.channel_score(X, y, W, N, sparse=True)


def main():
    sp5 = OUT / "raw_method_scores.pkl"; st = OUT / "online_jit_streams_v5.pkl"
    ff = OUT / "final_fusion_results.pkl"
    if not (sp5.exists() and st.exists() and ff.exists()):
        print(f"[{PROJECT}] missing caches; skip."); return
    rm = pickle.load(open(sp5, "rb")); y = np.asarray(rm["y"]); N = rm["N"]; W = rm["warmup"]
    S = pickle.load(open(open_path := st, "rb"))
    if not np.array_equal(np.asarray(S["y"]), y):
        print(f"[{PROJECT}] streams not aligned; skip."); return
    scores = {m: np.asarray(rm["scores"]["final"][m], float) for m in rfe.METHODS}
    F = pickle.load(open(ff, "rb"))["F"]
    blocks = build_blocks(S)

    variants = {"F+G (all)": BLOCKS}
    for b in BLOCKS:
        variants[f"$-$ {b}"] = [x for x in BLOCKS if x != b]   # leave-one-out
    for b in BLOCKS:
        variants[f"{b} only"] = [b]

    rows = []
    for name, keys in variants.items():
        g = g_from(blocks, keys, y, W, N)
        sc = dict(scores); sc["G"] = g if g is not None else np.zeros(N)
        m, _, _, _ = rff.eval_subset(sc, F + (["G"] if g is not None else []), y, W, N)
        rows.append((name, {k: float(m[k]) for k in M7}))

    # table
    PM.mkdir(parents=True, exist_ok=True)
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         f"\\caption{{Appendix F: CSTG component ablation on {PROJECT} (leave-one-out "
         r"and single-block), deployed $F{+}G$, 7 metrics.}}",
         f"\\label{{tab:appF_{PROJECT}}}",
         r"\begin{tabular}{l" + "c" * len(M7) + "}", r"\toprule",
         "CSTG variant & " + " & ".join(PP[m] for m in M7) + r" \\ \midrule"]
    for name, m in rows:
        L.append(name + " & " + " & ".join("%.3f" % m[k] for k in M7) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / f"appF_cstg__{PROJECT}.tex").write_text("\n".join(L), encoding="utf-8")

    # figure: Macro-F1 of each variant
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [r[0] for r in rows]; mf1 = [r[1]["Macro_F1"] for r in rows]
    ax.bar(range(len(names)), mf1, color="#8c564b")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Macro-F1"); ax.set_title(f"{PROJECT}", weight="bold")
    ax.grid(axis="y", color="#EEE"); ax.set_axisbelow(True); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(PM / f"appF_cstg__{PROJECT}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"[{PROJECT}] wrote CSTG component ablation (F + 4 LOO + 4 single).")


if __name__ == "__main__":
    main()
