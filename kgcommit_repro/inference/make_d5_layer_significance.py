"""
D5: paired significance of each representation layer, on MACRO-F1.

The stored significance.json tests ROC-AUC only, via DeLong. Macro-F1 is the paper's
primary metric and is threshold-dependent, so it needs a paired bootstrap at the
online-tuned operating point rather than a closed-form AUC test.

For each project and each of the five inference methods we resample commits with
replacement (B iterations), recompute Macro-F1 for two representations on each
resample, and take the distribution of the paired difference. Reported per project:
the mean gain over Core, aggregated across methods, and the fraction of methods for
which the paired difference is significant after Holm correction across the five
methods within that project.

Comparisons: Core+AST vs Core, and the full Core+AST+CSTG vs Core.

Cache-only (raw_method_scores.pkl). No Neo4j.

Run:  python inference/make_d5_layer_significance.py [--B 2000]
Out:  Paper/paper_material/discussions/D5_significance/tab_layer_significance_mf1.tex
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from paper_projects import ACTIVE as PROJECTS  # noqa: E402
from online_jit import online_decisions  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "discussions" / "D5_significance"

WARMUP, GAP, INIT = 0.20, 50, 300
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]


def _macro_f1(y, p):
    yhat = online_decisions(p, y, gap=GAP)
    return f1_score(y, yhat, average="macro", zero_division=0)


def paired_boot(y, pa, pb, B, rng):
    """Bootstrap distribution of MacroF1(a) - MacroF1(b) over resampled commits."""
    n = len(y)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            out[b] = 0.0
            continue
        out[b] = _macro_f1(ys, pa[idx]) - _macro_f1(ys, pb[idx])
    return out


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        adj[i] = min(1.0, run)
    return adj


def analyse(folder, B, rng):
    rp = OUTP / folder / "raw_method_scores.pkl"
    if not rp.exists():
        return None
    raw = pickle.load(open(rp, "rb"))
    if not {"core", "ast", "final"} <= set(raw["scores"]):
        return None
    y_all = np.asarray(raw["y"], int)
    N = int(raw["N"])
    W = int(N * WARMUP)
    ev = np.arange(W + INIT, N)
    y = y_all[ev]

    res = {}
    for which in ("ast", "final"):
        gains, pvals = [], []
        for m in METHODS:
            pa = np.clip(np.nan_to_num(np.asarray(raw["scores"][which][m], float)[ev],
                                       nan=float(y.mean())), 0, 1)
            pb = np.clip(np.nan_to_num(np.asarray(raw["scores"]["core"][m], float)[ev],
                                       nan=float(y.mean())), 0, 1)
            d = paired_boot(y, pa, pb, B, rng)
            gains.append(float(d.mean()))
            # two-sided bootstrap p: how often the difference crosses zero
            p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
            pvals.append(min(1.0, max(p, 1.0 / B)))
        adj = holm(np.array(pvals))
        res[which] = {"gain": float(np.mean(gains)),
                      "n_sig": int((adj < 0.05).sum()),
                      "n_sig_pos": int(((adj < 0.05) &
                                        (np.array(gains) > 0)).sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    PM.mkdir(parents=True, exist_ok=True)

    rows = []
    for disp, folder in PROJECTS:
        r = analyse(folder, args.B, rng)
        if r:
            rows.append((disp, r))
            print(f"  {disp}: AST {r['ast']['gain']:+.3f} "
                  f"({r['ast']['n_sig_pos']}/5 sig+)   "
                  f"CSTG {r['final']['gain']:+.3f} "
                  f"({r['final']['n_sig_pos']}/5 sig+)")
    if not rows:
        print("no data")
        return

    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{D5: paired significance of each representation layer on "
         r"\textbf{Macro-F1}, the primary metric. Each cell is the mean paired gain over "
         r"the bare Core graph across the five inference methods, from a per-commit "
         rf"bootstrap ($B={args.B}$) at the online-tuned operating point, with Holm "
         r"correction across methods within a project. The count in parentheses is the "
         r"number of methods (out of five) for which the gain is both positive and "
         r"Holm-significant.}",
         r"\label{tab:d5_layer_mf1}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"Project & Core$+$AST $-$ Core & Core$+$AST$+$CSTG $-$ Core \\ \midrule"]
    for disp, r in rows:
        L.append(f"{disp} & {r['ast']['gain']:+.3f} ({r['ast']['n_sig_pos']}/5)"
                 f" & {r['final']['gain']:+.3f} ({r['final']['n_sig_pos']}/5) " + r"\\")
    ga = np.mean([r["ast"]["gain"] for _, r in rows])
    gf = np.mean([r["final"]["gain"] for _, r in rows])
    L.append(r"\midrule \textbf{Mean} & \textbf{%+.3f} & \textbf{%+.3f} \\" % (ga, gf))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / "tab_layer_significance_mf1.tex").write_text("\n".join(L) + "\n",
                                                       encoding="utf-8")
    print(f"\nwrote {PM / 'tab_layer_significance_mf1.tex'}")
    print(f"mean gain: AST {ga:+.3f}, CSTG {gf:+.3f}")


if __name__ == "__main__":
    main()
