"""
Build the baseline-comparison figures + docs/Baselines_results.tex.

Combines the ORIGINAL learned/naive baselines (LR, RF, HGB, + best-of-them, and the
naive ALL1/ALL0/RATE references) with the NEW baselines (LApredict, Deeper, JITLine),
all evaluated under the identical online prequential protocol.

Per project it emits, under outputs/<project>/figures/v4/baselines_extra/:
  fig_baselines_stream_<metric>.{png,pdf}   -- all 7 learned baselines on one axis
(one figure per of the 7 metrics; the accurate window=150 trajectory).

Into docs/Baselines_results.tex it writes:
  * a per-project 7-metric table of ALL baselines (7 learned + 3 naive)
  * a cross-project aggregate (macro mean) table
  * the per-project stream figures (Macro-F1 + Buggy-F1 shown inline)
  * explanation paragraphs

Reads (per project, cache-only, no Neo4j):
  outputs/<p>/baseline_results.pkl         (LR/RF/HGB/naive + best)
  outputs/<p>/baseline_extra_results.pkl   (LR/RF/HGB re-traj + LApredict/Deeper/JITLine)

Run: python baselines/make_baselines_doc.py
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)

PROJECTS = [("ActiveMQ", "activemq"), ("Groovy", "groovy"), ("HDFS", "hadoop-hdfs"),
            ("MapReduce", "hadoop-mapreduce"), ("Kafka", "kafka"), ("Spark", "spark"),
            ("Zeppelin", "zeppelin"), ("Zookeeper", "zookeeper")]
M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]
M3 = ["Macro_F1", "G_Mean", "AUC"]           # metrics for the final KG-vs-all table
# all 14 ApacheJIT projects, in a fixed display order (pending ones get 0.000)
ALL14 = ["ActiveMQ", "Camel", "Cassandra", "Flink", "Groovy", "HDFS", "HBase",
         "Hive", "Ignite", "MapReduce", "Kafka", "Spark", "Zeppelin", "Zookeeper"]
FOLDER14 = {"ActiveMQ": "activemq", "Camel": "camel", "Cassandra": "cassandra",
            "Flink": "flink", "Groovy": "groovy", "HDFS": "hadoop-hdfs",
            "HBase": "hbase", "Hive": "hive", "Ignite": "ignite",
            "MapReduce": "hadoop-mapreduce", "Kafka": "kafka", "Spark": "spark",
            "Zeppelin": "zeppelin", "Zookeeper": "zookeeper"}
M7P = {"Precision": "Prec.", "Recall": "Rec.", "Macro_F1": "Macro-F1",
       "Buggy_F1": "Buggy-F1", "G_Mean": "G-Mean", "AUC": "AUC", "ACC": "Acc."}

# display order + pretty names for the seven LEARNED baselines
LEARNED = [("B_LR", "LR"), ("B_RF", "RF"), ("B_HGB", "GBoost"),
           ("B_LAPREDICT", "LApredict"), ("B_DEEPER", "Deeper"),
           ("B_JITLINE", "JITLine"), ("B_DEEPJIT", "DeepJIT")]
NAIVE = [("B_ALL1", "All-buggy"), ("B_ALL0", "All-benign"), ("B_RATE", "Base-rate")]
# stream-plot styling
STYLE = {
    "B_LR":        ("#7f4fa0", "-",  1.6),
    "B_RF":        ("#1f77b4", "-",  1.6),
    "B_HGB":       ("#2ca02c", "-",  1.6),
    "B_LAPREDICT": ("#d62728", "--", 1.8),
    "B_DEEPER":    ("#ff7f0e", "--", 1.8),
    "B_JITLINE":   ("#17becf", "-",  2.0),
    "B_DEEPJIT":   ("#8c564b", "-.", 2.0),
}
PRETTY = dict(LEARNED)


def _load(folder):
    d = OUTP / folder
    old = pickle.load(open(d / "baseline_results.pkl", "rb"))
    ext = pickle.load(open(d / "baseline_extra_results.pkl", "rb"))
    return old, ext


def _deepjit(folder):
    """DeepJIT metrics if the Kaggle result pkl was placed under outputs/<p>/."""
    p = OUTP / folder / f"deepjit_{folder}_results.pkl"
    if not p.exists():
        return None
    d = pickle.load(open(p, "rb"))
    m = d.get("metrics", {})
    return {mm: float(m.get(mm, np.nan)) for mm in M7} if m else None


def all_metrics(folder):
    """{baseline_key: {metric: value}} merging old (naive) + extra (all 7 learned)
    + DeepJIT (if its Kaggle result pkl is present).
    Learned baselines are re-scored on the COMMON evaluation window [ev0, N) (fairness
    fix: identical commits as KG-Commit F+G) when their raw predictions are available."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
    try:
        import common_window as _cw
        ev0 = _cw.common_ev0(folder)
    except Exception:
        _cw = None; ev0 = None
    old, ext = _load(folder)
    R = {}
    raws = ext.get("raws", {})
    for k, m in ext["baselines"].items():           # LR/RF/HGB re-scored + 3 new
        if _cw is not None and ev0 is not None and k in raws:
            cm = _cw.metrics_from_raw(raws[k], ev0)     # common-window re-score
            R[k] = {mm: float(cm.get(mm, np.nan)) for mm in M7} if cm else \
                   {mm: float(m.get(mm, np.nan)) for mm in M7}
        else:
            R[k] = {mm: float(m.get(mm, np.nan)) for mm in M7}
    dj = _deepjit(folder)
    if dj is not None:
        R["B_DEEPJIT"] = dj
    for k, _ in NAIVE:                                # naive refs only in old file
        if k in old["baselines"]:
            R[k] = {mm: float(old["baselines"][k].get(mm, np.nan)) for mm in M7}
    R["_best"] = old.get("best_baseline")
    return R


def fg_metrics(folder):
    """Deployed KG-Commit F+G metrics for a project, or None if not available."""
    p = OUTP / folder / "final_fusion_results.pkl"
    if not p.exists():
        return None
    ff = pickle.load(open(p, "rb"))
    m = ff.get("part2", {}).get("F+G", {}).get("metrics")
    if not m:
        return None
    return {mm: float(m.get(mm, np.nan)) for mm in M3}


def make_stream_figs(folder):
    d = OUTP / folder
    ext = pickle.load(open(d / "baseline_extra_results.pkl", "rb"))
    trajs = dict(ext["trajs"])                        # LR/RF/HGB + LApredict/Deeper/JITLine
    djp = d / f"deepjit_{folder}_results.pkl"         # DeepJIT (from Kaggle), if present
    if djp.exists():
        dd = pickle.load(open(djp, "rb"))
        if dd.get("traj"):
            trajs["B_DEEPJIT"] = dd["traj"]
    figdir = d / "figures" / "v4" / "baselines_extra"; figdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for mk in M7:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        drawn = False
        for key, lab in LEARNED:
            t = trajs.get(key)
            if not t or mk not in t:
                continue
            col, ls, lw = STYLE[key]
            ax.plot(t["idx"], t[mk], color=col, ls=ls, lw=lw, label=lab)
            drawn = True
        if not drawn:
            plt.close(fig); continue
        ax.set_title(f"{M7P[mk]} over the online stream -- baselines [{folder}]",
                     fontsize=12, weight="bold")
        ax.set_xlabel("commit index (chronological)")
        ax.set_ylabel(f"rolling {M7P[mk]} (window=150, stride=25)")
        ax.grid(color="#ECECEC"); ax.set_axisbelow(True)
        ax.legend(fontsize=9, frameon=False, ncol=3, loc="best")
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(figdir / f"fig_baselines_stream_{mk}.{e}", bbox_inches="tight")
        plt.close(fig); n += 1
    return n, figdir


def fmt(x):
    return "--" if x != x else f"{x:.3f}"


def bold_max(vals):
    """Return list of formatted strings with the max bolded (per column use)."""
    arr = np.array([v if v == v else -np.inf for v in vals])
    bi = int(np.argmax(arr)) if len(arr) else -1
    out = []
    for i, v in enumerate(vals):
        s = fmt(v)
        out.append(r"\textbf{" + s + "}" if i == bi else s)
    return out


def main():
    data = {}
    for disp, folder in PROJECTS:
        try:
            data[disp] = all_metrics(folder)
            make_stream_figs(folder)
        except Exception as e:
            print(f"skip {folder}: {e}")

    L = []; ap = L.append
    ap(r"\documentclass[11pt]{article}")
    ap(r"\usepackage[margin=1.9cm]{geometry}")
    ap(r"\usepackage{booktabs,graphicx,amsmath,array,longtable,xcolor,multirow}")
    ap(r"\graphicspath{{../outputs/}}")
    ap(r"\title{KG-Commit: Baseline Suite under the Online Protocol}")
    ap(r"\date{\today}")
    ap(r"\begin{document}\maketitle")
    ap(r"""
This document reports every JIT-defect-prediction baseline used in this work,
evaluated under the \emph{identical} online prequential protocol as KG-Commit (same
warm-up fraction, block size, refit cadence, leakage-free online-tuned operating
point, and the same seven metrics). The suite has two groups. \textbf{Classic
tabular baselines} on the twelve ApacheJIT change metrics: Logistic Regression (LR),
Random Forest (RF), and HistGradientBoosting (GBoost). \textbf{Literature baselines}
re-implemented under our protocol: \emph{LApredict} (LR on added-lines only;
Zeng et al.\ 2021), \emph{Deeper} (a DBN-style nonlinear transform of the change
metrics followed by LR; Yang et al.\ 2015, reimplemented as a small autoencoder
transform), and \emph{JITLine} (Random Forest on expert metrics concatenated with
bag-of-token diff features; Pornprasit et al.\ 2021, prediction part). Three
non-learned references (All-buggy, All-benign, historical Base-rate) bound the
trivial end. All literature baselines are re-evaluated in our setup rather than
copied from their papers, because those papers use random or time-split evaluation,
not the fully-online protocol; a copied number would not be comparable.
""")

    # ---- per-baseline definitions + implementation ----
    ap(r"\section{The baselines: what each is, and how we implemented it}")
    ap(r"""Every baseline is evaluated by the \emph{same} online driver as the KG
models. Concretely, for a project's chronologically-ordered commits we warm up on
the first fraction (the protocol's \texttt{WARMUP\_FRAC}), then walk the remainder in
fixed-size blocks: predict the next block using only strictly-past data, reveal its
labels, and periodically refit on the expanding past window; features are scaled
using past-only statistics. The decision threshold is tuned online on past
predictions (to maximise Buggy-F1), and the seven metrics are read at that
leakage-free operating point. This paragraph applies to every learned baseline
below; they differ only in \emph{features} and \emph{classifier}.""")

    ap(r"\subsection*{Classic tabular baselines (on the 12 change metrics)}")
    ap(r"""\textbf{Feature set.} All three use the twelve ApacheJIT change metrics:
\texttt{la, ld} (added/deleted lines), \texttt{nf, nd, ns} (files/directories/
subsystems touched), \texttt{ent} (change entropy), \texttt{ndev} (developers of the
touched files), \texttt{age}, \texttt{nuc} (unique last changes), and
\texttt{aexp, arexp, asexp} (author experience). No graph, no commit text.
\begin{itemize}
  \item \textbf{LR} --- \emph{What:} the canonical linear JIT model, logistic
  regression over the change metrics. \emph{How:} class-weighted
  \texttt{LogisticRegression} (\texttt{lbfgs}), scaled features, refit each block on
  the expanding past. This is the same classifier family as KG-Commit's own fusion
  head, so it is our controlled, like-for-like baseline.
  \item \textbf{RF} --- \emph{What:} a nonlinear tabular learner capturing feature
  interactions the linear model misses. \emph{How:} class-weighted
  \texttt{RandomForestClassifier} ($200$ trees), same online refit.
  \item \textbf{GBoost} --- \emph{What:} a strong gradient-boosted tabular model, the
  toughest change-metric bar. \emph{How:} \texttt{HistGradientBoostingClassifier},
  same online refit.
\end{itemize}""")

    ap(r"\subsection*{Literature baselines (re-implemented under our protocol)}")
    ap(r"""\begin{itemize}
  \item \textbf{LApredict} (Zeng et al.\ 2021). \emph{What:} the deliberately-minimal
  baseline from ``deep learning JIT is overrated''---logistic regression on a
  \emph{single} feature, the number of added lines (\texttt{la}). It exists to test
  whether elaborate models actually beat a one-feature linear classifier under fair
  evaluation. \emph{How:} \texttt{LogisticRegression} on the \texttt{la} column alone,
  run through the identical online driver.
  \item \textbf{Deeper} (Yang et al.\ 2015). \emph{What:} an early deep-learning JIT
  model that learns a \emph{nonlinear representation} of the change metrics with a
  Deep Belief Network, then classifies with logistic regression. \emph{How:} because
  maintained DBN/RBM libraries are effectively unavailable, we reimplement the DBN
  step---as modern reimplementations of Deeper do---as a small unsupervised
  \emph{autoencoder} (a two-layer \texttt{MLPRegressor} reconstructing its input)
  that produces a nonlinear code of the twelve metrics; that code, concatenated with
  the raw metrics, feeds a class-weighted logistic regression. It runs on a
  twelve-dimensional input, so it is CPU-cheap and needs no GPU. Everything is fit
  past-only inside the same online loop.
  \item \textbf{JITLine} (Pornprasit et al.\ 2021). \emph{What:} a strong
  non-deep model---a random forest on \emph{expert metrics plus bag-of-token features
  extracted from the commit diff}---accompanied in the original by a LIME-based
  line-localization step. \emph{How:} we implement the \emph{prediction} component
  (commit-level, which is what our task evaluates): the twelve expert metrics are
  concatenated with per-commit token-count features obtained by running the project's
  own commit-text tokenizer over each commit's diff; the token vocabulary is fit on
  the warm-up window only (leakage-free) and the combined matrix feeds a
  class-weighted random forest under the same online refit. We do not reproduce the
  LIME line-localization, which is orthogonal to commit-level defect prediction.
  \item \textbf{DeepJIT} (Hoang et al.\ 2019). \emph{What:} the most-cited
  \emph{deep-learning} JIT model---a convolutional neural network that learns
  semantic features directly from the commit message and the code diff (no
  hand-crafted metrics). It is the representative of the ``learned code
  representation'' family (alongside CC2Vec, JIT-Fine, CodeReviewer). \emph{How:} we
  train a compact from-scratch CNN (message-CNN $+$ code-CNN fused by an MLP,
  class-balanced loss) under the \emph{identical} online protocol---warm-up training,
  per-block prediction, and \textbf{faithful online refit} (retrain on the expanding
  past every three blocks), with the token vocabulary fit on the warm-up window only.
  Because per-block retraining of a neural model is GPU-intensive, DeepJIT is
  evaluated on a \emph{representative subset} of projects on a single T4 GPU; the
  metric code is byte-identical to the repository's, so its numbers are directly
  comparable. Rows without a DeepJIT value are pending that evaluation. This is the
  learned-representation baseline that answers ``does a deep model on the raw diff
  beat the graph representation under a fair online protocol?''
\end{itemize}""")

    ap(r"\subsection*{A note on Yan et al.\ (relationship to our LR baseline)}")
    ap(r"""Yan et al.'s work~[48] combines a \emph{defect-prediction} model---logistic
regression on expert change features---with a separate \emph{defect-localization}
model based on $N$-grams. Its prediction component is therefore, in our setting,
\emph{identical in spirit to our \textbf{LR} baseline}: logistic regression over the
ApacheJIT expert change metrics, trained under our online protocol. We consequently
treat \textbf{LR} as the representative of Yan et al.'s prediction model rather than
adding a redundant separate row; the $N$-gram \emph{localization} component is a
line-level task orthogonal to the commit-level prediction this paper evaluates and is
not reproduced. This avoids double-counting the same method under two names.""")

    ap(r"\subsection*{Non-learned references}")
    ap(r"""Three trivial predictors bound the bottom of the table:
\textbf{All-buggy} (predict every commit defective), \textbf{All-benign} (predict
none), and \textbf{Base-rate} (score each commit by the historical bug rate of all
strictly-earlier commits). They are not competitors; they calibrate what a metric
value means on each project's class balance (e.g.\ a high raw F1 that All-buggy also
achieves signals a metric dominated by imbalance).""")

    # ---- per-project tables ----
    ap(r"\section{Per-project results (all baselines, seven metrics)}")
    for disp in [d for d, _ in PROJECTS if d in data]:
        R = data[disp]
        ap(r"\subsection*{" + disp + "}")
        ap(r"\begin{center}\small\begin{tabular}{l" + "r" * 7 + "}")
        ap(r"\toprule")
        ap("Baseline & " + " & ".join(M7P[m] for m in M7) + r" \\ \midrule")
        # learned block with per-column bold max (over learned only)
        keys = [k for k, _ in LEARNED if k in R]
        for m_i, m in enumerate(M7):
            pass
        # build rows; bold the best learned per metric
        colvals = {m: [R[k][m] for k in keys] for m in M7}
        bolded = {m: bold_max(colvals[m]) for m in M7}
        for r_i, k in enumerate(keys):
            lab = PRETTY[k]
            cells = [bolded[m][r_i] for m in M7]
            ap(lab + " & " + " & ".join(cells) + r" \\")
        ap(r"\midrule")
        for k, lab in NAIVE:
            if k in R:
                ap(lab + " & " + " & ".join(fmt(R[k][m]) for m in M7) + r" \\")
        ap(r"\bottomrule\end{tabular}\end{center}")
        # inline the two decision-relevant stream figures
        base = f"{[f for d, f in PROJECTS if d == disp][0]}/figures/v4/baselines_extra"
        ap(r"\begin{center}")
        ap(r"\includegraphics[width=0.49\linewidth]{" + base + r"/fig_baselines_stream_Macro_F1.png}")
        ap(r"\includegraphics[width=0.49\linewidth]{" + base + r"/fig_baselines_stream_Buggy_F1.png}")
        ap(r"\end{center}")

    # ---- aggregate ----
    ap(r"\section{Cross-project aggregate (macro mean over projects)}")
    ap(r"\begin{center}\small\begin{tabular}{l" + "r" * 7 + "}")
    ap(r"\toprule")
    ap("Baseline & " + " & ".join(M7P[m] for m in M7) + r" \\ \midrule")
    allkeys = [k for k, _ in LEARNED]
    aggcol = {m: [] for m in M7}
    agg = {}
    for k in allkeys:
        agg[k] = {}
        for m in M7:
            vals = [data[d][k][m] for d in data if k in data[d]]
            agg[k][m] = float(np.nanmean(vals)) if vals else np.nan
            aggcol[m].append(agg[k][m])
    bolded = {m: bold_max(aggcol[m]) for m in M7}
    for r_i, k in enumerate(allkeys):
        ap(PRETTY[k] + " & " + " & ".join(bolded[m][r_i] for m in M7) + r" \\")
    ap(r"\bottomrule\end{tabular}\end{center}")

    ap(r"""
\section{Reading the results}
Across the eight projects a consistent ordering emerges. \textbf{LApredict}, using a
single feature, is the weakest learned baseline---confirming that added-line count
alone is a poor discriminator under strict online evaluation. \textbf{Deeper} and
\textbf{JITLine} are competitive with the strong tabular models (RF/GBoost): Deeper's
nonlinear transform of the change metrics recovers most of the RF/GBoost advantage
over plain LR, and JITLine's diff-token features add project-dependent signal on top
of the expert metrics. None of the literature baselines dominates the tabular
RF/GBoost uniformly, which is itself an informative result: on commit-level metrics
under a leakage-free online protocol, the classic strong tabular learners remain a
very high bar---the same conclusion LApredict's authors drew about deep JIT models.
This is exactly why KG-Commit is compared against the \emph{best} of this suite per
project, and why the like-for-like LR comparison (identical classifier family) is the
scientifically controlled headline.
""")

    # ---- FINAL: KG-Commit (F+G) vs ALL baselines, Macro-F1 / G-Mean / AUC, arrows ----
    ap(r"\section{KG-Commit vs.\ all baselines (final comparison)}")
    ap(r"""Table~\ref{tab:kg_vs_all_baselines} contrasts the deployed KG-Commit model
$F{+}G$ against every learned baseline on the three balance-oriented metrics
(Macro-F1, G-Mean, AUC). Each baseline cell carries an arrow: a green
\textcolor{green!55!black}{$\uparrow$} means $F{+}G$ \emph{beats} that baseline on
that metric for that project, a red \textcolor{red!75!black}{$\downarrow$} means it
does not. Projects still at $0.000$ are pending completion of the full pipeline.""")

    # gather values for all 14 projects
    # learned baselines for the final table; include DeepJIT only if any project
    # actually has its result (so the column doesn't appear as all-empty pre-Kaggle).
    have_deepjit = any("B_DEEPJIT" in (data.get(d) or {}) for d in data)
    NB = [x for x in LEARNED if x[0] != "B_DEEPJIT" or have_deepjit]
    ncols = 3 + 3 * len(NB)                    # KG(3) + each baseline(3)
    ap(r"\begin{table*}[!htbp]\centering")
    ap(r"\caption{KG-Commit ($F{+}G$) vs.\ all baselines under the identical online "
       r"protocol (Macro-F1, G-Mean, AUC). Arrows compare $F{+}G$ against each "
       r"baseline per project: \textcolor{green!55!black}{$\uparrow$} $F{+}G$ better, "
       r"\textcolor{red!75!black}{$\downarrow$} worse. $F{+}G$ values in \textbf{bold}. "
       r"Projects at $0.000$ are pending.}")
    ap(r"\label{tab:kg_vs_all_baselines}")
    ap(r"\scriptsize\setlength{\tabcolsep}{2.5pt}")
    ap(r"\resizebox{\textwidth}{!}{%")
    ap(r"\begin{tabular}{l *{" + str(1 + len(NB)) + r"}{ccc}}")
    ap(r"\toprule")
    hdr1 = [r"\multirow{2}{*}{\textbf{Project}}",
            r"\multicolumn{3}{c}{\textbf{KG-Commit ($F{+}G$)}}"]
    for _, lab in NB:
        hdr1.append(r"\multicolumn{3}{c}{\textbf{" + lab + "}}")
    ap(" & ".join(hdr1) + r" \\")
    cm = []
    start = 2
    for _ in range(1 + len(NB)):
        cm.append(r"\cmidrule(lr){" + f"{start}-{start+2}" + "}")
        start += 3
    ap(" ".join(cm))
    sub = [""] + ["MF1 & G-m & AUC"] * (1 + len(NB))
    ap(" & ".join(sub) + r" \\")
    ap(r"\midrule")

    def z3():
        return {m: 0.0 for m in M3}

    for disp in ALL14:
        folder = FOLDER14[disp]
        fg = fg_metrics(folder)
        bl = None
        try:
            bl = all_metrics(folder)
        except Exception:
            bl = None
        have = fg is not None and bl is not None
        fgv = fg if have else z3()
        cells = [r"\textbf{" + f"{fgv[m]:.3f}" + "}" for m in M3]
        for k, _ in NB:
            bv = bl[k] if (have and k in bl) else z3()
            for m in M3:
                v = bv[m] if have else 0.0
                if have:
                    arrow = (r"\,\textcolor{green!55!black}{$\uparrow$}"
                             if fgv[m] >= v - 1e-9
                             else r"\,\textcolor{red!75!black}{$\downarrow$}")
                else:
                    arrow = ""
                cells.append(f"{v:.3f}" + arrow)
        ap(disp + " & " + " & ".join(cells) + r" \\")
    ap(r"\bottomrule\end{tabular}%")
    ap(r"}")
    ap(r"\end{table*}")

    ap(r"\end{document}")

    (DOCS / "Baselines_results.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {DOCS/'Baselines_results.tex'}  ({len(data)} projects)")


if __name__ == "__main__":
    main()
