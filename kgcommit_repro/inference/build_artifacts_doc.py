"""
Assemble the standalone artifact document for the Experimental Results revision.
===============================================================================

Wraps the generated pieces into ONE compilable LaTeX file:

  outputs/tables/sig_tables.tex       15 significance tables (items 1 + 2)
  outputs/tables/seed_baselines.json  baseline seed variance (item 3)
  outputs/tables/seed_audit_kg.json   KG-Commit seed diagnosis (item 4)

Out: Paper/ResultsDiscussionsDraft/artifacts_for_revision.tex
Run: python inference/build_artifacts_doc.py
"""
import json
import os
os.environ.setdefault("KGC_PROJECT", "zookeeper")
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kgc_paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
DRAFT = ROOT / "Paper" / "ResultsDiscussionsDraft"

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]
DISP = {p: p.capitalize() for p in PROJECTS}
DISP.update({"activemq": "ActiveMQ", "hbase": "HBase"})

PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=1.9cm,landscape]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,multirow,array,longtable}
\usepackage[table,dvipsnames]{xcolor}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{iftex}
\ifPDFTeX\else
  \usepackage{fontspec}
  \setmainfont{TeX Gyre Termes}
  \setmonofont{TeX Gyre Cursor}
\fi
\setlength{\parskip}{2pt}
\title{\vspace{-1.4cm}\bfseries Supplementary artifacts for the
Experimental Results revision\\[4pt]
\large KG-Commit --- significance across five metrics, pooled totals,
and seed sensitivity}
\author{}
\date{\today}
\begin{document}
\maketitle
\vspace{-8mm}
"""


def seed_baseline_table():
    f = OUTP / "tables" / "seed_baselines.json"
    if not f.exists():
        return "% seed_baselines.json missing\n"
    d = json.loads(f.read_text(encoding="utf-8"))

    dj = {}
    g = OUTP / "DeepJIT_baseline_results" / "grid_summary.csv"
    if g.exists():
        import pandas as pd
        df = pd.read_csv(g)

        def f1(tp, fp, fn):
            a = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            return 2 * a * r / (a + r) if (a + r) else 0.0
        df["Macro_F1"] = df.apply(
            lambda r: (f1(r.tp, r.fp, r.fn) + f1(r.tn, r.fn, r.fp)) / 2, axis=1)
        s = df[(df.M == 200) & (df.gap_commits == 50) & (df.warmup_ratio == 0.05)]
        s = s.assign(proj=s.project.str.replace("apache/", "", regex=False))
        for p, grp in s[s.proj.isin(PROJECTS)].groupby("proj"):
            dj[p] = (float(grp.Macro_F1.mean()), float(grp.Macro_F1.std()))

    L = [r"\begin{table}[h]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{Seed sensitivity of the baselines on Macro-F1: mean $\pm$ "
         r"standard deviation over the five protocol seeds, re-run per project "
         r"under the deployed protocol ($K{=}0.05$, $M{=}200$, $G{=}50$). "
         r"LApredict is omitted from the re-run because it shares LR's "
         r"deterministic solver on a single feature; JITLine-online cannot be "
         r"re-seeded from cache because its featurisation re-tokenises the diff. "
         r"DeepJIT is taken from its own five-seed grid.}",
         r"\label{tab:seed_baselines}",
         r"\begin{tabular}{l cccc}", r"\toprule",
         r"Project & LR & HGB & RF & DeepJIT \\", r"\midrule"]
    acc = {k: [] for k in ("LR", "HGB", "RF", "DeepJIT")}
    for p in PROJECTS:
        if p not in d:
            continue
        cells = []
        for m in ("LR", "HGB", "RF"):
            e = d[p][m]["Macro_F1"]
            cells.append(f"{e['mean']:.3f} $\\pm$ {e['sd']:.4f}")
            acc[m].append(e["sd"])
        if p in dj:
            cells.append(f"{dj[p][0]:.3f} $\\pm$ {dj[p][1]:.4f}")
            acc["DeepJIT"].append(dj[p][1])
        else:
            cells.append("--")
        L.append(f"{DISP[p]} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\textbf{Mean sd} & " + " & ".join(
        (f"{np.mean(acc[m]):.5f}" if acc[m] else "--")
        for m in ("LR", "HGB", "RF", "DeepJIT")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def seed_kg_section():
    f = OUTP / "tables" / "seed_audit_kg.json"
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    q1 = d.get("Q1_rng_reachability", {})
    q2 = d.get("Q2_downstream_invariance", {})
    q3 = d.get("Q3_embedding_sensitivity", {})
    ok = [p for p, v in q2.items() if "sd" in v]
    maxsd = max((q2[p]["sd"] for p in ok), default=float("nan"))
    dep = sorted({v.get("deployed", "?") for v in q3.values() if "deployed" in v})

    return rf"""\section{{Why KG-Commit's measured seed sd is exactly zero}}
\label{{sec:seedkg}}

\texttt{{run\_seed\_robustness.py}} reports a standard deviation of
$0.000000$ for KG-Commit on every project. That number is correct as printed,
but it does \emph{{not}} mean what it appears to mean, and the distinction
matters for the paper. We audited it from the cached artifacts, without the
graph, and separate three questions.

\paragraph{{Q1. The seed never reached the stochastic components.}}
The seed loop sets the \emph{{legacy global}} generator,
\texttt{{np.random.seed(sd)}}, and then re-derives the five methods. But the two
stochastic call sites do not consult that generator:
\texttt{{dw\_embed}} calls \texttt{{randomized\_svd(\dots, random\_state=0)}},
and \texttt{{kge\_embed}} calls \texttt{{np.random.default\_rng(seed)}} with its
default \texttt{{seed=0}}. NumPy's \texttt{{default\_rng}} and scikit-learn's
\texttt{{random\_state}} both ignore the legacy global state. Measured directly:
identical output across all five seeds for
\texttt{{randomized\_svd}} ({q1.get('dw_randomized_svd_identical_across_seeds')})
and for \texttt{{default\_rng}} ({q1.get('kge_default_rng_identical_across_seeds')}).
\textbf{{All five ``seeds'' therefore ran the same random stream.}} This is a
defect in the experiment: it could not have detected variance had any existed.

\paragraph{{Q2. Everything downstream is genuinely deterministic.}}
Holding the embedding scores at their cached values, we re-ran the complete
deployed fusion --- prequential stacker, CSTG channel $G$, and the online
threshold tuner --- under all five seeds on {len(ok)} of 11 projects. The
standard deviation is ${maxsd:.10f}$ at the maximum across every project and
metric. The stacker and the $G$ channel are logistic regressions fitted with
\texttt{{lbfgs}}/\texttt{{liblinear}}, convex solvers that take no
\texttt{{random\_state}}; the threshold tuner is a deterministic sweep. Nothing
after the embeddings can introduce seed variance.

\paragraph{{Q3. The embeddings cannot reach the deployed model at all.}}
The deployed fusion is {', '.join(dep)} on every project. RN (neighbourhood
voting) and PPR (personalised PageRank) are closed-form over the graph: they are
computed from the topology rather than fitted, and return identical scores on
every execution. DW and DistMult are evaluated in RQ3 but are \emph{{not}} part
of the deployed configuration, so even a correctly plumbed seed could not move
the reported numbers.

\paragraph{{Conclusion.}}
The reported $0.000000$ is right for the wrong reason. The experiment was
incapable of measuring variance (Q1), but the deployed model is deterministic
regardless, and for two independent reasons: nothing downstream of the
embeddings is stochastic (Q2), and the embeddings are not in the deployed
fusion (Q3). The paper's claim --- that KG-Commit reproduces exactly across
seeds --- therefore stands on Q2 and Q3, which are established here, rather than
on the seed loop, which is not evidence.

\paragraph{{The fix, for completeness.}}
Two lines: pass \texttt{{random\_state=seed}} into \texttt{{randomized\_svd}}
inside \texttt{{dw\_embed}}, and \texttt{{seed=seed}} into \texttt{{kge\_embed}}.
Re-running the corrected loop requires the project's graph to be resident, since
the embeddings are re-derived from the KG; it would change the RQ3 embedding
rows, not the deployed numbers.
"""


def total_rows_section():
    """Total rows for the four existing draft tables that carry Macro/Micro."""
    import numpy as np
    f1 = OUTP / "tables" / "total_rows.json"
    f2 = OUTP / "tables" / "total_rows_rq4.json"
    if not f1.exists():
        return "% total_rows.json missing\n"
    D = json.loads(f1.read_text(encoding="utf-8"))
    R4 = json.loads(f2.read_text(encoding="utf-8")) if f2.exists() else None

    order = ["KGov", "KGpp", "LR", "HGB", "RF", "LApredict", "DeepJIT",
             "JITLine-online"]
    lab = {"KGov": r"KG-Commit ($F_{\mathrm{ov}}{+}G$)",
           "KGpp": r"KG-Commit ($F_{\mathrm{pp}}{+}G$)"}
    have = [m for m in order if m in D["models"]]

    L = [rf"""\section{{Total rows for the existing draft tables}}
\label{{sec:totals}}

Three of the tables in the Results draft report a \emph{{Macro-Mean}} (every
project weighted equally) and a \emph{{Micro-Mean}} (weighted by scored
commits). Neither answers the question a practitioner asks about a portfolio:
\emph{{across every commit we ship, how often is the model right?}} The
\textbf{{Total}} row answers exactly that, over all
{D['n_commits']:,} scored commits from the {D['n_projects']} projects.

\paragraph{{How the pooling is done, and why it matters}}
There are two ways to pool, and they do not agree. Concatenating the raw
\emph{{scores}} and re-tuning one global threshold forces a single operating
point onto projects whose defect rates run from $13.6\%$ (Camel) to $61.7\%$
(Hive); that destroys the per-project calibration every model is tuned under
and measures a deployment nobody runs. We therefore pool the
\emph{{decisions}}: each project keeps the online-tuned threshold it would
actually deploy, and the resulting binary predictions are concatenated into one
confusion matrix. The distinction is not cosmetic --- under score-pooling the
ranking of the top models inverts, which is a textbook Simpson's paradox and is
an artefact of the pooling rule rather than a property of the models.

Threshold-free metrics (AUC, $P_{{\mathrm{{opt}}}}$, ACC@20\%LOC) have no
operating point to preserve, so for those the Total row is the
commit-weighted mean of the per-project values; they are marked $^{{*}}$.

\begin{{table}}[h]\centering\small\setlength{{\tabcolsep}}{{5pt}}
\caption{{\textbf{{Total}} row for Tables~\ref{{tab:t_perf}}: all
{D['n_commits']:,} scored commits pooled by decision, each project retaining its
own online-tuned threshold.}}
\label{{tab:t_perf}}
\begin{{tabular}}{{l ccccc}}
\toprule
Model & Macro-F1 & G-Mean & AUC$^{{*}}$ & $P_{{\mathrm{{opt}}}}^{{*}}$ & ACC@20\%LOC$^{{*}}$ \\
\midrule"""]

    best = {k: max(D["models"][m][k] for m in have)
            for k in ("Macro_F1", "G_Mean", "AUC", "Popt", "ACC20")}
    for m in have:
        r = D["models"][m]
        cells = []
        for k in ("Macro_F1", "G_Mean", "AUC", "Popt", "ACC20"):
            v = r[k]
            s_ = f"{v:.4f}"
            cells.append(rf"\textbf{{{s_}}}" if abs(v - best[k]) < 1e-9 else s_)
        nm = lab.get(m, m)
        if m.startswith("KG"):
            nm = rf"\textbf{{{nm}}}"
        L.append(nm + " & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    if R4:
        st = R4["stages"]
        L.append(rf"""
\begin{{table}}[h]\centering\small\setlength{{\tabcolsep}}{{6pt}}
\caption{{\textbf{{Total}} row for the $F \rightarrow F{{+}}G \rightarrow$
Switch@200 build-up, pooled by decision over the same
{R4['n_commits']:,} commits. The semantic channel and the switch are each worth
a real gain at portfolio level: $+{R4['delta_switch_minus_F']['overall']:.4f}$
Macro-F1 for the overall rule and
$+{R4['delta_switch_minus_F']['per_project']:.4f}$ for the per-project rule,
relative to the graph fusion $F$ alone.}}
\label{{tab:t_rq4}}
\begin{{tabular}}{{l cc cc}}
\toprule
& \multicolumn{{2}}{{c}}{{$F_{{\mathrm{{ov}}}}$ rule}} & \multicolumn{{2}}{{c}}{{$F_{{\mathrm{{pp}}}}$ rule}} \\
\cmidrule(lr){{2-3}}\cmidrule(l){{4-5}}
Stage & Macro-F1 & G-Mean & Macro-F1 & G-Mean \\
\midrule
$F$ alone & {st['F_ov']['Macro_F1']:.4f} & {st['F_ov']['G_Mean']:.4f} & {st['F_pp']['Macro_F1']:.4f} & {st['F_pp']['G_Mean']:.4f} \\
$F{{+}}G$ & {st['FG_ov']['Macro_F1']:.4f} & {st['FG_ov']['G_Mean']:.4f} & {st['FG_pp']['Macro_F1']:.4f} & {st['FG_pp']['G_Mean']:.4f} \\
\textbf{{Switch@200 (deployed)}} & \textbf{{{st['SW_ov']['Macro_F1']:.4f}}} & \textbf{{{st['SW_ov']['G_Mean']:.4f}}} & \textbf{{{st['SW_pp']['Macro_F1']:.4f}}} & \textbf{{{st['SW_pp']['G_Mean']:.4f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}""")

    L.append(r"""
\paragraph{What the Total row shows}
KG-Commit leads on the pooled stream: the per-project fusion is highest on
Macro-F1, G-Mean, AUC and ACC@20\%LOC, and the fixed overall fusion is second
on the first three. JITLine-online is the closest competitor and takes
$P_{\mathrm{opt}}$ by a small margin. The ordering agrees with the Macro- and
Micro-means, which is the reassurance this row is meant to provide: the
advantage is not an artefact of weighting projects equally.""")
    return "\n".join(L)


def main():
    sig = OUTP / "tables" / "sig_tables.tex"
    parts = [PRE]

    parts.append(r"""\section*{What is in this document}
Four artifacts requested for the Experimental Results revision, all computed
from the cached \texttt{final\_final\_run} outputs under the deployed protocol
($K{=}0.05$ warm-up, refresh block $M{=}200$, verification gap $G{=}50$,
adaptive stacking-head initialisation, switch to $F{+}G$ at $S{=}200$):
\begin{enumerate}[leftmargin=*,itemsep=1pt]
  \item Significance of KG-Commit against every baseline on five metrics
        (Macro-F1, G-Mean, AUC, $P_{\mathrm{opt}}$, ACC@20\%LOC), for the
        overall fusion, the per-project fusion, and both side by side.
  \item A \textbf{Total} row in every table: all projects pooled into a single
        stream and evaluated as one project, alongside the existing Macro- and
        Micro-means.
  \item Seed sensitivity for the baselines, not only DeepJIT.
  \item A diagnosis of KG-Commit's reported zero seed variance.
\end{enumerate}
\tableofcontents\newpage
""")

    parts.append(r"\section{Significance across five metrics and three fusion rules}"
                 "\n\\label{sec:sig}\n"
                 r"""Each table pairs KG-Commit against every baseline on one metric.
Per project the paired bootstrap ($B{=}400$) resamples that project's scored
commits and scores every model on the identical resample, so the pairing carries
the inference. The \emph{Macro} and \emph{Micro} rows pair across the eleven
per-project scores; the \textbf{Total} row pools every scored commit into a
single stream and evaluates it as one project, which is neither a macro nor a
micro average. $P_{\mathrm{opt}}$ and ACC@20\%LOC use commit churn
($\mathrm{la}+\mathrm{ld}$) as inspection effort.

""")
    if sig.exists():
        parts.append(sig.read_text(encoding="utf-8"))
    else:
        parts.append("% sig_tables.tex not generated\n")

    parts.append(r"\clearpage" "\n"
                 r"\section{Seed sensitivity of the baselines}" "\n"
                 r"\label{sec:seedbase}" "\n"
                 r"""The paper reports five-seed variance for DeepJIT only. We re-ran the
change-metric baselines under the same five protocol seeds, from the cached
metric block, to establish which of them are actually stochastic.
Random Forest is the most seed-sensitive model in the study; Logistic
Regression is exactly deterministic, as its convex solver implies.

""")
    parts.append(seed_baseline_table())

    parts.append(r"\clearpage" "\n" + total_rows_section())
    parts.append(r"\clearpage" "\n" + seed_kg_section())
    parts.append(r"\end{document}")

    dst = DRAFT / "artifacts_for_revision.tex"
    dst.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
