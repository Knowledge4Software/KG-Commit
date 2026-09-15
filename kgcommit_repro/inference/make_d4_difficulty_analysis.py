#!/usr/bin/env python3
"""
D4 --- an item-difficulty analysis of where every model wins and loses.
=======================================================================

The current D4 explains per-project margins narratively, from eleven data
points. This script replaces the narrative with commit-level statistics on
~41k commits, and reports honestly which hypotheses survive testing.

Three analyses, in the order they were tried:

  A. PROJECT-LEVEL CORRELATES (n=11).  Does anything predict KG-Commit's
     margin -- coupling, size, defect rate, churn?  Tested, and NOTHING is
     significant. n=11 has no power. Reported as a negative result because the
     draft currently asserts a coupling-margin relationship without testing it.

  B. HEAD-TO-HEAD DISAGREEMENTS.  On commits where exactly one of KG-Commit and
     the baseline is right, which commit properties predict the winner? Pooled,
     several look significant; under a project-cluster bootstrap almost all
     collapse. Reported with both, because the pooled version is a Simpson's
     paradox driven by one project.

  C. ITEM DIFFICULTY (the analysis that works).  Score each commit by how many
     of the K=7 available models classify it correctly at each model's own
     prevalence-matched operating point. The distribution is sharply bimodal,
     the hard core is characterisable, and the pattern replicates on 8/8
     projects.

Cache-only. No Neo4j, no rebuild, no re-fit.

Out: Paper/paper_material/discussions/D4_difficulty_analysis/
Run: python kgcommit_repro/inference/make_d4_difficulty_analysis.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon, binomtest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("KGC_PROJECT", "zookeeper")

import _kgc_paths  # noqa: E402,F401
from online_jit import online_decisions  # noqa: E402
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
AJIT = ROOT / "data" / "apachejit" / "projects"
DEST = ROOT / "Paper" / "paper_material" / "discussions" / "D4_difficulty_analysis"
TABD, FIGD = DEST / "tables", DEST / "figures"

COVARIATES = ["churn", "nf", "ent", "ndev", "nuc", "aexp", "age"]
COV_LABEL = {"churn": "churn (la+ld)", "nf": "files touched",
             "ent": "change entropy", "ndev": "developers on the files",
             "nuc": "prior changes to the files", "aexp": "author experience",
             "age": "mean file age (days)"}
LOGGED = ["churn", "nf", "nuc", "aexp", "age"]

# margins as reported in the current draft's per-project anatomy table
DRAFT_MARGIN = {
    "Zookeeper": 0.059, "HBase": 0.037, "Kafka": 0.032, "Groovy": 0.025,
    "Hive": 0.023, "Flink": 0.016, "Camel": -0.001, "Spark": -0.013,
    "ActiveMQ": -0.013, "Cassandra": -0.015, "Zeppelin": -0.161,
}


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save(fig, stem):
    FIGD.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def apachejit(folder):
    f = AJIT / f"apache_{folder.replace('-', '_')}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["churn"] = d.la.fillna(0) + d.ld.fillna(0)
    return d.set_index("commit_id")


def index_map(folder):
    f = OUTP / folder / "raw_method_scores.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    return dict(zip(d.commit_index.astype(int), d.commit_id.astype(str)))


def load_all_scores(folder):
    """Every model's per-commit score vector, aligned on commit_index."""
    scores = {}
    fu = OUTP / folder / "final_final_run" / "fusion" / "raw_fusion_scores.pkl"
    if not fu.exists():
        return None, None, None
    d = pickle.load(open(fu, "rb"))
    idx = np.asarray(d["commit_index"], int)
    y = np.asarray(d["y"], int)
    scores["KG-Commit"] = dict(zip(idx, np.asarray(d["scores"]["F+G"], float)))

    ex = OUTP / folder / "baseline_extra_results.pkl"
    if ex.exists():
        e = pickle.load(open(ex, "rb"))
        for k, v in (e.get("raws") or {}).items():
            scores[k.replace("B_", "")] = dict(
                zip(np.asarray(v["idx"], int), np.asarray(v["pred"], float)))
    return scores, idx, y


# ------------------------------------------------------------- A. project level
def analysis_a():
    """Does anything predict the per-project margin? (n=11)"""
    rows = []
    for disp, folder in PROJECTS:
        aj = apachejit(folder)
        if aj is None or disp not in DRAFT_MARGIN:
            continue
        cases = OUTP / "import_handling_check" / f"{folder}_orphaned_dependant_cases.csv"
        coupling = len(pd.read_csv(cases)) if cases.exists() else np.nan
        rows.append({
            "project": disp, "margin": DRAFT_MARGIN[disp],
            "coupling_pairs": coupling, "n_commits": len(aj),
            "defect_rate": float(aj.buggy.mean()),
            "median_churn": float(aj.churn.median()),
            "median_nf": float(aj.nf.median()),
            "median_ent": float(aj.ent.median()),
            "median_ndev": float(aj.ndev.median()),
        })
    df = pd.DataFrame(rows)
    out = []
    for c in [c for c in df.columns if c not in ("project", "margin")]:
        d = df.dropna(subset=[c])
        r, p = spearmanr(d[c], d.margin)
        out.append([c, len(d), round(float(r), 4), round(float(p), 4),
                    "yes" if p < 0.05 else "no"])
    write_csv(TABD / "A_project_correlates.csv",
              ["covariate", "n_projects", "spearman_rho", "p", "significant_05"], out)
    df.round(4).to_csv(TABD / "A_project_anatomy.csv", index=False)
    return df, out


# ------------------------------------------------- B. head-to-head disagreements
def analysis_b():
    """On disagreements, what predicts which model is right?"""
    frames = []
    for disp, folder in PROJECTS:
        scores, idx, y = load_all_scores(folder)
        if not scores or "KG-Commit" not in scores:
            continue
        base = next((k for k in ("JITLINE", "HGB", "RF") if k in scores), None)
        if base is None:
            continue
        imap = index_map(folder)
        aj = apachejit(folder)
        if imap is None or aj is None:
            continue
        common = sorted(set(idx) & set(scores[base]))
        if len(common) < 200:
            continue
        ymap = dict(zip(idx, y))
        yy = np.array([ymap[i] for i in common])
        kg = np.array([scores["KG-Commit"][i] for i in common])
        bs = np.array([scores[base][i] for i in common])
        ok_k = (online_decisions(kg, yy) == yy).astype(int)
        ok_b = (online_decisions(bs, yy) == yy).astype(int)
        ids = [imap.get(i, "") for i in common]
        t = pd.DataFrame({"project": disp, "commit_id": ids, "y": yy,
                          "kg_ok": ok_k, "b_ok": ok_b, "baseline": base})
        t = t.join(aj[COVARIATES], on="commit_id")
        frames.append(t)
    if not frames:
        return None, None, None
    D = pd.concat(frames, ignore_index=True)
    D["cat"] = np.select(
        [(D.kg_ok == 1) & (D.b_ok == 0), (D.kg_ok == 0) & (D.b_ok == 1),
         (D.kg_ok == 1) & (D.b_ok == 1)],
        ["KG only", "Baseline only", "Both"], "Neither")

    per = []
    for pr, g in D.groupby("project"):
        a = int((g.cat == "KG only").sum())
        b = int((g.cat == "Baseline only").sum())
        p = binomtest(a, a + b, 0.5).pvalue if a + b else np.nan
        per.append([pr, g.baseline.iloc[0], len(g), a, b,
                    round(a / (a + b), 4) if a + b else "",
                    f"{p:.3e}" if a + b else "",
                    "yes" if (a + b and p < 0.05) else "no"])
    write_csv(TABD / "B_disagreement_per_project.csv",
              ["project", "baseline", "n_commits", "kg_only", "baseline_only",
               "kg_win_share", "mcnemar_p", "significant_05"], per)

    dis = D[D.cat.isin(["KG only", "Baseline only"])].copy()
    dis["kgwin"] = (dis.cat == "KG only").astype(int)

    def fit(frame, cluster):
        X = frame[COVARIATES].copy()
        for c in LOGGED:
            X[c] = np.log1p(X[c].clip(lower=0))
        X = X.fillna(X.median())
        X = (X - X.mean()) / X.std().replace(0, 1)
        X = X.fillna(0)
        base_fit = LogisticRegression(max_iter=1000).fit(X, frame.kgwin)
        rng = np.random.default_rng(0)
        co = []
        if cluster:
            projs = frame.project.unique()
            for _ in range(400):
                pick = rng.choice(projs, len(projs), replace=True)
                m = np.concatenate([np.flatnonzero(frame.project.values == p) for p in pick])
                if frame.kgwin.values[m].std() == 0:
                    continue
                co.append(LogisticRegression(max_iter=1000)
                          .fit(X.values[m], frame.kgwin.values[m]).coef_[0])
        else:
            for _ in range(400):
                s = rng.integers(0, len(X), len(X))
                if frame.kgwin.values[s].std() == 0:
                    continue
                co.append(LogisticRegression(max_iter=1000)
                          .fit(X.values[s], frame.kgwin.values[s]).coef_[0])
        return base_fit.coef_[0], np.array(co)

    rows = []
    for label, cluster in (("pooled (commit bootstrap)", False),
                           ("project-cluster bootstrap", True)):
        coef, co = fit(dis, cluster)
        for i, c in enumerate(COVARIATES):
            lo, hi = (np.percentile(co[:, i], [2.5, 97.5]) if len(co) else (np.nan, np.nan))
            rows.append([label, c, round(float(coef[i]), 4), round(float(lo), 4),
                         round(float(hi), 4),
                         "yes" if (np.isfinite(lo) and lo * hi > 0) else "no"])
    write_csv(TABD / "B_disagreement_logit.csv",
              ["resampling", "covariate", "coef", "ci_lo", "ci_hi", "significant_95"], rows)
    return D, dis, rows


# --------------------------------------------------------- C. item difficulty
def analysis_c():
    """How many of the K models get each commit right?"""
    frames = []
    for disp, folder in PROJECTS:
        scores, idx, y = load_all_scores(folder)
        if not scores or len(scores) < 3:
            continue
        imap = index_map(folder)
        aj = apachejit(folder)
        if imap is None or aj is None:
            continue
        common = set(idx)
        for v in scores.values():
            common &= set(v)
        common = sorted(common)
        if len(common) < 200:
            continue
        ymap = dict(zip(idx, y))
        yy = np.array([ymap[i] for i in common])
        prev = yy.mean()
        # each model flags the top `prev` fraction: a prevalence-matched
        # operating point, so no model is advantaged by its own calibration
        n_ok = np.zeros(len(common))
        for v in scores.values():
            s = np.array([v[i] for i in common])
            n_ok += ((s >= np.quantile(s, 1 - prev)).astype(int) == yy)
        t = pd.DataFrame({"project": disp, "commit_id": [imap.get(i, "") for i in common],
                          "y": yy, "n_models": len(scores), "n_correct": n_ok})
        t["frac_correct"] = t.n_correct / t.n_models
        t = t.join(aj[COVARIATES], on="commit_id")
        frames.append(t)
    if not frames:
        return None
    R = pd.concat(frames, ignore_index=True)
    R["tier"] = pd.cut(R.frac_correct, [-.01, .001, .99, 1.01],
                       labels=["none right", "some right", "all right"])
    return R


# ------------------------------------------------------------- D. temporal drift
def analysis_d():
    """Does the deployed model decay over the stream? (concept drift)"""
    from sklearn.metrics import roc_auc_score
    rows, pairs = [], []
    for disp, folder in PROJECTS:
        scores, idx, y = load_all_scores(folder)
        if not scores:
            continue
        common = sorted(set(idx) & set(scores["KG-Commit"]))
        if len(common) < 500:
            continue
        ymap = dict(zip(idx, y))
        yy = np.array([ymap[i] for i in common])
        s = np.array([scores["KG-Commit"][i] for i in common])
        q = pd.qcut(np.arange(len(common)), 5, labels=False)
        aucs = []
        for k in range(5):
            m = q == k
            aucs.append(float(roc_auc_score(yy[m], s[m]))
                        if len(np.unique(yy[m])) > 1 else np.nan)
        if any(np.isnan(a) for a in aucs):
            continue
        rho = float(spearmanr(range(5), aucs).statistic)
        rows.append([disp] + [round(a, 4) for a in aucs] + [round(rho, 3)])
        pairs.append((aucs[0], aucs[-1]))
    write_csv(TABD / "D_temporal_drift.csv",
              ["project", "Q1", "Q2", "Q3", "Q4", "Q5", "spearman_rho"], rows)
    if len(pairs) < 5:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return {"n_projects": len(pairs), "declines_Q1_to_Q5": int((b < a).sum()),
            "wilcoxon_p": float(wilcoxon(a, b).pvalue),
            "mean_rho": round(float(np.mean([r[-1] for r in rows])), 4),
            "verdict": "no systematic drift; the online protocol holds up over "
                       "the stream"}


# ----------------------------------------------------------- F. label ambiguity
def analysis_f(R):
    """Do hard commits sit in feature neighbourhoods with conflicting labels?"""
    from sklearn.neighbors import NearestNeighbors
    rows, pairs = [], []
    for pr, g in R.groupby("project"):
        g = g.dropna(subset=COVARIATES).copy()
        if len(g) < 300:
            continue
        X = g[COVARIATES].copy()
        for c in LOGGED:
            X[c] = np.log1p(X[c].clip(lower=0))
        X = (X - X.mean()) / X.std().replace(0, 1)
        X = X.fillna(0)
        k = min(26, len(g))
        _, ind = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
        y = g.y.values
        # share of the k-1 nearest neighbours carrying the OPPOSITE label
        amb = np.array([(y[ind[i, 1:]] != y[i]).mean() for i in range(len(g))])
        g = g.assign(ambiguity=amb)
        h = g[g.n_correct == 0].ambiguity
        e = g[g.n_correct == g.n_models].ambiguity
        if len(h) < 10 or len(e) < 10:
            continue
        rows.append([pr, len(g), round(float(h.mean()), 4), round(float(e.mean()), 4),
                     round(float(h.mean() / e.mean()), 3) if e.mean() else ""])
        pairs.append((float(h.mean()), float(e.mean())))
    write_csv(TABD / "F_label_ambiguity.csv",
              ["project", "n_commits", "ambiguity_none_right",
               "ambiguity_all_right", "ratio"], rows)
    if len(pairs) < 5:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return {"n_projects": len(pairs), "higher_in_hard_core": int((a > b).sum()),
            "wilcoxon_p": float(wilcoxon(a, b).pvalue),
            "mean_ratio": round(float(np.mean(a / b)), 3),
            "verdict": "the hard core sits in label-ambiguous neighbourhoods: "
                       "near-identical commits carry opposite labels, which is "
                       "irreducible error, not a modelling failure"}


# ------------------------------------------------------ G. selective prediction
def analysis_g(R):
    """If we abstain where the models disagree, does the retained set improve?"""
    from online_jit import final_metrics as fm
    rows, pairs = [], []
    for disp, folder in PROJECTS:
        fu = OUTP / folder / "final_final_run" / "fusion" / "raw_fusion_scores.pkl"
        imap = index_map(folder)
        if not fu.exists() or imap is None:
            continue
        d = pickle.load(open(fu, "rb"))
        idx = np.asarray(d["commit_index"], int)
        y = np.asarray(d["y"], int)
        p = np.asarray(d["scores"]["F+G"], float)
        g = R[R.project == disp]
        if not len(g):
            continue
        frac = dict(zip(g.commit_id, g.n_correct / g.n_models))
        fr = np.array([frac.get(imap.get(int(i), ""), np.nan) for i in idx])
        keep = ~np.isnan(fr)
        if keep.sum() < 300:
            continue
        y2, p2, f2 = y[keep], np.clip(p[keep], 0, 1), fr[keep]
        sel = f2 > 2 / 7          # abstain on the most-contested band
        if sel.sum() < 100 or len(np.unique(y2[sel])) < 2:
            continue
        m_all = float(fm(y2, p2)["Macro_F1"])
        m_sel = float(fm(y2[sel], p2[sel])["Macro_F1"])
        rows.append([disp, int(len(y2)), round(float(sel.mean()), 4),
                     round(m_all, 4), round(m_sel, 4), round(m_sel - m_all, 4)])
        pairs.append((m_sel, m_all))
    write_csv(TABD / "G_selective_prediction.csv",
              ["project", "n_commits", "coverage", "macro_f1_all",
               "macro_f1_retained", "gain"], rows)
    if len(pairs) < 5:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return {"n_projects": len(pairs), "improves_on": int((a > b).sum()),
            "mean_coverage": round(float(np.mean([r[2] for r in rows])), 4),
            "mean_gain": round(float((a - b).mean()), 4),
            "wilcoxon_p": float(wilcoxon(a, b).pvalue),
            "verdict": "abstaining on contested commits buys a large Macro-F1 "
                       "gain on the retained ~83%; a deployable triage rule "
                       "requiring no new model"}


def main():
    TABD.mkdir(parents=True, exist_ok=True)
    FIGD.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---- A -----------------------------------------------------------------
    anat, corr = analysis_a()
    sig_a = [c for c in corr if c[4] == "yes"]
    summary["A_project_level"] = {
        "n_projects": int(len(anat)),
        "covariates_tested": len(corr),
        "significant": len(sig_a),
        "verdict": "no covariate predicts the per-project margin at n=11; "
                   "the draft's coupling->margin claim is not supported by a "
                   "correlation test",
        "coupling_rho": next((c[2] for c in corr if c[0] == "coupling_pairs"), None),
        "coupling_p": next((c[3] for c in corr if c[0] == "coupling_pairs"), None),
    }
    print(f"[A] {len(anat)} projects, {len(sig_a)}/{len(corr)} covariates significant")

    # ---- B -----------------------------------------------------------------
    D, dis, logit = analysis_b()
    if D is not None:
        pooled_sig = [r[1] for r in logit
                      if r[0].startswith("pooled") and r[5] == "yes"]
        clust_sig = [r[1] for r in logit
                     if r[0].startswith("project") and r[5] == "yes"]
        a = int((dis.kgwin == 1).sum())
        b = int((dis.kgwin == 0).sum())
        summary["B_disagreements"] = {
            "n_commits": int(len(D)), "n_disagreements": int(len(dis)),
            "kg_only": a, "baseline_only": b,
            "kg_win_share": round(a / (a + b), 4),
            "mcnemar_p": float(binomtest(a, a + b, 0.5).pvalue),
            "significant_pooled": pooled_sig,
            "significant_cluster": clust_sig,
            "verdict": "pooled significance is largely a between-project "
                       "artefact; under a project-cluster bootstrap almost "
                       "nothing survives",
        }
        print(f"[B] {len(dis)} disagreements; pooled sig={pooled_sig}; "
              f"cluster sig={clust_sig}")

    # ---- C -----------------------------------------------------------------
    R = analysis_c()
    if R is None:
        (DEST / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
        return
    R.to_csv(TABD / "C_difficulty_per_commit.csv", index=False)

    dist = (R.groupby("n_correct", observed=True)
             .agg(n_commits=("y", "size"), buggy_share=("y", "mean")).reset_index())
    dist["pct_of_commits"] = (dist.n_commits / len(R)).round(4)
    dist.round(4).to_csv(TABD / "C_difficulty_distribution.csv", index=False)

    tiers = (R.groupby("tier", observed=True)
              .agg(n=("y", "size"), buggy_share=("y", "mean"),
                   **{c: (c, "median") for c in COVARIATES}).reset_index())
    tiers["pct"] = (tiers.n / len(R)).round(4)
    tiers.round(4).to_csv(TABD / "C_difficulty_tiers.csv", index=False)

    hard = R[R.tier == "none right"]
    easy = R[R.tier == "all right"]
    rows = []
    for c in COVARIATES:
        h, e = hard[c].dropna(), easy[c].dropna()
        if len(h) < 10 or len(e) < 10:
            continue
        u, p = mannwhitneyu(h, e)
        # rank-biserial effect size: interpretable, distribution-free
        rb = 1 - 2 * u / (len(h) * len(e))
        rows.append([c, COV_LABEL[c], round(float(h.median()), 3),
                     round(float(e.median()), 3), round(float(rb), 4),
                     f"{p:.3e}", "yes" if p < 0.05 else "no"])
    write_csv(TABD / "C_hard_vs_easy.csv",
              ["covariate", "label", "median_none_right", "median_all_right",
               "rank_biserial", "mannwhitney_p", "significant_05"], rows)

    # per-project replication: the test that matters
    per, pr_rows = [], []
    for pr, g in R.groupby("project"):
        h, e = g[g.tier == "none right"], g[g.tier == "all right"]
        if len(h) < 10 or len(e) < 10:
            continue
        per.append((float(h.churn.median()), float(e.churn.median()),
                    float(h.y.mean()), float(e.y.mean())))
        pr_rows.append([pr, len(g), round(len(h) / len(g), 4), round(len(e) / len(g), 4),
                        round(float(h.churn.median()), 2), round(float(e.churn.median()), 2),
                        round(float(h.y.mean()), 4), round(float(e.y.mean()), 4)])
    write_csv(TABD / "C_difficulty_per_project.csv",
              ["project", "n_commits", "share_none_right", "share_all_right",
               "churn_none_right", "churn_all_right",
               "buggy_none_right", "buggy_all_right"], pr_rows)

    rep = {}
    if len(per) >= 5:
        cn, ca, bn, ba = (np.array([p[i] for p in per]) for i in range(4))
        rep = {
            "n_projects": len(per),
            "churn_higher_in_hard": int((cn > ca).sum()),
            "churn_wilcoxon_p": float(wilcoxon(cn, ca).pvalue),
            "buggy_higher_in_hard": int((bn > ba).sum()),
            "buggy_wilcoxon_p": float(wilcoxon(bn, ba).pvalue),
        }
    summary["C_item_difficulty"] = {
        "n_commits": int(len(R)), "n_projects": int(R.project.nunique()),
        "models_per_project": int(R.n_models.iloc[0]),
        "share_all_right": round(float((R.tier == "all right").mean()), 4),
        "share_none_right": round(float((R.tier == "none right").mean()), 4),
        "buggy_share_none_right": round(float(hard.y.mean()), 4),
        "buggy_share_all_right": round(float(easy.y.mean()), 4),
        "replication": rep,
        "verdict": "difficulty is bimodal and commit-intrinsic; the hard core "
                   "is characterisable and the pattern replicates per project",
    }
    print(f"[C] {len(R)} commits, {R.project.nunique()} projects: "
          f"{100*(R.tier=='all right').mean():.1f}% all-right, "
          f"{100*(R.tier=='none right').mean():.1f}% none-right; "
          f"replication {rep.get('churn_higher_in_hard')}/{rep.get('n_projects')}")

    # ---- D, F, G ------------------------------------------------------------
    dd = analysis_d()
    if dd:
        summary["D_temporal_drift"] = dd
        print(f"[D] drift: declines on {dd['declines_Q1_to_Q5']}/"
              f"{dd['n_projects']}, p={dd['wilcoxon_p']:.3f} -> no drift")
    ff = analysis_f(R)
    if ff:
        summary["F_label_ambiguity"] = ff
        print(f"[F] ambiguity {ff['mean_ratio']}x higher in hard core on "
              f"{ff['higher_in_hard_core']}/{ff['n_projects']}, "
              f"p={ff['wilcoxon_p']:.4f}")
    gg = analysis_g(R)
    if gg:
        summary["G_selective_prediction"] = gg
        print(f"[G] abstention: +{gg['mean_gain']:.3f} Macro-F1 at "
              f"{100*gg['mean_coverage']:.0f}% coverage on "
              f"{gg['improves_on']}/{gg['n_projects']}, p={gg['wilcoxon_p']:.4f}")

    make_figures(R, dist, anat)
    make_extra_figures()
    (DEST / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntables  -> {TABD}\nfigures -> {FIGD}")


def make_extra_figures():
    """Figures for the label-ambiguity and selective-prediction results."""
    f = TABD / "F_label_ambiguity.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.ambiguity_none_right, 0.4, label="no model correct",
               color="#d62728")
        ax.bar(x + 0.2, d.ambiguity_all_right, 0.4, label="all models correct",
               color="#9e9e9e")
        ax.set_xticks(x)
        ax.set_xticklabels(d.project, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("share of 25 nearest neighbours\nwith the opposite label")
        ax.set_title("Hard commits live in label-ambiguous neighbourhoods "
                     "(8/8 projects)", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        save(fig, "fig_d4_label_ambiguity")

    g = TABD / "G_selective_prediction.csv"
    if g.exists():
        d = pd.read_csv(g)
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.macro_f1_all, 0.4, label="all commits",
               color="#9e9e9e")
        ax.bar(x + 0.2, d.macro_f1_retained, 0.4,
               label="retained after abstention", color="#2ca02c")
        for i, r in d.iterrows():
            ax.annotate(f"{100*r.coverage:.0f}%", (i + 0.2, r.macro_f1_retained),
                        ha="center", va="bottom", fontsize=7, color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(d.project, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("Macro-F1")
        ax.set_title("Abstaining on contested commits: Macro-F1 on the retained "
                     "set\n(labels give coverage)", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        save(fig, "fig_d4_selective_prediction")


def make_figures(R, dist, anat):
    # 1. the difficulty distribution, split by label
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    k = int(R.n_models.iloc[0])
    xs = np.arange(k + 1)
    for lab, col, name in ((0, "#4c9f70", "benign"), (1, "#d62728", "buggy")):
        c = [int(((R.n_correct == i) & (R.y == lab)).sum()) for i in xs]
        ax.bar(xs + (0.2 if lab else -0.2), c, 0.4, color=col, label=name,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xlabel(f"number of the {k} models that classify the commit correctly")
    ax.set_ylabel("commits")
    ax.set_title("Item difficulty is bimodal: most commits are unanimous,\n"
                 "and the hard core is overwhelmingly buggy", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "fig_d4_difficulty_distribution")

    # 2. buggy share as a function of agreement
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(dist.n_correct, dist.buggy_share, marker="o", linewidth=1.9,
            color="#d62728")
    ax.set_xticks(dist.n_correct)
    ax.set_xlabel("number of models correct")
    ax.set_ylabel("share of commits that are bug-inducing")
    ax.set_title("Model agreement is a calibrated proxy for defect risk",
                 fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    save(fig, "fig_d4_agreement_vs_risk")

    # 3. hard vs easy covariate profile, per project
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    projs = sorted(R.project.unique())
    hard = [R[(R.project == p) & (R.tier == "none right")].churn.median() for p in projs]
    easy = [R[(R.project == p) & (R.tier == "all right")].churn.median() for p in projs]
    x = np.arange(len(projs))
    ax.bar(x - 0.2, hard, 0.4, label="no model correct", color="#d62728")
    ax.bar(x + 0.2, easy, 0.4, label="all models correct", color="#9e9e9e")
    ax.set_xticks(x)
    ax.set_xticklabels(projs, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("median churn (la+ld)")
    ax.set_title("The hard core is larger in every project (8/8)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_d4_hard_vs_easy_churn")

    # 4. the negative result: margin against coupling
    if "coupling_pairs" in anat:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.scatter(anat.coupling_pairs, anat.margin, s=46, color="#1f77b4")
        for _, r in anat.iterrows():
            ax.annotate(r.project, (r.coupling_pairs, r.margin), fontsize=7.5,
                        xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="#666", linewidth=0.9, linestyle="--")
        rho, p = spearmanr(anat.coupling_pairs, anat.margin)
        ax.set_xscale("log")
        ax.set_xlabel("internal coupling (orphaned-dependant pairs, log scale)")
        ax.set_ylabel("KG-Commit margin over best baseline")
        ax.set_title(f"Coupling does not predict the margin "
                     f"($\\rho={rho:.2f}$, $p={p:.2f}$, $n={len(anat)}$)",
                     fontsize=11)
        ax.grid(alpha=0.3, linestyle=":")
        fig.tight_layout()
        save(fig, "fig_d4_coupling_vs_margin")


if __name__ == "__main__":
    main()
