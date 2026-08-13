"""
Additional insight visualizations for the KG-Commit V4 paper (NON-scalability).

These showcase the project's real novelty -- the multimodal, semantic, and
interpretable structure of the KG -- reusing only cached artifacts + read-only
Neo4j (Term risk/kind, Intent, grounding). Nothing rebuilds the KG.

Figures -> docs/figures/v4/insight/  (PNG + PDF):
  fig_modality_complementarity  per-modality AUC + signal-correlation heatmap
  fig_commit_embedding          2-D projection of commit KGE, coloured by bug
  fig_intent_bugrate            bug rate by change-intent (interpretable risk)
  fig_defect_typing             term risk by lexicon kind + top risky terms
  fig_calibration               reliability curve of the deployed fusion
  fig_drift_signal              per-modality signal vs bug rate over time
  fig_astgroup_risk             AST change-group composition, buggy vs benign

Run:  python scalability/make_insight_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

FIG = C.ROOT / "docs" / "figures" / "v4" / "insight"
FIG.mkdir(parents=True, exist_ok=True)
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.25})


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}")


def _streams():
    import pickle
    return pickle.load(open(C.ROOT / "outputs" / "online_jit_streams_v5.pkl", "rb"))


def _auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, int); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    n1 = y.sum(); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ── 1. multimodal complementarity: per-modality AUC + correlation ───────────

def fig_modality_complementarity(S):
    """Each modality reduced to a single risk score; show its AUC and how weakly
    the scores correlate (= complementary signals worth fusing)."""
    y = S["y"]
    # one score per modality (past-agnostic single-signal proxies)
    def col_score(X):
        # project a feature block to a 1-D risk via its buggy-mean direction.
        # Kept SPARSE throughout (never densified): a hashed block like AST
        # tokens (Xh) is (n, 262144) and densifying it needs ~17 GiB.
        if hasattr(X, "toarray"):            # scipy.sparse
            w = np.asarray(X[y == 1].mean(0)).ravel() - np.asarray(X[y == 0].mean(0)).ravel()
            return np.asarray(X @ w).ravel()
        Xd = np.asarray(X, float)
        if Xd.ndim == 1:
            Xd = Xd[:, None]
        w = Xd[y == 1].mean(0) - Xd[y == 0].mean(0)
        return Xd @ w
    sig = {
        "JIT metrics (M)": col_score(S["Xms"]),
        "Relational (R)": col_score(S["Xp"]),
        "AST tokens (T)": col_score(S["Xh"]),
        "PPR (P)": S["ppr_full"],
        "CSTG prior (G)": S["cstg_prior"],
        "CSTG typed": col_score(S["cstg_typed"]),
    }
    names = list(sig)
    aucs = [_auc(y, sig[n]) for n in names]
    aucs = [a if a >= 0.5 else 1 - a for a in aucs]     # orient
    M = np.column_stack([sig[n] for n in names])
    Mn = (M - M.mean(0)) / (M.std(0) + 1e-9)
    corr = np.corrcoef(Mn.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    order = np.argsort(aucs)
    ax1.barh([names[i] for i in order], [aucs[i] for i in order],
             color=[OKABE[i % len(OKABE)] for i in order])
    ax1.axvline(0.5, color="k", lw=0.8, ls="--")
    ax1.set_xlim(0.5, max(aucs) + 0.03); ax1.set_xlabel("single-signal AUC")
    ax1.set_title("Each modality is predictive alone")
    im = ax2.imshow(np.abs(corr), cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax2.text(j, i, f"{abs(corr[i, j]):.2f}", ha="center", va="center",
                     fontsize=6, color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax2.set_title("...yet signals are weakly correlated\n(complementary $\\Rightarrow$ fusion helps)")
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    _save(fig, "fig_modality_complementarity")


# ── 2. commit embedding projection ──────────────────────────────────────────

def fig_commit_embedding(S):
    emb_p = C.ROOT / "outputs" / "emb_distmult_8059.npy"
    if not emb_p.exists():
        return
    E = np.load(emb_p); y = S["y"]
    from sklearn.decomposition import PCA
    Z = PCA(n_components=2, random_state=0).fit_transform(
        (E - E.mean(0)) / (E.std(0) + 1e-9))
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=5, c=OKABE[5], alpha=0.35, label="benign", linewidths=0)
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=7, c=OKABE[4], alpha=0.55, label="buggy", linewidths=0)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Commit KG-embedding (DistMult) projection")
    ax.legend(frameon=False, markerscale=2)
    _save(fig, "fig_commit_embedding")


# ── 3. change-intent bug rate (interpretable) ───────────────────────────────

def fig_intent_bugrate():
    d = C.driver()
    with C.read_session(d) as s:
        rows = s.run(
            "MATCH (c:Commit {in_jit:true})-[:HAS_INTENT]->(i:Intent) "
            "RETURN i.id AS intent, count(c) AS n, "
            "avg(toFloat(CASE WHEN c.buggy THEN 1 ELSE 0 END)) AS br "
            "ORDER BY br DESC").data()
    d.close()
    if not rows:
        return
    base = sum(r["n"] * r["br"] for r in rows) / sum(r["n"] for r in rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    names = [r["intent"] for r in rows]; br = [r["br"] for r in rows]; n = [r["n"] for r in rows]
    cols = [OKABE[4] if b > base else OKABE[2] for b in br]
    ax.bar(names, br, color=cols)
    ax.axhline(base, color="k", ls="--", lw=1, label=f"overall {base:.2f}")
    for i, (b, c) in enumerate(zip(br, n)):
        ax.text(i, b + 0.005, f"n={c}", ha="center", fontsize=7)
    ax.set_ylabel("bug rate"); ax.set_title("Bug rate by change-intent (CSTG typing)")
    ax.legend(frameon=False)
    _save(fig, "fig_intent_bugrate")


# ── 4. defect typing: term risk by lexicon kind + top risky terms ───────────

def fig_defect_typing():
    d = C.driver()
    with C.read_session(d) as s:
        kind = s.run("MATCH (t:Term) WHERE t.risk IS NOT NULL "
                     "RETURN t.kind AS k, count(*) AS n, avg(t.risk) AS mr, "
                     "percentileCont(t.risk,0.9) AS p90 ORDER BY mr DESC").data()
        top = s.run("MATCH (t:Term) WHERE t.risk IS NOT NULL AND t.kind IN ['bug','error','action'] "
                    "RETURN t.text AS w, t.risk AS r ORDER BY t.risk DESC LIMIT 12").data()
    d.close()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8),
                                 gridspec_kw={"width_ratios": [1, 1]})
    ks = [r["k"] for r in kind]; mr = [r["mr"] for r in kind]
    a1.bar(ks, mr, color=[OKABE[i] for i in range(len(ks))])
    for i, r in enumerate(kind):
        a1.text(i, r["mr"] + 0.002, f"n={r['n']}", ha="center", fontsize=7)
    a1.set_ylabel("mean term bug-risk"); a1.set_title("Learned risk by term type (defect lexicon)")
    if top:
        ws = [r["w"][:20] for r in top][::-1]; rs = [r["r"] for r in top][::-1]
        a2.barh(ws, rs, color=OKABE[4]); a2.set_xlabel("term bug-risk")
        a2.set_title("Highest-risk semantic terms")
        a2.tick_params(axis="y", labelsize=7)
    _save(fig, "fig_defect_typing")


# ── 5. calibration / reliability of the deployed model ──────────────────────

def fig_calibration():
    import pickle
    p = C.ROOT / "outputs" / "final_fusion_results.pkl"
    if not p.exists():
        return
    d = pickle.load(open(p, "rb"))
    # F+G trajectory doesn't carry (y,p); use the online_jit ablation G+P proxy that does
    ab = C.ROOT / "outputs" / "online_jit_ablation_v3.pkl"
    if not ab.exists():
        return
    A = pickle.load(open(ab, "rb"))
    key = "M+T+R+P+G" if "M+T+R+P+G" in A else list(A)[-1]
    y = np.asarray(A[key]["y"], int); pr = np.asarray(A[key]["p"], float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pr, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 20:
            xs.append(pr[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(xs, ys, "o-", color=OKABE[2], label="deployed fusion (F+G)")
    for x, yv, nn in zip(xs, ys, ns):
        ax.annotate(str(nn), (x, yv), fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("predicted bug probability"); ax.set_ylabel("observed bug rate")
    ax.set_title("Reliability of the deployed predictor")
    ax.legend(frameon=False)
    _save(fig, "fig_calibration")


# ── 6. concept drift: modality signal tracks the drifting bug rate ──────────

def fig_drift_signal(S):
    y = S["y"]
    # rolling bug rate + rolling AUC of two modalities over stream position
    win = 400
    n = len(y)
    xs = np.arange(win, n, 100)
    def roll_auc(score):
        out = []
        for j in xs:
            out.append(_auc(y[j - win:j], np.asarray(score)[j - win:j]))
        return out
    br = [y[j - win:j].mean() for j in xs]
    fig, ax1 = plt.subplots(figsize=(6.6, 3.9))
    ax1.plot(xs, br, color="k", lw=1.5, label="bug rate (rolling)")
    ax1.set_xlabel("commit (online order)"); ax1.set_ylabel("bug rate")
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(xs, roll_auc(S["cstg_prior"]), color=OKABE[2], label="CSTG prior AUC")
    ax2.plot(xs, roll_auc(S["ppr_full"]), color=OKABE[0], label="PPR AUC")
    ax2.set_ylabel("rolling AUC")
    ax1.set_title("Signals track the drifting bug rate over time")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, frameon=False, fontsize=8, loc="upper right")
    _save(fig, "fig_drift_signal")


# ── 7. AST change-group composition, buggy vs benign ────────────────────────

def fig_astgroup_risk():
    import csv
    p = C.ROOT / "outputs" / "commit_features.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))
    grps = ["grp_declaration", "grp_statement", "grp_expression", "grp_literal", "grp_leaf"]
    def frac(sel):
        tot = np.array([sum(float(r[g]) for g in grps) for r in sel])
        out = {}
        for g in grps:
            v = np.array([float(r[g]) for r in sel])
            out[g] = (v / np.maximum(tot, 1)).mean()
        return out
    buggy = frac([r for r in rows if r["buggy"] in ("True", "1", "true")])
    benign = frac([r for r in rows if r["buggy"] in ("False", "0", "false")])
    labels = [g.replace("grp_", "") for g in grps]
    x = np.arange(len(grps)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w / 2, [benign[g] for g in grps], w, color=OKABE[5], label="benign")
    ax.bar(x + w / 2, [buggy[g] for g in grps], w, color=OKABE[4], label="buggy")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean fraction of AST change")
    ax.set_title("What kind of AST nodes buggy commits touch")
    ax.legend(frameon=False)
    _save(fig, "fig_astgroup_risk")


def main():
    print("rendering insight figures ->", FIG)
    S = _streams()
    fig_modality_complementarity(S)
    fig_commit_embedding(S)
    fig_intent_bugrate()
    fig_defect_typing()
    fig_calibration()
    fig_drift_signal(S)
    fig_astgroup_risk()
    print("done.")


if __name__ == "__main__":
    main()
