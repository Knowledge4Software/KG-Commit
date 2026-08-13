"""
Render the scalability / complexity figures (PNG + PDF) into
docs/figures/v4/scalability/ from the JSON produced by the E1-E5 collectors.

Figures (each PNG+PDF, Okabe-Ito palette, theme-neutral):
  fig_growth_cumulative      cumulative delta-edges vs commit index, per layer
  fig_growth_year            delta volume + bug rate per calendar year (drift)
  fig_delta_size_box         per-commit change-size, buggy vs benign, per layer
  fig_degree_loglog          commit change-token degree distribution (log-log)
  fig_build_cost             sampled parse+diff time vs delta size + linear fit
  fig_predict_latency        per-commit predict latency per method x graph
  fig_pareto_cost_accuracy   (E6) accuracy vs per-commit inference cost frontier

Run:  python scalability/make_scalability_figures.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

C.FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colourblind-safe palette (matches the rest of the v4 figures)
OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                     "axes.splines.top" if False else "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.25})

LAYER_ORDER = ["ast", "dfg", "cfg", "pdg", "seq"]
LAYER_LABEL = {"ast": "AST", "cfg": "CFG", "dfg": "DFG", "pdg": "PDG", "seq": "Token-seq"}


def _load(name):
    p = C.OUT / name
    return json.load(open(p)) if p.exists() else None


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png/.pdf")


def fig_growth_cumulative(arrays):
    if not arrays:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, lyr in enumerate(LAYER_ORDER):
        a = arrays["layers"].get(lyr)
        if not a:
            continue
        ax.plot(np.asarray(a["cum_edges"]) / 1e6, color=OI[i], label=LAYER_LABEL[lyr])
    ax.set_xlabel("commit (online arrival order)")
    ax.set_ylabel("cumulative delta-edges (millions)")
    ax.set_title("Online KG growth per structural layer")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_growth_cumulative")


def fig_growth_year(growth):
    if not growth or "ast" not in growth:
        return
    by = growth["ast"]["by_year"]
    yrs = sorted(int(k) for k in by)
    edges = [by[str(y)]["delta_edges"] / 1e6 for y in yrs]
    rate = [by[str(y)]["bug_rate"] for y in yrs]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.bar(yrs, edges, color=OI[0], alpha=0.75, label="AST delta-edges")
    ax1.set_ylabel("AST delta-edges (millions)", color=OI[0])
    ax1.set_xlabel("year")
    ax2 = ax1.twinx()
    ax2.plot(yrs, rate, color=OI[4], marker="o", label="bug rate")
    ax2.set_ylabel("bug rate", color=OI[4]); ax2.grid(False)
    ax1.set_title("Change volume and concept drift over time")
    _save(fig, "fig_growth_year")


def fig_delta_size_box(growth):
    if not growth:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labels, data, colors = [], [], []
    for i, lyr in enumerate(LAYER_ORDER):
        if lyr not in growth:
            continue
        b = growth[lyr]["delta_per_buggy"]; g = growth[lyr]["delta_per_benign"]
        # draw as mean +/- p95 markers (we only stored the summary, not raw arrays)
        labels += [f"{LAYER_LABEL[lyr]}\nbenign", f"{LAYER_LABEL[lyr]}\nbuggy"]
        data += [g["mean"], b["mean"]]
        colors += [OI[5], OI[4]]
    x = np.arange(len(data))
    ax.bar(x, data, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean per-commit delta-edges")
    ax.set_title("Change size: buggy vs benign (all significant, MW p$\\ll$0.001)")
    _save(fig, "fig_delta_size_box")


def fig_degree_loglog(profile):
    if not profile:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, lyr in enumerate(LAYER_ORDER):
        r = profile.get(lyr)
        if not r:
            continue
        pl = r.get("delta_per_commit_powerlaw", {})
        d = r["delta_per_commit"]
        # illustrative: annotate the fitted alpha on a schematic tail line
        alpha = pl.get("alpha", float("nan"))
        xmin = pl.get("xmin", 1) or 1
        xs = np.logspace(np.log10(max(xmin, 1)), np.log10(max(d["max"], xmin + 1)), 50)
        ys = (xs / xmin) ** (1 - alpha)
        ax.loglog(xs, ys, color=OI[i],
                  label=f"{LAYER_LABEL[lyr]} ($\\alpha$={alpha:.2f})")
    ax.set_xlabel("per-commit change-token degree")
    ax.set_ylabel("P(X $\\geq$ x)  (fitted tail)")
    ax.set_title("Heavy-tailed change sizes (power-law tail fit)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_degree_loglog")


def fig_build_cost(build):
    if not build:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, lyr in enumerate(LAYER_ORDER):
        r = build.get(lyr)
        if not r or "fit_vs_delta_size" not in r:
            continue
        f = r["fit_vs_delta_size"]; d = r["delta_size"]
        xs = np.linspace(0, max(d["p95"], 1), 50)
        ax.plot(xs, f["intercept"] + f["slope"] * xs, color=OI[i],
                label=f"{LAYER_LABEL[lyr]} ($R^2$={f['r2']:.2f})")
    ax.set_xlabel("per-file change size (delta nodes)")
    ax.set_ylabel("parse+diff time (ms)")
    ax.set_title("Build cost is linear in change size (O(change))")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_build_cost")


def fig_predict_latency(lat):
    if not lat:
        return
    graphs = [g for g in C.FINAL_GRAPHS if g in lat]
    methods = C.FINAL_METHODS
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(graphs)); w = 0.16
    for j, m in enumerate(methods):
        vals = [lat[g]["predict_ms_per_commit"][m]["median"] for g in graphs]
        ax.bar(x + (j - 2) * w, vals, w, color=OI[j], label=m)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([C.GRAPH_NAME[g] for g in graphs],
                                         rotation=20, fontsize=8, ha="right")
    ax.set_ylabel("median predict ms / commit (log)")
    ax.set_title("Prediction latency by method and graph richness")
    ax.legend(frameon=False, fontsize=8, ncol=5)
    _save(fig, "fig_predict_latency")


def fig_pareto_cost_accuracy(lat, final_exp):
    """E6: accuracy (final graph Buggy-F1) vs per-commit inference cost."""
    if not lat or not final_exp or "final" not in lat or "final" not in final_exp:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for j, m in enumerate(C.FINAL_METHODS):
        cost = lat["final"]["predict_ms_per_commit"][m]["median"]
        # train cost folded in for embeddings
        if m in ("DW", "KGE"):
            tb = lat["final"]["train_ms_per_block"][m]["median"]
            cost = cost + tb / 50.0     # amortise per-block refit across BLOCK=50
        acc = final_exp["final"][m]["metrics"]["Buggy_F1"]
        ax.scatter(cost, acc, color=OI[j], s=70)
        ax.annotate(m, (cost, acc), textcoords="offset points", xytext=(6, 3),
                    fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("per-commit inference cost (ms, amortised)")
    ax.set_ylabel("Buggy-F1 on final KG")
    ax.set_title("Cost vs accuracy on the deployed graph (Pareto)")
    _save(fig, "fig_pareto_cost_accuracy")


def fig_cstg_ablation(abl):
    """CSTG internal-layer ablation: matched-RF comparison (fair, same-learner)
    next to the balanced-LR component-isolation ablation."""
    if not abl:
        return
    comp = abl["components"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)

    rf_keys = ["plain_tfidf_RF", "tw_idf_RF", "cstg_full_RF", "jit_cstg_full_RF"]
    rf_labels = ["Plain\nTF-IDF", "TW-IDF\n(graph-of-words)", "Full\nCSTG", "JIT+\nfull CSTG"]
    ax = axes[0]
    vals = [comp[k]["ROC"] for k in rf_keys]
    bars = ax.bar(range(len(vals)), vals, color=[OI[0], OI[5], OI[2], OI[3]])
    ax.axhline(abl["_meta"]["published_baseline_RF"]["ROC"], color=OI[4],
              linestyle="--", label="published TF-IDF-768D baseline")
    ax.set_xticks(range(len(rf_labels))); ax.set_xticklabels(rf_labels, fontsize=8)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.5, 0.9)
    ax.set_title("Matched learner (RF): fair comparison")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                   ha="center", va="bottom", fontsize=8)

    lr_keys = ["plain_tfidf_LR", "tw_idf_LR", "tw_idf_prior_LR", "cstg_full_LR"]
    lr_labels = ["(a) Plain\nTF-IDF", "(b) TW-IDF", "(d) TW-IDF\n+prior", "(e) Full\nCSTG"]
    ax = axes[1]
    vals = [comp[k]["ROC"] for k in lr_keys]
    bars = ax.bar(range(len(vals)), vals, color=[OI[0], OI[5], OI[1], OI[2]])
    ax.set_xticks(range(len(lr_labels))); ax.set_xticklabels(lr_labels, fontsize=8)
    ax.set_title("Component isolation (LR): TW-IDF alone $\\approx$ TF-IDF")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                   ha="center", va="bottom", fontsize=8)
    fig.suptitle("CSTG internal-layer ablation (controlled split)", y=1.03)
    _save(fig, "fig_cstg_ablation")


def fig_cstg_graph_ablation(ga):
    """Online graph-inference ablation over the separated CSTG hub-graph layers:
    each method's AUC as the CSTG graph is enriched Core -> +MENTIONS -> +TW-IDF
    -> +COOCCURS -> Full, with the G feature-classifier as a reference line."""
    if not ga:
        return
    variants = ga["_meta"]["variants"]
    methods = ga["_meta"]["methods"]
    vlabels = {"core": "Core", "mentions": "+MENT", "twidf": "+TW-IDF",
               "cooccurs": "+COOCC", "full": "Full"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    # Left: AUC line per method across the enrichment axis
    ax = axes[0]
    x = np.arange(len(variants))
    for j, m in enumerate(methods):
        ys = [ga[v]["methods"][m]["AUC"] for v in variants]
        ax.plot(x, ys, marker="o", color=OI[j], label=m,
                lw=(2.4 if m == "PPR" else 1.3))
    if "G_channel" in ga:
        ax.axhline(ga["G_channel"]["AUC"], color=OI[6], ls="--", lw=1.2,
                   label="G (feature clf.)")
    ax.set_xticks(x); ax.set_xticklabels([vlabels[v] for v in variants], fontsize=8)
    ax.set_ylabel("AUC (online)"); ax.set_title("AUC as the CSTG graph is enriched")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # Right: MCC the same way (balanced-metric view)
    ax = axes[1]
    for j, m in enumerate(methods):
        ys = [ga[v]["methods"][m]["MCC"] for v in variants]
        ax.plot(x, ys, marker="s", color=OI[j], lw=(2.4 if m == "PPR" else 1.3))
    if "G_channel" in ga:
        ax.axhline(ga["G_channel"]["MCC"], color=OI[6], ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([vlabels[v] for v in variants], fontsize=8)
    ax.set_ylabel("MCC (online)"); ax.set_title("MCC as the CSTG graph is enriched")
    fig.suptitle("CSTG online graph ablation: 5 graph methods over separated CSTG "
                 "layers (PPR bold)", y=1.03, fontsize=11)
    _save(fig, "fig_cstg_graph_ablation")


def main():
    profile = _load("kg_profile.json")
    growth = _load("growth.json")
    arrays = _load("growth_arrays.json")
    build = _load("build_complexity.json")
    lat = _load("prediction_latency.json")
    # final experiments live in the main outputs dir (produced by run_final_experiments)
    fe_path = C.ROOT / "outputs" / "final_experiments_results.pkl"
    final_exp = None
    if fe_path.exists():
        import pickle
        final_exp = pickle.load(open(fe_path, "rb"))

    print("rendering figures ->", C.FIG)
    fig_growth_cumulative(arrays)
    fig_growth_year(growth)
    fig_delta_size_box(growth)
    fig_degree_loglog(profile)
    fig_build_cost(build)
    fig_predict_latency(lat)
    fig_pareto_cost_accuracy(lat, final_exp)
    fig_cstg_ablation(_load("cstg_ablation.json"))
    fig_cstg_graph_ablation(_load("cstg_graph_ablation.json"))
    print("done.")


if __name__ == "__main__":
    main()
