"""
Build (and execute) the V4 scalability / complexity / statistics notebook:
  experiments/notebooks/scalability_v4.ipynb

The notebook loads ONLY cached artifacts (outputs/scalability/*.json + the final
result pickles) -- no live Neo4j, no KG rebuild -- and renders every scalability
table and figure inline with the per-phase findings, for the Q1 write-up. It
covers all four phases: the Core graph, the structural-subgraph family
(AST/CFG/DFG/PDG/Token-seq), the CSTG layer on the final graph, and the five
graph-inference methods + deployed fusion.

Run:  python scalability/build_scalability_notebook.py          # build + execute
      python scalability/build_scalability_notebook.py --no-exec
"""
import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "experiments" / "notebooks" / "scalability_v4.ipynb"

md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r"""# KG-Commit V4 --- Scalability, Complexity & Statistical Analysis

**Apache Groovy, ApacheJIT, 8,059 labelled commits.** This notebook is the
scalability companion to the V4 methodology. It answers, for the *final* graph and
inference stack, four questions a Q1 reviewer expects:

1. **How big is the final KG, and how is it distributed?** (statistical profile)
2. **How does it grow and get modified per commit?** (online growth analysis)
3. **What does it cost to build, modify, and predict?** (time/space complexity)
4. **Are the headline accuracy gaps statistically real?** (significance testing)

Everything here is computed **without rebuilding the KG**: the final graph
statistics come from read-only Neo4j snapshots cached to `outputs/scalability/`,
the growth curves from attributing delta edges to commits in online order, the
prediction latency from replaying the cached graph family in-memory, and the
build cost from a sampled replay of the real parse+diff work. The five inference
methods are the final ones --- **RN, PPR, LP, DW, KGE** --- on the six-graph family
(Core, +AST, +CFG, +DFG, +PDG, and the deployed **Core+AST+CSTG**); the deployed
model is the graph-native fusion **F+G = RN+PPR+CSTG**.
"""))

cells.append(code(r"""import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import Image, display, Markdown

ROOT = Path.cwd()
while not (ROOT / "outputs" / "scalability").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
SCAL = ROOT / "outputs" / "scalability"
FIG  = ROOT / "docs" / "figures" / "v4" / "scalability"

def load(name):
    p = SCAL / name
    return json.load(open(p)) if p.exists() else None

profile = load("kg_profile.json")
growth  = load("growth.json")
build   = load("build_complexity.json")
lat     = load("prediction_latency.json")
sig     = load("significance.json")
print("loaded:", [n for n,v in [("profile",profile),("growth",growth),
      ("build",build),("latency",lat),("significance",sig)] if v])
"""))

# ---- Phase overview / final KG profile -------------------------------------
cells.append(md(r"""## 1. Statistical profile of the final knowledge graph

Node/edge scale, type vocabulary, per-file size distribution and the heavy-tail
(power-law) exponent of the per-commit change size, for every materialised layer
plus the CSTG semantic layer. The AST layer is an order of magnitude larger than
the others purely because of *granularity* (a node per syntactic construct vs.\ a
node per statement); the power-law exponents $2<\alpha<3$ show change sizes are
heavy-tailed across all layers.
"""))
cells.append(code(r"""rows = []
for vid in ["ast","cfg","dfg","pdg","seq"]:
    r = profile.get(vid, {})
    if not r: continue
    pl = r.get("delta_per_commit_powerlaw", {})
    rows.append(dict(Layer=r["pretty"], Nodes=r["nodes"], DeltaEdges=r["delta_total"],
                     NodeTypes=r["n_node_types"], TokenTypes=r["n_token_types"],
                     MedNodesPerFile=round(r["per_file_nodes"]["p50"]),
                     Gini=round(r["per_file_nodes"]["gini"],2),
                     PL_alpha=round(pl.get("alpha", float("nan")),2)))
cstg = profile.get("cstg", {})
display(pd.DataFrame(rows).set_index("Layer"))
print("CSTG:  Term=%d  MENTIONS=%d  COOCCURS=%d  GROUNDS_IN=%d  Intent=%d (HAS_INTENT=%d)" % (
    cstg.get("n_Term",0), cstg.get("e_MENTIONS",0), cstg.get("e_COOCCURS",0),
    cstg.get("e_GROUNDS_IN",0), cstg.get("n_Intent",0), cstg.get("e_HAS_INTENT",0)))
"""))
cells.append(code(r"""display(Image(str(FIG/"fig_degree_loglog.png")))"""))

# ---- Growth -----------------------------------------------------------------
cells.append(md(r"""## 2. Per-commit growth and modification (online order)

Because commits are ingested in `author_ts` order and only deltas are stored, the
KG grows with the *amount of change*, not with history length. The per-commit
change size is strongly separated between buggy and benign commits (Mann--Whitney
$p \ll 0.001$ on every layer), which is the basic reason the graph is predictive.
"""))
cells.append(code(r"""rows = []
for vid in ["ast","cfg","dfg","pdg","seq"]:
    r = growth.get(vid, {})
    if not r: continue
    bb = r["buggy_vs_benign"]; rt = bb["mean_size_ratio"]
    rows.append(dict(Layer=vid.upper(), TotalDelta=r["total_delta_edges"],
                     Churn=round(r["churn_ratio"],2),
                     MedPerCommit=round(r["delta_per_commit"]["p50"]),
                     BuggyMean=round(r["delta_per_buggy"]["mean"]),
                     BenignMean=round(r["delta_per_benign"]["mean"]),
                     Ratio=round(rt["ratio"],2),
                     CI=f"[{rt['lo']:.2f},{rt['hi']:.2f}]",
                     MW_p=f"{bb['mannwhitney_p']:.0e}"))
display(pd.DataFrame(rows).set_index("Layer"))
"""))
cells.append(code(r"""display(Image(str(FIG/"fig_growth_cumulative.png")))
display(Image(str(FIG/"fig_growth_year.png")))
display(Image(str(FIG/"fig_delta_size_box.png")))"""))

# ---- Build/predict complexity ----------------------------------------------
cells.append(md(r"""## 3. Time & space complexity

**Build/modify** (CPU parse+diff, Neo4j write I/O excluded): time is near-linear in
file size ($R^2\approx0.99$), confirming per-commit work is $O(\text{change})$.
AST is ~20x costlier per file than the coarse layers, matching its ~10x node
granularity. **Prediction** latency scales with graph density; the deployed
graph-native fusion runs at sub-millisecond per commit even on the richest graph.
"""))
cells.append(code(r"""rows = []
for vid in ["ast","cfg","dfg","pdg","seq"]:
    r = build.get(vid, {})
    if "ms_per_file" not in r: continue
    rows.append(dict(Layer=vid.upper(), NSamples=r["n"],
                     MedMsPerFile=round(r["ms_per_file"]["p50"],1),
                     R2_vs_nodes=round(r["fit_vs_nodes_after"]["r2"],2),
                     R2_vs_delta=round(r["fit_vs_delta_size"]["r2"],2),
                     EstTotalMin=round(r["extrapolation"].get("est_total_seconds",0)/60,1)))
display(pd.DataFrame(rows).set_index("Layer"))
display(Image(str(FIG/"fig_build_cost.png")))"""))
cells.append(code(r"""graphs = [g for g in ["core","ast","cfg","dfg","pdg","final"] if g in lat]
name = {"core":"Core","ast":"+AST","cfg":"+CFG","dfg":"+DFG","pdg":"+PDG","final":"+AST+CSTG"}
rows = []
for m in ["RN","PPR","LP","DW","KGE"]:
    rows.append(dict(Method=m, **{name[g]: round(lat[g]["predict_ms_per_commit"][m]["median"],3) for g in graphs}))
df = pd.DataFrame(rows).set_index("Method")
display(df)
print("Deployed F=RN+PPR on final graph: %.3f ms/commit  (%.0f commits/s)" % (
    lat["final"]["deployed_F_predict_ms_per_commit"], lat["final"]["deployed_F_throughput_cps"]))
display(Image(str(FIG/"fig_predict_latency.png")))"""))

# ---- Significance -----------------------------------------------------------
cells.append(md(r"""## 4. Statistical significance of the headline comparisons

Paired per-commit out-of-sample scores, replayed from the cached graph family.
DeLong's test (Holm-corrected) confirms: graph richness significantly helps the
informative methods, **PPR is significantly the best** method on the final graph,
and the deployed **semantic channel F+G significantly beats F** on every headline
metric.
"""))
cells.append(code(r"""def dl_rows(block, key="p_holm"):
    out = []
    for k, v in block.items():
        d = v.get("roc_delong", v)
        out.append(dict(Comparison=k, dAUC=round(d["auc_a"]-d["auc_b"],3),
                        p=f"{d.get(key, d.get('p')):.0e}"))
    return out

print("Graph richness (final vs Core):")
display(pd.DataFrame([dict(Comparison=k, dAUC=round(v["auc_a"]-v["auc_b"],3),
        p_holm=f"{v.get('p_holm',v['p']):.0e}")
        for k,v in sig["richness_vs_core"].items() if "final>Core" in k]).set_index("Comparison"))

print("PPR vs other methods on the final graph:")
display(pd.DataFrame(dl_rows(sig["ppr_vs_others_final"])).set_index("Comparison"))

print("Deployed F+G vs F (from run_final_fusion.py):")
eff = sig["fusion"].get("F+G_semantic_effect", {})
display(pd.DataFrame([dict(Metric=k, F=round(v["F"],3), FG=round(v["F+G"],3),
        delta=round(v["delta"],3)) for k,v in eff.items()]).set_index("Metric"))
"""))
cells.append(code(r"""display(Image(str(FIG/"fig_pareto_cost_accuracy.png")))"""))

cells.append(md(r"""## Findings

* **Scale is granularity-driven.** AST (3.70M nodes) is ~10x the coarse layers by
  design (a node per syntactic token vs.\ per statement); PDG shares the CFG's
  vertex set exactly (dependence lives in edges). Change sizes are heavy-tailed
  ($2<\alpha<3$) across every layer.
* **Growth is bounded by change, not history.** Only deltas are stored; per-commit
  work and space both scale with the size of the change. Buggy commits make
  significantly larger structural changes than benign ones on every layer.
* **Cheap to run.** Build cost is near-linear in file size ($R^2\approx0.99$);
  the deployed graph-native fusion predicts at **sub-millisecond per commit**
  (thousands of commits/s), so the whole pipeline is comfortably real-time and
  GPU-free.
* **The accuracy story is statistically real.** Richer graphs significantly help
  PPR/LP/DW/KGE; PPR is significantly the strongest method on the final graph; and
  the semantic channel (F+G) significantly improves every headline metric --- the
  cost--accuracy Pareto front puts PPR (and the F=RN+PPR fusion it anchors) at the
  efficient frontier.
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
NB.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB))
print(f"wrote {NB}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()
    if args.no_exec:
        return
    print("executing notebook ...")
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", str(NB)],
        capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2000:] if r.stdout else "")
    sys.stderr.write(r.stderr[-2000:] if r.stderr else "")
    print("OK" if r.returncode == 0 else f"nbconvert exit {r.returncode}")


if __name__ == "__main__":
    main()
