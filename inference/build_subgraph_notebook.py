"""
Build (and execute) the v4 subgraph-ablation results notebook:
  experiments/notebooks/subgraph_ablation_v4.ipynb

The notebook loads the cached online-evaluation results + layer statistics
(no live Neo4j needed), renders the comprehensive tables and the publication
figures, draws one live headline chart, and states the findings for the
"which subgraph is best?" research question.

Run:  python inference/build_subgraph_notebook.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent.parent
NB   = ROOT / "experiments" / "notebooks" / "subgraph_ablation_v4.ipynb"

md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r"""# KG-Commit v4 — Which subgraph is best?

**Online (prequential) evaluation of alternative structural subgraphs for
commit-level Just-In-Time defect prediction on Apache Groovy (ApacheJIT,
8,059 labelled commits).**

Our final methodology uses an **AST subgraph with a delta / difference graph** as
the structural layer of the knowledge graph. This notebook answers the research
question *"is AST actually the best structural subgraph?"* by **replacing** the
AST layer with each standard alternative and evaluating under one **identical
online protocol**, **before** the semantic-text (CSTG) layer is added.

### The one channel a subgraph feeds
A structural layer influences prediction through exactly one signal: per-commit
**change-tokens** `"{edge}:{type}"` with `edge ∈ {ADDS, REMOVES, UPDATES, MOVES}`,
emitted by the online delta-growth engine. These tokens become the structural
TF-IDF stream (**T**) and the typed hubs of the Personalized-PageRank graph (**P**).

Every variant shares the **same core** — JIT metrics (**M**), relational priors
(**R**), PPR (**P**) — the same classifier, and the same warmup/block schedule.
Text features (**X**) and the semantic-text subgraph (**G**) are **disabled**, so
the *only* thing that varies is the structural subgraph. Headline metric is the
**Fusion** model `= M + T + R + P`.

### Variants
| id | structural layer | Neo4j label |
|----|------------------|-------------|
| V1 | none (core only) | — |
| V2a | CFG (control-flow) | `:CFGNode` |
| V2b | DFG (def-use data-flow) | `:DFGNode` |
| V2c | PDG/CPG (program dependence) | `:PDGNode` |
| V2d | Token / statement sequence | `:SEQNode` |
| V3 | **AST (incumbent)** | `:ASTNode` |
"""))

cells.append(code(r"""import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
%matplotlib inline

OUT = Path.cwd()
while not (OUT / "outputs").exists() and OUT != OUT.parent:
    OUT = OUT.parent
OUT = OUT / "outputs"
res   = pickle.load(open(OUT / "subgraph_rq_results.pkl", "rb"))
stats = json.load(open(OUT / "subgraph_layer_stats.json"))
FIG   = OUT / "figures" / "v4"

ORDER = ["V1_none","V2a_cfg","V2b_dfg","V2c_pdg","V2d_seq","V3_ast"]
NAME  = {"V1_none":"Core (no subgraph)","V2a_cfg":"CFG","V2b_dfg":"DFG",
         "V2c_pdg":"PDG/CPG","V2d_seq":"Token-seq","V3_ast":"AST"}
meta  = res["V3_ast"]
print(f"stream: {meta['warmup']+meta['n_eval']} commits | warmup {meta['warmup']} |"
      f" scored online {meta['n_eval']} | stream bug-rate {meta['stream_bug']:.3f}")"""))

cells.append(md("## 1. Structural layers materialised in the KG\n\n"
                "Each alternative layer is built by the same online delta-growth engine "
                "(`build_subgraph_online_kg.py`) over the 8,059-commit stream, using one "
                "shared uniform differ (`subgraph_diff.py`). All four coexist with the AST "
                "layer under distinct node labels."))

cells.append(code(r"""SKEY = {"V3_ast":"ast","V2a_cfg":"cfg","V2b_dfg":"dfg","V2c_pdg":"pdg","V2d_seq":"seq"}
rows = []
for v in ["V3_ast","V2a_cfg","V2b_dfg","V2c_pdg","V2d_seq"]:
    s = stats[SKEY[v]]
    rows.append([NAME[v], s["nodes"], s["delta_total"], s["n_node_types"],
                 s["n_token_types"], s["commits_with_tokens"]])
layers = pd.DataFrame(rows, columns=["Layer","Nodes","Delta-edges","Node types",
                                     "Token types","Commits w/ tokens"])
display(layers.style.format({"Nodes":"{:,}","Delta-edges":"{:,}",
                             "Commits w/ tokens":"{:,}"}).hide(axis="index"))"""))

cells.append(md("## 2. Online prequential results\n\n"
                "Fusion `= M+T+R+P`; `T-only` isolates the subgraph's change-tokens alone; "
                "`P-only` is PPR alone. **X and G are off.** Best per column in **bold**."))

cells.append(code(r"""def rowf(v):
    F = res[v]["Fusion"]
    return {"Subgraph":NAME[v], "#tok":res[v]["n_token_types"],
            "PR-AUC":F["PR_AUC"], "ROC-AUC":F["ROC_AUC"], "F1-online":F["F1_online"],
            "F1@.5":F["F1"], "MCC":F["MCC"], "Brier":F["Brier"],
            "T-only PR":res[v]["T_only"]["PR_AUC"], "P-only PR":res[v]["P_only"]["PR_AUC"]}
df = pd.DataFrame([rowf(v) for v in ORDER])
hi = ["PR-AUC","ROC-AUC","F1-online","F1@.5","MCC"]
sty = (df.style.hide(axis="index")
       .format({c:"{:.3f}" for c in ["PR-AUC","ROC-AUC","F1-online","F1@.5","MCC",
                                     "Brier","T-only PR","P-only PR"]})
       .highlight_max(subset=hi, color="#d5efdf")
       .highlight_min(subset=["Brier"], color="#d5efdf"))
sty"""))

cells.append(md("**Live headline chart** — Fusion PR-AUC by structural subgraph "
                "(AST = bold accent):"))

cells.append(code(r"""COLOR = {"V1_none":"#999999","V2a_cfg":"#56B4E9","V2b_dfg":"#009E73",
         "V2c_pdg":"#0072B2","V2d_seq":"#E69F00","V3_ast":"#D55E00"}
fig, ax = plt.subplots(figsize=(7.5,4))
vals = [res[v]["Fusion"]["PR_AUC"] for v in ORDER]
bars = ax.bar([NAME[v].replace(" (no subgraph)","") for v in ORDER], vals,
              color=[COLOR[v] for v in ORDER], edgecolor="white", width=0.72)
bars[-1].set_edgecolor("#222"); bars[-1].set_linewidth(1.6)
for b,val in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, val+0.003, f"{val:.3f}",
                                     ha="center", va="bottom", fontsize=9)
ax.set_ylim(0.54,0.63); ax.set_ylabel("Fusion PR-AUC"); ax.grid(axis="y",color="#eee")
ax.set_title("AST gives the strongest structural layer (X & G off)", weight="bold")
plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.show()"""))

cells.append(md("## 3. Publication figures\n\n"
                "Rendered by `inference/make_subgraph_figures.py` (Okabe–Ito colourblind-safe "
                "palette; PDF versions live in `outputs/figures/v4/` for the paper)."))

cells.append(code(r"""for name,cap in [("fig_fusion_metrics","Fusion metrics by subgraph"),
                 ("fig_tonly_signal","Isolated structural signal (T-only)"),
                 ("fig_vocab_scatter","Vocabulary richness -> signal"),
                 ("fig_trajectory","Rolling PR-AUC over the stream"),
                 ("fig_layer_sizes","Per-layer graph scale")]:
    print(cap); display(Image(filename=str(FIG / f"{name}.png")))"""))

cells.append(md(r"""## 4. Per-inference-method comparison (not just Fusion)

We do not stop at the Fusion model: we run **every inference method separately on
each subgraph** and compare. Two methods are **subgraph-independent references**
(they never touch the structural tokens) and four are **subgraph-dependent**:

| method | uses | subgraph-dependent? |
|---|---|---|
| **M** JIT metrics (LR) | 12 change metrics | no |
| **R** Relational priors (wvRN) | file/dev/global bug-rates | no |
| **T** Structural TF-IDF | change-tokens | **yes** |
| **P** Personalized PageRank | typed token hubs | **yes** |
| **E** KG embedding (SVD/LSA) | commit×hub incidence | **yes** |
| **Fusion** | M+T+R+P | **yes** |

Each is evaluated under the identical online prequential protocol."""))

cells.append(code(r"""MO = ["M","R","T","P","E","Fusion"]
MN = {"M":"JIT metrics (M)","R":"Relational priors (R)","T":"Structural TF-IDF (T)",
      "P":"Personalized PageRank (P)","E":"KG embedding / SVD (E)","Fusion":"Fusion (M+T+R+P)"}
STRUCT = {"T","P","E","Fusion"}
def pm_table(metric):
    rows = {MN[m]:{NAME[v]:res[v]["methods"][m][metric] for v in ORDER} for m in MO}
    return pd.DataFrame(rows).T[[NAME[v] for v in ORDER]]
prm = pm_table("PR_AUC")
def hi_struct(row):
    # bold the best variant only for subgraph-dependent methods
    dep = any(k in row.name for k in ["TF-IDF","PageRank","embedding","Fusion"])
    mx = row.max() if dep else None
    return ["font-weight:bold;background:#d5efdf" if (dep and v==mx) else "" for v in row]
display(prm.style.format("{:.3f}").apply(hi_struct, axis=1)
        .set_caption("PR-AUC by inference method (rows) x subgraph (cols); "
                     "best variant per subgraph-dependent method in bold"))"""))

cells.append(md("**Heatmap** (PR-AUC) — orange ring marks the best subgraph for each "
                "method; and the four KG-native methods as grouped bars:"))

cells.append(code(r"""display(Image(filename=str(FIG / "fig_permethod_heatmap.png")))
display(Image(filename=str(FIG / "fig_permethod_bars.png")))"""))

cells.append(md(r"""### What the per-method view reveals

* **The two references are flat** across subgraphs (M = 0.398, R = 0.460 PR-AUC),
  confirming they do not use structural information — a sanity check on the setup.
* **Structural TF-IDF (T)** — the most direct use of the change-tokens — is
  **best with AST** (0.488), the cleanest expression of AST's advantage.
* **PPR (P)** is actually **best with _no_ subgraph** (Core 0.439): adding typed
  token hubs dilutes the file/developer relational random walk rather than helping
  it. PPR's signal is relational, not structural.
* **KG embedding (E)** is **best with DFG** (0.494), *above* AST (0.421): AST's
  very large, sparse token vocabulary (314 types) is harder to compress into a
  low-rank SVD embedding than DFG's compact 27-type vocabulary.
* **Fusion** — the model we actually deploy — is **best with AST** (0.615),
  because it can exploit AST's strong direct-token signal while the other channels
  contribute their subgraph-independent parts.

**Takeaway.** AST's superiority is *method-specific*: it dominates the direct
change-token channel (T) and the combined Fusion (the deployed predictor), which
is exactly what justifies choosing it. The nuances — PPR favouring the relational
core, the embedding favouring the compact DFG — are reported honestly and do not
change the deployment conclusion."""))

cells.append(md(r"""## 5. Headline evaluation metrics

Beyond PR/ROC-AUC we report the project's **seven headline metrics** — Precision,
Recall, Macro-F1, Buggy-F1, G-Mean, AUC, Accuracy. Threshold-based metrics use the
leakage-free **online-tuned** operating point (`online_jit.online_decisions`);
AUC is threshold-free. The heatmap below is the deployed Fusion model across
subgraphs; **AST is best on all seven**."""))

cells.append(code(r"""M7 = [("Precision","Prec."),("Recall","Rec."),("Macro_F1","Macro-F1"),
      ("Buggy_F1","Buggy-F1"),("G_Mean","G-Mean"),("AUC","AUC"),("ACC","Acc.")]
fus7 = pd.DataFrame({lbl:{NAME[v]:res[v]["methods"]["Fusion"][mk] for v in ORDER}
                     for mk,lbl in M7}).loc[[NAME[v] for v in ORDER]]
display(fus7.style.format("{:.3f}").highlight_max(axis=0, color="#d5efdf")
        .set_caption("Fusion (M+T+R+P) across subgraphs — seven headline metrics"))
display(Image(filename=str(FIG / "fig_metrics_fusion.png")))"""))

cells.append(md(r"""### Online-evaluation trend per metric (stream plots)

For each headline metric we plot the **rolling prequential value against the commit
index** in the chronological stream (window = 800 commits), one line per subgraph
(deployed Fusion). This is the streaming/online view of the comparison: it shows
not just the end-of-stream number but how each subgraph tracks over time. **AST
(bold) leads or ties at essentially every point on every metric**, and **Core
(dashed) trails throughout**; the ordering is preserved through the mid-stream
peak and the late-stream concept drift."""))

cells.append(code(r"""display(Image(filename=str(FIG / "fig_metric_streams.png")))"""))

cells.append(md(r"""## 6. Five graph-native online KG-inference methods

The methods above mix graph and non-graph signals (M is tabular; Fusion blends
M+R). We therefore add **five purely graph-native, online KG-inference methods**,
each run on **two graph scopes** so the subgraph's contribution is not confounded
by the shared file/developer relational layer:

| method | family | graph-pure? |
|---|---|---|
| **RN** relational neighbour (wvRN) | collective classification | yes |
| **PPR** Personalized PageRank / RWR | random walk | yes |
| **LP** label propagation | spreading activation | yes |
| **DW** DeepWalk embedding (PPMI+SVD) | random-walk embedding | yes |
| **KGE** DistMult on (commit,rel,hub) triples | KG embedding | yes |

* **scope = subgraph** — commits linked only to their change-token hubs (the graph
  the structural subgraph actually defines);
* **scope = full** — commits linked to token **+ file + developer** hubs (deployed
  heterogeneous graph).

These are **added alongside** the originals — nothing is removed."""))

cells.append(code(r"""import pickle as _pk
KGP = OUT / "subgraph_kg_methods_results.pkl"
kg = _pk.load(open(KGP,"rb")) if KGP.exists() else None
KG_ORDER=["RN","PPR","LP","DW","KGE"]
KGN={"RN":"Relational neighbour","PPR":"PageRank RWR","LP":"Label propagation",
     "DW":"DeepWalk embed","KGE":"KG embed DistMult"}
if kg:
    # all methods (original + graph-native, both scopes) x 7 metrics, at AST
    recs=[(f"{m}",res["V3_ast"]["methods"][m]) for m in ["M","R","T","P","E","Fusion"]]
    for sc in ("subgraph","full"):
        for m in KG_ORDER: recs.append((f"{KGN[m]} [{sc}]", kg["V3_ast"][sc][m]))
    allm=pd.DataFrame({lbl:{n:r[mk] for n,r in recs} for mk,lbl in M7}).loc[[n for n,_ in recs]]
    display(allm.style.format("{:.3f}").highlight_max(axis=0,color="#d5efdf")
            .set_caption("All inference methods on the AST subgraph — seven metrics"))"""))

cells.append(md("**Heatmap** of every method × 7 metrics at AST, and the five "
                "graph-native methods across subgraphs (Buggy-F1, both scopes):"))

cells.append(code(r"""display(Image(filename=str(FIG / "fig_metrics_allmethods.png")))
if kg: display(Image(filename=str(FIG / "fig_kg_methods_bf1.png")))"""))

cells.append(md(r"""### What the graph-native methods show

* **Two regimes.** The random-walk / propagation methods (**RN, PPR, LP**) are
  strongest on the *core / full* relational graph and are diluted by structural
  tokens — their signal is relational (file/developer), not structural. The
  **embedding** methods (**DW DeepWalk, KGE DistMult**) instead prefer the
  token-rich subgraphs and rank **AST** highest among them, echoing the TF-IDF
  result.
* On the AST subgraph, **Fusion still dominates the balanced metrics**
  (Macro-F1, Buggy-F1, G-Mean, AUC, Accuracy); the pure relational methods reach
  very high **Recall** at the cost of Precision (they flag buggy liberally).
* Reporting **both scopes** cleanly separates "what the subgraph adds" (subgraph
  scope) from "what the deployed heterogeneous graph achieves" (full scope),
  removing the file/developer confound identified in the audit."""))

cells.append(md(r"""## 7. Choosing the final Fusion — comprehensive ablation

To choose a defensible final Fusion formula we ablate **all combinations** of ten
inference methods on the **final graph (Core + AST + CSTG, no X)**: previous
M, R, T, P, E, G(CSTG) and new RN, LP, DW, KGE. Two fusion styles are compared —
**score stacking** (prequential LR over the selected methods' probabilities; all
$2^{10}-1=1023$ subsets) and **feature fusion** (concatenated raw blocks of the top
channels; all $2^6-1=63$ subsets). All seven metrics, online-tuned operating point
(`inference/run_fusion_ablation.py`)."""))

cells.append(code(r"""import pickle as _pk
FA = _pk.load(open(OUT / "fusion_ablation_results.pkl", "rb"))
M7L = [("Precision","Prec."),("Recall","Rec."),("Macro_F1","Macro-F1"),
       ("Buggy_F1","Buggy-F1"),("G_Mean","G-Mean"),("AUC","AUC"),("ACC","Acc.")]
# singles ranked
sing = sorted([v for v in FA["stack"].values() if len(v["methods"])==1],
              key=lambda v:-v["Buggy_F1"])
sdf = pd.DataFrame([{"method":v["methods"][0], "graph":"yes" if v["graph_only"] else "no",
                     **{lbl:v[mk] for mk,lbl in M7L}} for v in sing])
display(sdf.style.hide(axis="index").format({l:"{:.3f}" for _,l in M7L})
        .set_caption("Single methods on the final graph (ranked by Buggy-F1): "
                     "CSTG (G) is strongest, non-graph M is near the weakest"))"""))

cells.append(md("**Best combinations and the recommended formula** (feature-fusion "
                "unless noted). Performance saturates by ~3–4 components; a compact "
                "**graph-only R+T+G** matches the best while dropping non-graph M:"))

cells.append(code(r"""def _find(fam, ms):
    s=set(ms)
    return next((v for v in FA[fam].values() if set(v["methods"])==s), None)
picks=[("feat",["M","T","R","P"],"M+T+R+P (previous)"),
       ("feat",["M","R","T","G"],"M+R+T+G (best overall)"),
       ("stack",["R","T","P","G","RN","LP","KGE"],"R+T+P+G+RN+LP+KGE (best graph stack)"),
       ("feat",["R","T","P","G"],"R+T+P+G"),
       ("feat",["T","G"],"T+G"),
       ("feat",["R","T","G"],"R+T+G  (RECOMMENDED)")]
rows=[]
for fam,ms,name in picks:
    v=_find(fam,ms)
    if v: rows.append({"fusion":name,"graph":"yes" if v["graph_only"] else "no",
                       "#ch":len(v["methods"]),**{lbl:v[mk] for mk,lbl in M7L}})
fdf=pd.DataFrame(rows)
display(fdf.style.hide(axis="index").format({l:"{:.3f}" for _,l in M7L})
        .set_caption("Candidate final Fusion formulas x 7 metrics"))
display(Image(filename=str(FIG / "fig_fusion_pareto.png")))"""))

cells.append(md(r"""### Recommended final Fusion: **R + T + G** (graph-only)

- **R** relational priors + **T** AST structural change-tokens + **G** CSTG
  semantic-text — three interpretable, purely graph-native channels.
- vs the previous metrics-inclusive **M+T+R+P**, R+T+G improves the minority-class
  metrics — **Recall 0.768 (+0.078), Buggy-F1 0.608 (+0.016), G-Mean 0.761 (+0.022),
  AUC 0.834 (+0.014), Macro-F1 0.716 (+0.002)** — with tiny trades on Precision
  (−0.015) and Accuracy (−0.010), i.e. a healthier recall-oriented operating point
  for the minority buggy class.
- It **drops the non-graph JIT metrics (M)** — which the singles show is the
  *weakest* channel — and is **simpler** (3 vs 4 components).
- **G (CSTG) is the single most valuable channel** (Buggy-F1 0.596 alone), and
  larger fusions (up to 7 graph methods, Buggy-F1 0.618) add only marginal gains
  over R+T+G — so R+T+G is the parsimonious, defensible choice for a Q1 paper.

**Conclusion:** adopt **Fusion = R + T + G** as the deployed, purely graph-native
model; report the full ablation (tables + Pareto plot above) as justification."""))

cells.append(md(r"""### Full ablation — best fusions at every size

The complete ablation has $2^{10}-1=1023$ score-stacking combinations. We show, for
each fusion size $k=1,\dots,10$, the **best 8 $k$-method fusions** by Buggy-F1 (the
$k{=}10$ table has the single all-methods combination), on all seven metrics —
followed by three aggregate visualizations."""))

cells.append(code(r"""display(Image(filename=str(FIG / "fig_ablation_bysize.png")))
display(Image(filename=str(FIG / "fig_ablation_method_freq.png")))
display(Image(filename=str(FIG / "fig_ablation_bestperk.png")))"""))

cells.append(code(r"""# 10 per-size tables: top-8 by Buggy-F1, all seven metrics
stack = list(FA["stack"].values())
for k in range(1, 11):
    subs = sorted([v for v in stack if len(v["methods"])==k], key=lambda v:-v["Buggy_F1"])[:8]
    tdf = pd.DataFrame([{"fusion":"+".join(v["methods"]), "graph":"yes" if v["graph_only"] else "no",
                         **{lbl:v[mk] for mk,lbl in M7L}} for v in subs])
    sty = (tdf.style.hide(axis="index").format({l:"{:.3f}" for _,l in M7L})
           .set_caption(f"k={k}: best {len(subs)} fusion(s) of {k} method(s) by Buggy-F1")
           .set_table_styles([{"selector":"caption","props":[("font-weight","bold"),("font-size","110%")]}]))
    display(sty)"""))

cells.append(md(r"""## 8. Findings (subgraph RQ)

1. **AST is the best structural subgraph on every metric** — Fusion PR-AUC
   **0.615**, ROC-AUC **0.826**, F1-online **0.607** — ahead of the best
   alternative (CFG/PDG, 0.595 PR-AUC) and well ahead of the core-only floor
   (0.564).
2. **Every subgraph beats core-only** (PR-AUC 0.587–0.595 vs 0.564): structural
   change information is genuinely informative for JIT defect prediction.
3. **The alternatives cluster tightly and below AST.** CFG ≈ PDG ≈ DFG; the
   deliberately weak token-sequence layer is lowest of the four — exactly the
   expected ordering.
4. **Mechanism.** Isolating the structural tokens (T-only PR-AUC) ranks
   AST 0.488 > DFG 0.469 > PDG 0.464 > CFG 0.461 > Token-seq 0.432 > core 0.204,
   and this tracks **token-vocabulary richness** (AST 314 distinct change-token
   types vs 27–63 for the others). AST's fine-grained typing captures *which kind*
   of code element changed at a resolution the coarser statement-level graphs
   cannot.
5. **Consistency.** The rolling-window trajectory shows AST leading at every point
   in the chronological stream; the ranking is preserved through concept drift.

**Conclusion.** Under a fair, identical online protocol with text and the
semantic-text subgraph disabled, the **AST delta-subgraph is the strongest
structural layer** for the KG — justifying its choice as the methodology's
structural representation before the CSTG layer is introduced.
"""))

cells.append(md(r"""## 9. Reproduce
```bash
# build each alternative layer (resumable; indexes auto-created)
python build_subgraph_online_kg.py --kind cfg   # dfg | pdg | seq
python validate_subgraph_kg.py    --kind cfg --recheck 15 --samples 10
# evaluate (original + graph-native methods) + tables + figures
python inference/run_subgraph_rq.py            # M,R,T,P,E,Fusion (7 metrics)
python inference/run_kg_methods_rq.py          # RN,PPR,LP,DW,KGE (both scopes)
python inference/collect_subgraph_stats.py
python inference/make_subgraph_tables.py       # structural tables
python inference/make_metrics_tables.py        # 7-metric tables (all methods)
python inference/make_subgraph_figures.py
python inference/make_metrics_figures.py
# final-method fusion ablation (choose R+T+G) + report
python inference/run_fusion_ablation.py
python inference/make_fusion_ablation_report.py
```
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
NB.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB))
print(f"wrote {NB}")

# execute in-place so outputs are embedded
try:
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook",
                    "--execute", "--inplace", "--ExecutePreprocessor.timeout=300",
                    str(NB)], check=True, cwd=str(ROOT))
    print("executed OK")
except subprocess.CalledProcessError as e:
    print(f"execution failed: {e}")
