"""
Build + execute experiments/notebooks/final_experiments.ipynb: the five canonical
graph-inference methods (RN, PPR, LP, DW, KGE) evaluated online on the six-graph
family (Core, +AST, +CFG, +DFG, +PDG, final Core+AST+CSTG), all seven metrics,
plus the full set of online-evaluation stream plots.

Run:  python inference/build_final_experiments_notebook.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "experiments" / "notebooks" / "final_experiments.ipynb"
md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r"""# KG-Commit — Final Experiments

**The five canonical graph-inference methods evaluated online (prequentially) on a
family of six knowledge graphs of increasing richness (Groovy, 8,059 labelled
commits). No commit text ($X$) anywhere.**

### Methods (rows)
| | method | paradigm |
|--|--|--|
| RN | Relational neighbour (wvRN) | collective classification |
| PPR | Personalized PageRank / RWR | random walk |
| LP | Label propagation | label diffusion |
| DW | DeepWalk embedding | random-walk node embedding |
| KGE | DistMult | knowledge-graph embedding |

### Graphs (columns), all sharing the Core commit–file–developer layer
`Core` · `Core+AST` · `Core+CFG` · `Core+DFG` · `Core+PDG` · **`Core+AST+CSTG` (final)**

Reading a **row** answers *which graph helps this method* (the *which-subgraph* RQ);
reading a **column** answers *which method wins on this graph* (the *which-method*
RQ). The final graph adds the CSTG term-hubs so the walks/embeddings also traverse
the semantic signal."""))

cells.append(code(r"""import pickle
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display

OUT = Path.cwd()
while not (OUT / "outputs").exists() and OUT != OUT.parent:
    OUT = OUT.parent
OUT = OUT / "outputs"
R = pickle.load(open(OUT / "final_experiments_results.pkl", "rb"))
FIG = OUT / "figures" / "v4" / "final"

GRAPHS = ["core","ast","cfg","dfg","pdg","final"]
GNAME = {"core":"Core","ast":"Core+AST","cfg":"Core+CFG","dfg":"Core+DFG",
         "pdg":"Core+PDG","final":"Core+AST+CSTG"}
METHODS = ["RN","PPR","LP","DW","KGE"]
M7 = [("Precision","Prec."),("Recall","Rec."),("Macro_F1","Macro-F1"),
      ("Buggy_F1","Buggy-F1"),("G_Mean","G-Mean"),("AUC","AUC"),("ACC","Acc.")]
print(f"warmup={R['meta']['warmup']}  N={R['meta']['N']}  "
      f"eval={R['meta']['N']-R['meta']['warmup']} commits scored online")"""))

cells.append(md("## 1. Results — seven per-metric tables (methods × graphs)\n\n"
                "One table per metric; **best method per graph in bold** (green). "
                "This is the two-RQ answer table."))

cells.append(code(r"""for mk, lbl in M7:
    df = pd.DataFrame({GNAME[g]:{m:R[g][m]["metrics"][mk] for m in METHODS} for g in GRAPHS})
    df = df.loc[METHODS, [GNAME[g] for g in GRAPHS]]
    display(df.style.format("{:.3f}").highlight_max(axis=0, color="#d5efdf")
            .set_caption(f"{lbl} — 5 methods x 6 graphs (best method per graph highlighted)"))"""))

cells.append(md(r"""## 2. Online-evaluation stream plots

The prequential value of each metric over the commit stream (rolling window = 800).
Two complementary views."""))

cells.append(md(r"""### 2a. By method — trends are graphs (35 stream plots)

Five figures (one per method), each with the seven metric panels; within each
panel the six **graphs** are the trends. Shows how graph richness moves each
method's stream (the final graph, bold, is usually on top)."""))

cells.append(code(r"""for m in METHODS:
    display(Image(filename=str(FIG / f"fig_final_bymethod_{m}.png")))"""))

cells.append(md(r"""### 2b. By graph — trends are methods (42 stream plots)

Six figures (one per graph), each with the seven metric panels; within each panel
the five **methods** are the trends (PPR bold). Shows which method leads on each
graph across the stream."""))

cells.append(code(r"""for g in GRAPHS:
    display(Image(filename=str(FIG / f"fig_final_bygraph_{g}.png")))"""))

cells.append(md(r"""## 3. Fusion ablation — best combination of the five methods

We fuse the five methods on the **final graph** by prequential logistic-regression
stacking and evaluate **all $2^5-1=31$ combinations** on the seven metrics. The
final fusion is chosen to balance **load** (fewer methods) and **performance**
(especially Macro-F1 and G-Mean): the smallest combination within a small tolerance
of the best $(\text{Macro-F1}+\text{G-Mean})/2$."""))

cells.append(code(r"""FF = pickle.load(open(OUT / "final_fusion_results.pkl", "rb"))
p1 = FF["part1"]; chosen = FF["chosen"]
def bal(v): return 0.5*(v["metrics"]["Macro_F1"] + v["metrics"]["G_Mean"])
rows = [{"fusion":k, "n":v["n"], "balance":bal(v),
         **{lbl:v["metrics"][mk] for mk,lbl in M7}} for k,v in p1.items()]
abl = pd.DataFrame(rows).sort_values("balance", ascending=False).reset_index(drop=True)
abl.insert(0, "rank", abl.index+1)
def hl(row):
    return ["background:#d5efdf;font-weight:bold" if row["fusion"]==chosen else "" for _ in row]
display(abl.style.hide(axis="index").format({"balance":"{:.3f}", **{l:"{:.3f}" for _,l in M7}})
        .apply(hl, axis=1)
        .set_caption(f"All 31 fusions ranked by (Macro-F1+G-Mean)/2 — chosen: {chosen} (highlighted)"))
print(f"CHOSEN FUSION  F = {chosen}  (fewest methods within {0.005} of the best balance)")"""))

cells.append(md("**Stream plots (Part 1)** — 7 metric panels; trends = the chosen "
                "fusion **F** (bold) and the five single methods:"))

cells.append(code(r"""display(Image(filename=str(FIG.parent / "final" / "fig_final_fusion_part1.png")))"""))

cells.append(md(r"""## 4. Adding CSTG (G) and JIT metrics (M) to the fusion

We append the CSTG semantic channel **G** and the non-graph JIT metrics **M** (and
both) to the chosen graph-method fusion **F**, to see whether they add anything."""))

cells.append(code(r"""p2 = FF["part2"]
rows = [{"fusion":name, **{lbl:p2[name]["metrics"][mk] for mk,lbl in M7}}
        for name in ["F","F+G","F+M","F+G+M"]]
p2df = pd.DataFrame(rows)
display(p2df.style.hide(axis="index").format({l:"{:.3f}" for _,l in M7})
        .highlight_max(axis=0, color="#d5efdf")
        .set_caption(f"F = {chosen}; effect of adding CSTG (G) and JIT metrics (M)"))
display(Image(filename=str(FIG.parent / "final" / "fig_final_fusion_part2.png")))"""))

cells.append(md(r"""**What this shows.** Adding **G (CSTG) gives a large jump**
(Macro-F1 and AUC rise sharply); adding **M (JIT metrics) alone *hurts*** and adds
nothing over G (F+G+M ≈ F+G). So the recommended deployed model is the purely
graph-native **F + G** (= the chosen graph-method fusion plus the CSTG channel),
**without the non-graph JIT metrics** — consistent with the rest of the study."""))

cells.append(md(r"""## 5. Findings

- **Graph richness helps.** For the stronger methods (PPR, LP, DW) the metrics rise
  from Core to the final **Core+AST+CSTG** graph — the CSTG term-hubs let the
  walks/embeddings absorb the semantic signal. **PPR on the final graph is the best
  single (method, graph) cell** (Buggy-F1 ≈ 0.555, AUC ≈ 0.78).
- **PPR is the strongest method** on the informative graphs, with LP and the
  embeddings (DW, KGE) close behind; **RN** (pure relational neighbour) is flat and
  actually prefers the Core graph — its signal is relational, and structural/semantic
  hubs dilute a one-hop vote.
- The ordering holds **across the whole stream** (Section 2), not just at the end,
  and across all seven metrics — so the choice of the five methods and the final KG
  is robust, not an artefact of one operating point.
- These five methods are the inputs to the deployed **fusion** on the final KG
  (reported separately); their combination on the richest graph gives the best
  performance, which is exactly the intended design."""))

cells.append(md(r"""## 6. Reproduce
```bash
python inference/run_final_experiments.py          # 5 methods x 6 graphs, 7 metrics + trajectories
python inference/make_final_experiments_report.py  # tables + 35 + 42 stream plots
python inference/run_final_fusion.py               # 31-combo fusion ablation + F/F+G/F+M/F+G+M
```"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
NB.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB))
print(f"wrote {NB}")
try:
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=300", str(NB)],
                   check=True, cwd=str(ROOT))
    print("executed OK")
except subprocess.CalledProcessError as e:
    print(f"execution failed: {e}")
