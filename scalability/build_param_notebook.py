"""
Build (and execute) the parameter-experiments notebook:
  experiments/notebooks/parameter_experiments.ipynb

Presents the setup-parameter sensitivity study for the paper's Discussion: for the
final Core+AST+CSTG graph, the five graph-native methods RN/PPR/LP/DW/KGE and the
deployed fusion F = RN+PPR (no CSTG classifier G, no JIT metrics M), how the seven
headline metrics respond to four setup parameters -- warmup K, label gap M, the
rolling reporting window, and the online CSTG-hub refresh cadence l.

Loads only cached results (outputs/param_experiments/*.json) -- no KG rebuild, no
live DB -- and renders the comprehensive tables and figures inline.

Run:  python scalability/build_param_notebook.py            # build + execute
      python scalability/build_param_notebook.py --no-exec
"""
import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "experiments" / "notebooks" / "parameter_experiments.ipynb"

md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r"""# KG-Commit V4 --- Setup-Parameter Sensitivity (Discussion)

**Apache Groovy, ApacheJIT, 8,059 commits.** How robust is the deployed model to
its online-protocol setup? We sweep four parameters, one at a time, on the **final
`Core+AST+CSTG` graph**, scoring the five graph-native methods **RN, PPR, LP, DW,
KGE** and the deployed fusion **F = RN+PPR** (the CSTG classifier channel *G* and
the JIT-metrics channel *M* are deliberately **not** used). We report the **seven
headline metrics**, both whole-stream and as rolling trajectories.

| Parameter | Meaning |
|-----------|---------|
| **K** (warmup) | fraction of the stream used to fit before scoring starts |
| **M** (label gap) | labels of the *M* commits just before each block are withheld (late-arriving labels); graph structure stays visible |
| **rolling window** | the trailing window each stream-plot point averages over (a *reporting* choice; whole-stream scores are invariant to it) |
| **l** (CSTG refresh) | the CSTG Term-hub weights are refreshed from past-only NPMI every *l* commits (the graph the methods walk) |

All results are precomputed by `scalability/run_param_experiments.py`; this
notebook only loads and displays them (no rebuild, no database).
"""))

cells.append(code(r"""import json
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown

ROOT = Path.cwd()
while not (ROOT / "outputs" / "param_experiments").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
PE  = ROOT / "outputs" / "param_experiments"
FIG = ROOT / "docs" / "figures" / "v4" / "param"
def load(p):
    fp = PE / f"{p}.json"
    return json.load(open(fp)) if fp.exists() else None
DATA = {p: load(p) for p in ["K","M","ROLL","l","KM"]}
M7 = ["Precision","Recall","Macro_F1","Buggy_F1","G_Mean","AUC","ACC"]
METHODS = ["RN","PPR","LP","DW","KGE","F"]
print("loaded:", [p for p,v in DATA.items() if v])
def show(stem): display(Image(str(FIG / f"{stem}.png")))
def grid(d): return [str(v) for v in d["_meta"]["grid"]]
"""))

blocks = [
    ("K", "## 1. Warmup fraction $K$",
     r"""How much history must the model see before it predicts well? Too little "
     warmup starves the methods of past labels; too much wastes stream for "
     evaluation. We look for the smallest $K$ at which the deployed fusion has "
     effectively saturated."""),
    ("M", "## 2. Label-availability gap $M$",
     r"""In practice a commit's bug label is only known once enough time has passed "
     (SZZ needs future fixes). We emulate this by withholding the labels of the "
     $M$ commits immediately before each prediction block, while still letting the "
     methods see those commits' *graph structure*. This is the most deployment-"
     realistic of the four sweeps: it quantifies the cost of late labels."""),
    ("ROLL", "## 3. Rolling / smoothing window",
     r"""The rolling window governs only how the stream *plots* are smoothed -- each "
     point averages the trailing-window commits. The **whole-stream** table scores "
     are therefore invariant to it (identical across values); what changes is the "
     variance and readability of the trajectory. We show the same run summarised at "
     several window widths."""),
    ("l", "## 4. Online CSTG-hub refresh cadence $l$",
     r"""The final graph carries CSTG **Term** hubs. Here their edge weights are "
     re-estimated from past-only NPMI every $l$ commits, so a larger $l$ means "
     staler semantic hub weights on the graph the five methods walk. (The CSTG "
     *classifier* $G$ is still not used.) This isolates how fresh the semantic "
     layer needs to be."""),
]

for pm, title, desc in blocks:
    cells.append(md(title + "\n\n" + desc))
    cells.append(code(f"""d = DATA["{pm}"]
vals = grid(d)
# whole-stream Buggy-F1 / AUC per method x param value
rows = []
for v in vals:
    row = {{"{pm}": v}}
    for m in METHODS:
        w = d[v][m]["whole"]
        row[f"{{m}}:BF1"] = round(w["Buggy_F1"],3); row[f"{{m}}:AUC"] = round(w["AUC"],3)
    rows.append(row)
display(pd.DataFrame(rows).set_index("{pm}"))
print("Deployed fusion F -- all 7 metrics:")
display(pd.DataFrame([dict({pm}=v, **{{k: round(d[v]["F"]["whole"][k],3) for k in M7}}) for v in vals]).set_index("{pm}"))
show("fig_param_{pm}_lines"); show("fig_param_{pm}_heat")
"""))

cells.append(md(r"""## 5. Real-world constraints: ours vs. baselines (fixed gap $M=200$, warmup sweep)

The single most deployment-relevant experiment. We fix the label gap at a realistic
$M=200$ commits and sweep the warmup $K$ from $0$ (cold start) upward, and we add
**labels-only baselines** -- a JIT-metrics logistic regression and the naive prior
bug-rate -- evaluated under the *identical* prequential setup (same gap, same warmup,
same blocks, only the labelled past $[0,i{-}M)$). Because the baselines have no graph,
the gap and the cold start hit them harder: this isolates the value of the knowledge
graph precisely under the constraints a real deployment faces."""))
cells.append(code(r"""d = DATA["KM"]
vals = grid(d)
cols = ["F","PPR","RN","LP","DW","KGE","JIT_LR","Naive"]
rows = []
for v in vals:
    row = {"K": v}
    for c in cols:
        row[c] = round(d[v][c]["whole"]["Buggy_F1"], 3)
    rows.append(row)
print("Buggy-F1 at fixed gap M=200 (ours = F/PPR/RN/LP/DW/KGE ; baselines = JIT_LR/Naive):")
display(pd.DataFrame(rows).set_index("K"))
show("fig_param_KM_compare")
show("fig_param_KM_lines")
"""))
cells.append(code(r"""# margin of the deployed fusion F over the best baseline, per warmup
print("F minus best labels-only baseline (Buggy-F1) -- the KG's real-world advantage:")
rows = []
for v in vals:
    fF = d[v]["F"]["whole"]["Buggy_F1"]
    bb = max(d[v]["JIT_LR"]["whole"]["Buggy_F1"], d[v]["Naive"]["whole"]["Buggy_F1"])
    rows.append(dict(K=v, F=round(fF,3), best_baseline=round(bb,3), margin=round(fF-bb,3)))
display(pd.DataFrame(rows).set_index("K"))
"""))

cells.append(md(r"""## 6. Robustness synthesis

The deployed fusion $F$'s seven metrics across the four sweep parameters at once."""))
cells.append(code(r"""show("fig_param_F_robustness")"""))

cells.append(md(r"""## Takeaways (for the Discussion)

* **Warmup $K$:** performance saturates by a modest warmup; beyond it, extra warmup
  trades evaluation stream for no gain -- our default sits at the knee.
* **Label gap $M$:** the most practically important knob. Withholding recent labels
  degrades all methods gracefully (no collapse), quantifying the real-world cost of
  late-arriving SZZ labels and showing the KG signal is robust to it.
* **Rolling window:** a pure reporting choice -- whole-stream scores are invariant;
  only the trajectory's smoothness changes -- so none of the paper's conclusions
  depend on it.
* **CSTG refresh $l$:** the semantic hubs tolerate stale refresh well, so the online
  NPMI re-propagation can be infrequent (cheap) without hurting the deployed fusion.
* **Ours vs. baselines under real-world constraints:** at a fixed realistic gap
  $M=200$ and across all warmup levels, the deployed fusion and PPR **clearly and
  consistently beat** the labels-only JIT-metrics LR and the naive prior-rate --
  the advantage is largest exactly where it matters (low warmup / cold start),
  because the graph carries structural and relational evidence even when recent
  labels are missing.
* Across every sweep the **method ordering and the fusion's dominance are preserved**,
  i.e. the headline results are not artefacts of a particular setup.
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
NB.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB))
print(f"wrote {NB}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--kernel", default="kgc-py313")
    args = ap.parse_args()
    if args.no_exec:
        return
    print("executing ...")
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", f"--ExecutePreprocessor.kernel_name={args.kernel}",
         "--ExecutePreprocessor.timeout=600", str(NB)],
        capture_output=True, text=True)
    sys.stdout.write((r.stdout or "")[-1200:]); sys.stderr.write((r.stderr or "")[-1200:])
    print("OK" if r.returncode == 0 else f"exit {r.returncode}")


if __name__ == "__main__":
    main()
