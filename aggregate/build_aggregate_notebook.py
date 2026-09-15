"""
Build + execute the cross-project aggregate notebook at
outputs/aggregate/notebooks/aggregate_evaluation.ipynb.

It loads outputs/aggregate/aggregate_results.pkl (produced by
aggregate_projects.py) and renders the general/total evaluation across ALL
projects that have results: Type-1 (macro mean) and Type-2 (commit-weighted
total) for the deployed fusion, the 5 methods, the fusion-combo ablation, the
subgraph variants, and the summed KG size -- plus a macro-vs-total comparison
chart.

Run (after aggregate_projects.py):  python aggregate/build_aggregate_notebook.py
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PKG_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PKG_ROOT.parent / "outputs"
AGG = OUTPUTS / "aggregate"
NB = AGG / "notebooks" / "aggregate_evaluation.ipynb"
NB.parent.mkdir(parents=True, exist_ok=True)
md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r"""# KG-Commit — Cross-Project (General) Evaluation

Aggregates every project under `outputs/<project>/` that has evaluation results,
and reports **two kinds of general evaluation** across all of them:

* **Type 1 — Macro** : the *unweighted mean* of each metric across projects
  (every project counts equally): $\text{macro}(m) = \frac{1}{P}\sum_i m_i$.
* **Type 2 — Total** : the *commit-weighted mean* (bigger projects count more):
  $\text{total}(m) = \frac{\sum_i w_i\, m_i}{\sum_i w_i}$, reported for
  $w=n_{\text{eval}}$ (scored commits $=N-\text{warmup}$) and $w=N$ (all labelled
  commits).

> Example (F1): A: 100 commits F1=0.4 ; B: 200 commits F1=0.8 →
> Type 1 $=(0.4+0.8)/2=0.60$ ; Type 2 $=(0.4\cdot100+0.8\cdot200)/300=0.67$.

Projects without a folder or without a given result file are **skipped**, so this
notebook can be re-run after each new project is built and simply picks up
whatever is present. It reads only the saved result files — it does **not** touch
Neo4j or re-run any experiment."""))

cells.append(code(
"""import pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AGG = Path.cwd()
# notebook executes with cwd = its own dir; resolve outputs/aggregate robustly
for cand in [AGG, AGG.parent, AGG.parents[1] if len(AGG.parents) > 1 else AGG]:
    if (cand / 'aggregate_results.pkl').exists():
        AGG = cand; break
R = pickle.load(open(AGG / 'aggregate_results.pkl', 'rb'))
projs = R['meta']['projects']
print(f"{R['meta']['n_projects']} projects aggregated:", ', '.join(projs))
HEAD = ['Buggy_F1','Macro_F1','G_Mean','AUC','PR_AUC','MCC']"""))

cells.append(md("## Projects included (and per-project commit counts)"))
cells.append(code(
"""idx = R['projects_index']
rows = []
for p in projs:
    ip = idx.get(p, {})
    rows.append(dict(project=p, N=ip.get('N'), warmup=ip.get('warmup'),
                     n_eval=ip.get('n_eval'),
                     chosen_fusion=str(R['fusion']['chosen_per_project'].get(p)) if R.get('fusion') else '-'))
pd.DataFrame(rows).set_index('project')"""))

cells.append(md(r"""## Deployed fusion (F+G) — Type-1 vs Type-2

The deployed model is **F+G** (the per-project best fusion **F** plus the CSTG
channel **G**). Note that **F** is itself a per-project-varying composition
(chosen by the pipeline), shown in the table above."""))
cells.append(code(
"""def block(leaf, keys=HEAD):
    return pd.DataFrame({
        'Type1_macro':  [leaf[k]['macro']       for k in keys],
        'Type2_total_neval': [leaf[k]['total_neval'] for k in keys],
        'Type2_total_N': [leaf[k]['total_N']    for k in keys],
    }, index=keys)

dep = R['fusion']['deployed_F']
print('Deployed model:', R['fusion']['deployed_label'])
block(dep).round(3)"""))

cells.append(md("## Graph-inference methods on the `final` graph"))
cells.append(code(
"""fe = R['final_experiments']
g = 'final' if 'final' in fe['graphs'] else fe['graphs'][-1]
tab = {}
for m in fe['methods']:
    tab[m] = [fe['aggregated'][g][m][k]['macro'] for k in HEAD]
pd.DataFrame(tab, index=HEAD).T.round(3)  # rows=methods, Type-1 macro"""))

cells.append(md("## Fusion-combination ablation (F / F+G / F+M / F+G+M)"))
cells.append(code(
"""combos = R['fusion']['combos']
order = [c for c in ['F','F+G','F+M','F+G+M'] if c in combos]
pd.DataFrame({c: [combos[c][k]['macro'] for k in HEAD] for c in order},
             index=HEAD).T.round(3)"""))

cells.append(md("## Subgraph variants (Fusion leaf), aggregated"))
cells.append(code(
"""sq = R['subgraph_rq']
pd.DataFrame({v: [sq['aggregated'][v]['Fusion'][k]['macro'] for k in HEAD]
              for v in sq['variants']}, index=HEAD).T.round(3)"""))

cells.append(md("## KG size, summed across all projects (per structural layer)"))
cells.append(code(
"""ls = R['layer_stats']['summed']
keys = ['nodes','delta_total','ADDS','REMOVES','UPDATES','MOVES','commits_with_tokens','n_token_types']
pd.DataFrame({lyr: [ls.get(lyr,{}).get(k) for k in keys] for lyr in R['layer_stats']['layers']},
             index=keys).T"""))

cells.append(md(r"""## Type-1 vs Type-2 comparison (deployed fusion)

Where the macro mean and the commit-weighted total diverge, larger projects pull
the total away from the unweighted mean — i.e. the method behaves differently on
big vs small projects."""))
cells.append(code(
"""dep = R['fusion']['deployed_F']
x = np.arange(len(HEAD)); w = 0.27
fig, ax = plt.subplots(figsize=(9,4.2))
ax.bar(x-w, [dep[k]['macro'] for k in HEAD], w, label='Type-1 macro')
ax.bar(x,   [dep[k]['total_neval'] for k in HEAD], w, label='Type-2 total (n_eval)')
ax.bar(x+w, [dep[k]['total_N'] for k in HEAD], w, label='Type-2 total (N)')
ax.set_xticks(x); ax.set_xticklabels(HEAD, rotation=20)
ax.set_ylabel('score'); ax.set_ylim(0,1); ax.grid(axis='y', color='#EEE'); ax.set_axisbelow(True)
ax.set_title(f"Deployed {R['fusion']['deployed_label']}: Type-1 vs Type-2 over {R['meta']['n_projects']} projects")
ax.legend(frameon=False, ncol=3, fontsize=9)
fig.tight_layout(); plt.show()"""))

cells.append(md("""## Baselines — KG method vs. standard JIT change-metric models

Within-project baselines (LR / RF / HGB on the 12 ApacheJIT change metrics, plus
naive references) under the same online protocol. Present only for projects where
`baselines/run_baselines.py` has been run."""))
cells.append(code(
"""if R.get('baselines'):
    bl = R['baselines']['aggregated']
    used = [p for p in projs if idx.get(p,{}).get('baselines')]
    print('baselines from:', used or '(none yet)')
    display(pd.DataFrame({b: [bl[b][k]['macro'] for k in HEAD] for b in R['baselines']['baselines']},
                         index=HEAD).T.round(3))
    if R.get('fusion'):
        dep = R['fusion']['deployed_F']
        cmp = pd.DataFrame({'KG '+R['fusion']['deployed_label']: [dep[k]['macro'] for k in HEAD]}, index=HEAD)
        for b in ['B_RF','B_HGB','B_LR']:
            if b in bl: cmp[b] = [bl[b][k]['macro'] for k in HEAD]
        display(cmp.T.round(3))
else:
    print('No baselines yet. Run: KGC_PROJECT=<p> python baselines/run_baselines.py, then re-aggregate.')"""))

cells.append(md("""## Effort-aware evaluation — Popt / ACC@20%LOC

How well each KG channel ranks buggy commits per unit of inspection effort
(la+ld). Present for projects where `inference/run_effort_eval.py` has been run."""))
cells.append(code(
"""if R.get('effort'):
    ef = R['effort']['aggregated']
    display(pd.DataFrame({m: [ef[m][k]['macro'] for k in ['Popt','ACC20','Buggy_F1','PR_AUC']]
                          for m in R['effort']['models']},
                         index=['Popt','ACC20','Buggy_F1','PR_AUC']).T.round(3))
else:
    print('No effort results yet. Run: KGC_PROJECT=<p> python inference/run_effort_eval.py')"""))

cells.append(md("""## Scalability rolled up across projects

Predict-time latency (ms/commit, median) and build cost (ms/file) aggregated."""))
cells.append(code(
"""if R.get('scalability'):
    sc = R['scalability']
    bc = sc.get('build_ms_per_file', {})
    if bc:
        print('Build cost (ms/file), macro vs total-N:')
        display(pd.DataFrame({lyr: [bc[lyr]['macro'], bc[lyr]['total_N']] for lyr in bc},
                             index=['macro','total_N']).T.round(1))
else:
    print('No scalability roll-up (needs scalability/*.json per project).')"""))

cells.append(md(r"""## Genuinely-pooled fusion (if available)

Type-2 is a commit-weighted *average of per-project scores*. Because F1/AUC are
non-linear, the honest *metric-on-pooled-commits* differs. When projects store
their raw per-commit predictions (`raw_stream`, added to `run_subgraph_rq.py`),
we pool them all and compute one metric set on the concatenated commit set."""))
cells.append(code(
"""pf = R.get('pooled_fusion', {})
if pf.get('available'):
    print(f"pooled over {pf['n_commits_pooled']} commits from {pf['projects_used']}, "
          f"bug rate {pf['pooled_bug_rate']:.3f}")
    display(pd.Series({k: pf['metrics'].get(k) for k in HEAD}).round(3).to_frame('pooled'))
else:
    print('pooled fusion not available yet:', pf.get('note'))
    print('(re-run each project\\'s experiments with the updated run_subgraph_rq.py '
          'to persist raw_stream, then re-run aggregate_projects.py)')"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(nb, str(NB))
print("wrote", NB)

# execute in place (cache-only; no Neo4j)
print("executing notebook ...")
subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                "--execute", "--inplace", "--ExecutePreprocessor.timeout=600",
                str(NB)], check=True)
print("executed OK ->", NB)
