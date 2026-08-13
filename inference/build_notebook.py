"""Build experiments/notebooks/inference_results.ipynb (via nbformat)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Commit-Level Bug Detection — Inference Methods on the Full KG

This notebook evaluates **every** inference method on the finalized KG-Commit
graph (all 8,059 ApacheJIT-labelled apache/groovy commits) and compares them.

All methods are **CPU-only / GNN-free** and evaluated in two protocols:
* **OFFLINE** — a single time-ordered 70/30 split (train on older commits).
* **ONLINE (prequential)** — predict each commit from the *past only*, then
  learn it (the correct just-in-time protocol; expanding-window refit).

Methods: classic JIT metrics, relational priors (wvRN), structural TF-IDF over
AST-change tokens, Personalized PageRank propagation, KG embeddings
(Truncated-SVD/LSA, node2vec, TransE, DistMult), and an early **Fusion**.

The heavy work (feature extraction, embedding training) is done once by
`inference/run_all.py` and cached; this notebook just loads and visualises it.""")

code("""import sys, os
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT/'inference').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT/'inference'))
print('repo root:', ROOT)

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi']=110

import run_all
results = run_all.evaluate_all()        # loads cached results (fast)
split = results['split']
print(f"commits={split['Nc']}  offline test bug-rate={split['test_bug']:.3f}  "
      f"online bug-rate={split['online_bug']:.3f}")""")

md("## 1. Results table (offline vs online)")
code("""def to_df(d):
    return pd.DataFrame({k: v['metrics'] for k, v in d.items()}).T[['ROC_AUC','PR_AUC','F1','MCC']]
off = to_df(results['offline']).add_suffix('_off')
on  = to_df(results['online']).add_suffix('_on')
comp = off.join(on).sort_values('PR_AUC_on', ascending=False)
order = comp.index.tolist()
comp.round(3)""")

md("""## 2. Headline comparison — PR-AUC and ROC-AUC

PR-AUC is the most informative metric under class imbalance (~20% buggy). The
dashed line is the trivial random baseline (= the positive rate).""")
code("""fig, axes = plt.subplots(1, 2, figsize=(14,5))
x = np.arange(len(order)); wdt = 0.38
for ax, metric, rnd in [(axes[0],'PR_AUC',split['online_bug']), (axes[1],'ROC_AUC',0.5)]:
    ax.bar(x-wdt/2, [results['offline'][m]['metrics'][metric] for m in order], wdt, label='offline')
    ax.bar(x+wdt/2, [results['online'][m]['metrics'][metric] for m in order], wdt, label='online')
    ax.axhline(rnd, ls='--', c='gray', lw=1, label='random')
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_ylabel(metric); ax.set_title(metric+' by method'); ax.legend()
plt.tight_layout(); plt.show()""")

md("## 3. F1 and MCC (decision-threshold metrics at 0.5)")
code("""fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ax, metric in [(axes[0],'F1'), (axes[1],'MCC')]:
    ax.bar(x-wdt/2, [results['offline'][m]['metrics'][metric] for m in order], wdt, label='offline')
    ax.bar(x+wdt/2, [results['online'][m]['metrics'][metric] for m in order], wdt, label='online')
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_ylabel(metric); ax.set_title(metric+' by method'); ax.legend()
plt.tight_layout(); plt.show()""")

md("""## 4. ROC and Precision–Recall curves (online / prequential)

The full curves, not just the summary numbers — the fairest visual comparison.""")
code("""fig, axes = plt.subplots(1, 2, figsize=(15,6))
for m in order:
    d = results['online'][m]
    fpr, tpr, _ = roc_curve(d['y'], d['p'])
    axes[0].plot(fpr, tpr, lw=1.4, label=f"{m} ({d['metrics']['ROC_AUC']:.3f})")
    pr, rc, _ = precision_recall_curve(d['y'], d['p'])
    axes[1].plot(rc, pr, lw=1.4, label=f"{m} ({d['metrics']['PR_AUC']:.3f})")
axes[0].plot([0,1],[0,1],'k--',lw=1); axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC (online)')
axes[0].legend(fontsize=8, loc='lower right')
axes[1].axhline(split['online_bug'], ls='--', c='gray', lw=1)
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–Recall (online)')
axes[1].legend(fontsize=8, loc='upper right')
plt.tight_layout(); plt.show()""")

md("## 5. Metric heatmap (online)")
code("""hm = pd.DataFrame({m: results['online'][m]['metrics'] for m in order}).T[['ROC_AUC','PR_AUC','F1','MCC']]
plt.figure(figsize=(7, 0.5*len(order)+1))
sns.heatmap(hm, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label':'score'})
plt.title('Online metrics by method'); plt.tight_layout(); plt.show()""")

md("""## 6. Ranking and the value of KG features

Methods ranked by online PR-AUC. We also compute the **lift of Fusion over the
JIT-metrics baseline** — the core question: do the KG's AST/graph signals help?""")
code("""rank = comp[['PR_AUC_on','ROC_AUC_on','F1_on','MCC_on']].round(3)
display(rank)
base = results['online']['JIT-metrics']['metrics']['PR_AUC']
best = comp['PR_AUC_on'].max(); best_m = comp['PR_AUC_on'].idxmax()
print(f"Best online method: {best_m}  (PR-AUC {best:.3f})")
print(f"JIT-metrics baseline PR-AUC: {base:.3f}")
print(f"Lift of best over baseline: {100*(best-base)/base:+.0f}% relative")""")

md("""## 7. Why it works — the KG bug signal

Two structural facts that make the graph useful: buggy commits make **larger AST
changes**, and the **bug rate drifts over time** (motivating online evaluation).""")
code("""import datetime
meta = results['meta']
fig, axes = plt.subplots(1, 2, figsize=(14,4.5))
db = meta['delta_total'][meta['buggy']==1]; dg = meta['delta_total'][meta['buggy']==0]
axes[0].boxplot([np.log1p(dg), np.log1p(db)], labels=['benign','buggy'], showfliers=False)
axes[0].set(ylabel='log(1 + AST delta edges)', title='AST-change size by label')
yrs = np.array([datetime.datetime.utcfromtimestamp(int(t)).year for t in meta['author_ts']])
dfp = pd.DataFrame({'year':yrs,'buggy':meta['buggy']}).groupby('year').buggy.mean()
axes[1].bar(dfp.index, dfp.values, color='#c0392b', alpha=0.8)
axes[1].set(ylabel='buggy fraction', title='Bug rate per year (temporal drift)')
plt.tight_layout(); plt.show()
print(f"median delta edges  benign={np.median(dg):.0f}  buggy={np.median(db):.0f}")""")

md("""## 8. Conclusions

* **Fusion of KG-native signals with JIT metrics is the strongest** model in the
  online (prequential) setting, clearly beating the JIT-metrics-only baseline —
  the AST/delta layers add real predictive value for commit-level bug detection.
* Among single KG methods, **structural TF-IDF** and **DistMult embeddings** are
  the most competitive; relational priors (wvRN) are a strong cheap baseline.
* All of this is **CPU-only, no GNN** — a practical inference path for the
  resource budget, and a baseline to beat if a GNN is added later.

*Reproduce:* `python inference/run_all.py` (rebuilds the cache), then re-run this
notebook. Per-method code lives in `inference/` (`train_eval.py`,
`advanced_infer.py`, `online_infer.py`, `kge_infer.py`).""")

nb['cells'] = cells
out = Path("experiments/notebooks/inference_results.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
