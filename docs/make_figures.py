"""
Generate the figures embedded in KG_v1.tex, from the cached result pkls
(no Neo4j needed). Outputs PNGs into docs/figures/.

Run:  python docs/make_figures.py
"""
import sys, pickle, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"inference"))
FIG = ROOT/"docs"/"figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi":130, "font.size":10, "axes.grid":True, "grid.alpha":0.3})

# ---------------------------------------------------------------- bug signal
R = pickle.load(open(ROOT/"outputs"/"inference_results.pkl","rb"))
meta = R["meta"]; import datetime
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
db = meta["delta_total"][meta["buggy"]==1]; dg = meta["delta_total"][meta["buggy"]==0]
ax[0].boxplot([np.log1p(dg), np.log1p(db)], labels=["benign","buggy"], showfliers=False)
ax[0].set(ylabel="log(1 + AST delta edges)", title="AST-change size by label")
yrs = np.array([datetime.datetime.utcfromtimestamp(int(t)).year for t in meta["author_ts"]])
import pandas as pd
br = pd.DataFrame({"y":yrs,"b":meta["buggy"]}).groupby("y").b.mean()
ax[1].bar(br.index, br.values, color="#c0392b", alpha=0.85)
ax[1].set(ylabel="buggy fraction", title="Bug rate per year (temporal drift)")
plt.tight_layout(); plt.savefig(FIG/"bug_signal.png"); plt.close()

# ------------------------------------------------- offline-blocked ROC/PR curves
on = R["online"]
order = sorted(on, key=lambda k: -on[k]["metrics"]["PR_AUC"])
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for m in order:
    d = on[m]; fpr,tpr,_ = roc_curve(d["y"], d["p"]); pr,rc,_ = precision_recall_curve(d["y"], d["p"])
    ax[0].plot(fpr, tpr, lw=1.2, label=f"{m} ({d['metrics']['ROC_AUC']:.2f})")
    ax[1].plot(rc, pr, lw=1.2, label=f"{m} ({d['metrics']['PR_AUC']:.2f})")
ax[0].plot([0,1],[0,1],"k--",lw=0.8); ax[0].set(xlabel="FPR",ylabel="TPR",title="ROC (online, blocked)")
ax[0].legend(fontsize=6, loc="lower right")
ax[1].axhline(R["split"]["online_bug"], ls="--", c="gray", lw=0.8)
ax[1].set(xlabel="Recall",ylabel="Precision",title="Precision–Recall (online, blocked)"); ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig(FIG/"online_curves.png"); plt.close()

# ----------------------------------------------- fully-online JIT comparison bars
J = pickle.load(open(ROOT/"outputs"/"online_jit_results.pkl","rb"))
M = J["methods"]; jbug = J["meta"]["stream_bug"]
names = sorted(M, key=lambda k: M[k]["cum"]["F1_online"])
labels = [M[k]["name"] for k in names]
colors = ["#1f77b4" if M[k]["group"]=="kg" else "#ff7f0e" for k in names]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, metric, rnd in [(ax[0],"F1_online",None),(ax[1],"PR_AUC",jbug)]:
    a.barh(range(len(names)), [M[k]["cum"][metric] for k in names], color=colors)
    a.set_yticks(range(len(names))); a.set_yticklabels(labels, fontsize=8)
    if rnd is not None: a.axvline(rnd, ls="--", c="gray", lw=0.8)
    a.set_xlabel(metric); a.set_title(f"fully-online {metric}")
ax[0].legend(handles=[Patch(color="#1f77b4",label="KG method"),Patch(color="#ff7f0e",label="baseline")],
             loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"online_jit_compare.png"); plt.close()

# ----------------------------------------------- rolling trajectories (PR-AUC)
fig, ax = plt.subplots(figsize=(9.5, 4))
for k, d in M.items():
    t = J["traj"][k]; st = "-" if d["group"]=="kg" else "--"
    lw = 2.4 if k=="K_fusion" else 1.0
    ax.plot(t["idx"], t["PR_AUC"], st, lw=lw, label=d["name"])
ax.axhline(jbug, ls=":", c="gray", label="random")
ax.set(xlabel="commit index (arrival order)", ylabel="rolling PR-AUC",
       title="Prequential PR-AUC as commits arrive (window=800)")
ax.legend(fontsize=6, ncol=2, loc="upper right")
plt.tight_layout(); plt.savefig(FIG/"jit_trajectories.png"); plt.close()

# ----------------------------------------------------------- Fusion ablation
import online_jit as oj
S = oj.precompute_streams(); ABL = oj.ablation_all(S)
ab = FEAT = oj.FEAT_ABBR
full_key = "+".join(ab[f] for f in oj.FEATURES)            # M+T+R+P+X
singles = {ab[f]: ABL[ab[f]]["cum"]["F1_online"] for f in oj.FEATURES}
full = ABL[full_key]["cum"]["F1_online"]
loo = {ab[f]: full - ABL["+".join(ab[g] for g in oj.FEATURES if g!=f)]["cum"]["F1_online"] for f in oj.FEATURES}
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].bar(list(singles), list(singles.values()), color="#1f77b4")
ax[0].set(title="Single-feature online F1", ylabel="F1_online")
ax[1].bar(list(loo), list(loo.values()), color=["#2ca02c" if v>=0 else "#d62728" for v in loo.values()])
ax[1].axhline(0, c="k", lw=0.8)
ax[1].set(title="Leave-one-out: F1_online drop", ylabel="$\\Delta$ F1_online")
plt.tight_layout(); plt.savefig(FIG/"ablation_importance.png"); plt.close()

# all 31 combos ranked by F1_online (M,T,R,P,X)
import pandas as pd
adf = pd.DataFrame([dict(combo=k, n=v["n"], F1=v["cum"]["F1_online"]) for k,v in ABL.items()]).sort_values("F1")
cmap = {1:"#d62728",2:"#ff7f0e",3:"#1f77b4",4:"#2ca02c",5:"#9467bd"}
fig, ax = plt.subplots(figsize=(7.5, 7.0))
ax.barh(range(len(adf)), adf["F1"], color=[cmap[n] for n in adf["n"]])
ax.set_yticks(range(len(adf))); ax.set_yticklabels(adf["combo"], fontsize=7)
ax.set(xlabel="F1_online", title="All 31 Fusion subsets (M,T,R,P,X) by online F1")
ax.legend(handles=[Patch(color=cmap[k], label=f"{k} feature(s)") for k in [1,2,3,4,5]], loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"ablation_combos.png"); plt.close()

print("wrote figures to", FIG)
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
