"""
Visual comparison of the fully-online (prequential) inference methods on
ONLINE BUGGY-F1 (and PR-AUC), grouped into three families:

    * Within-project BASELINES      (JIT metrics only)
    * KG-v1 inference methods       (KG signals, NO commit-text)
    * KG-v2 inference methods       (+ commit-text stream X / recommended subset)

Single methods + baselines come from the streaming engine cache
(outputs/online_jit_results.pkl); the Fusion variants come from the unified
feature-ablation cache (outputs/online_jit_ablation_v2.pkl) so that the
V1 (M+T+R+P) vs V2 (M+T+R+P+X, M+T+R) fusion deltas are produced by ONE engine
and are directly comparable.

Run:  python docs/make_v1_v2_baseline_compare.py
Out:  docs/figures/v1_v2_baseline_f1.png
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "outputs"
FIG  = ROOT / "docs" / "figures" / "v1_v2_baseline_f1.png"

KGBLUE, KGGREEN, KGORANGE = "#1F77B4", "#2CA02C", "#FF7F0E"

res = pickle.load(open(OUT / "online_jit_results.pkl", "rb"))
abl = pickle.load(open(OUT / "online_jit_ablation_v2.pkl", "rb"))
M = {v["name"]: v["cum"] for v in res["methods"].values()}

def g(name):           # streaming-engine method -> (F1_online, PR_AUC)
    c = M[name];  return c["F1_online"], c["PR_AUC"]
def a(label):          # ablation subset -> (F1_online, PR_AUC)
    c = abl[label]["cum"];  return c["F1_online"], c["PR_AUC"]

# (display label, (F1_online, PR_AUC), group)  -- group in {base, v1, v2}
ROWS = [
    ("Naive prior bug-rate",            g("Naive prior bug-rate"),               "base"),
    ("LR / JIT metrics",                g("LR / JIT metrics (incremental)"),     "base"),
    ("RandomForest / JIT metrics",      g("RandomForest / JIT metrics"),         "base"),
    ("GradBoost / JIT metrics",         g("GradBoost / JIT metrics"),            "base"),
    ("Relational priors (wvRN)",        g("Relational priors (wvRN)"),           "v1"),
    ("KG embedding (SVD/LSA)",          g("KG embedding (SVD/LSA)"),             "v1"),
    ("Structural TF-IDF",               g("Structural TF-IDF (incremental)"),    "v1"),
    ("Personalized PageRank",           g("Personalized PageRank"),              "v1"),
    ("Fusion  M+T+R+P  (v1)",           a("M+T+R+P"),                            "v1"),
    ("Commit-text  X  alone",           a("X"),                                  "v2"),
    ("Fusion+text  M+T+R+P+X",          a("M+T+R+P+X"),                          "v2"),
    ("Fusion  M+T+R  (recommended)",    a("M+T+R"),                              "v2"),
]
COLOR = {"base": KGORANGE, "v1": KGBLUE, "v2": KGGREEN}
LEGEND = {"base": "Within-project baselines (JIT metrics)",
          "v1":   "KG-v1 inference (no commit-text)",
          "v2":   "KG-v2 inference (+ commit-text)"}

# stack groups bottom->top: baselines, then v1, then v2 (so v2 is on top)
order = [r for r in ROWS if r[2] == "base"] + \
        [r for r in ROWS if r[2] == "v1"] + \
        [r for r in ROWS if r[2] == "v2"]
labels   = [r[0] for r in order]
f1       = np.array([r[1][0] for r in order])
pr       = np.array([r[1][1] for r in order])
colors   = [COLOR[r[2]] for r in order]
y        = np.arange(len(order))

best_base_f1 = max(r[1][0] for r in ROWS if r[2] == "base")
best_base_pr = max(r[1][1] for r in ROWS if r[2] == "base")
stream_bug   = res["meta"]["stream_bug"]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
for ax, vals, title, ref, refname in [
    (axes[0], f1, "Online buggy-F1  (F1$_{\\mathrm{online}}$)", best_base_f1, "best baseline"),
    (axes[1], pr, "PR-AUC (prequential)",                       best_base_pr, "best baseline"),
]:
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.74)
    ax.axvline(ref, ls="--", lw=1.3, color="0.35", zorder=0)
    ax.text(ref, len(order)-0.3, f" {refname} {ref:.3f}", color="0.35",
            fontsize=8, va="center", ha="left")
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.5); ax.set_axisbelow(True)

axes[1].axvline(stream_bug, ls=":", lw=1.2, color=KGORANGE, zorder=0)
axes[1].text(stream_bug, -0.9, f"random {stream_bug:.3f}", color=KGORANGE,
             fontsize=8, ha="center", va="top")
axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=9.5)
axes[0].invert_yaxis()
handles = [Patch(facecolor=COLOR[k], label=LEGEND[k]) for k in ("base", "v1", "v2")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Inference methods under one fully-online protocol: baselines vs. KG-v1 vs. KG-v2"
             f"  (stream bug-rate {stream_bug:.3f})", fontsize=12.5)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, bbox_inches="tight")
print("wrote", FIG)
