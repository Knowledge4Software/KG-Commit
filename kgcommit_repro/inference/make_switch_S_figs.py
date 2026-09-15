#!/usr/bin/env python3
"""
Discussion — sensitivity to the switch point S (Switch@S).

Per-project curves over the S grid, one figure per metric, for both
fusion-selection rules. Cache-only: reads
outputs/<p>/final_final_run/switch/switch_results.json. No Neo4j, no rebuild.

The S grid has two meaningful endpoints, which are what make the curves
interpretable:
    S = 0    switch immediately  -> F+G for the whole stream (never use F alone)
    S = inf  never switch        -> F alone for the whole stream
Everything between is a genuine one-time switch. Because "inf" cannot be placed
on a numeric axis, S is plotted against its grid index and the ticks are
labelled with the actual values, with S=inf last.

Out: Paper/paper_material/discussions/Over_Switch_Values/
       figures/fig_switchS_perproject_<rule>_<metric>.{pdf,png}
       figures/fig_switchS_perproject_<rule>_<metric>.csv
       Source_data/<Project>/switch_results.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_projects import ACTIVE as PROJECTS  # noqa: E402

ROOT = HERE.parent.parent
OUTP = ROOT / "outputs"
DEST = ROOT / "Paper" / "paper_material" / "discussions" / "Over_Switch_Values"
FIGD = DEST / "figures"
SRCD = DEST / "Source_data"

RULES = {
    "perproj": ("per_project", "per-project chosen $F$"),
    "overall": ("overall", "overall fixed $F$ = RN+PPR"),
}

# canonical metric -> (display label, higher-is-better)
METRICS = {
    "Macro_F1": ("Macro-F1", True),
    "G_Mean": ("G-Mean", True),
    "AUC": ("AUC", True),
    "Buggy_F1": ("F1 (buggy class)", True),
    "PR_AUC": ("PR-AUC", True),
    "MCC": ("MCC", True),
    "Precision": ("Precision", True),
    "Recall": ("Recall", True),
    "ACC": ("Accuracy", True),
    "Brier": ("Brier score", False),
}

# the JSON carries some near-duplicate keys from different code vintages
ALIASES = {
    "AUC": ("AUC", "ROC_AUC"),
    "Buggy_F1": ("Buggy_F1", "F1_online", "F1"),
    "ACC": ("ACC", "Acc"),
}


def get_metric(block: dict, metric: str):
    for key in ALIASES.get(metric, (metric,)):
        if key in block:
            v = block[key]
            if v is None or v != v:  # NaN
                return None
            return v
    return None


def s_sort_key(s: str) -> float:
    return float("inf") if s == "inf" else float(s)


def s_label(s: str) -> str:
    if s == "inf":
        return r"$\infty$"
    if s == "0":
        return "0"
    return s


def load(folder: str, rule_key: str):
    f = OUTP / folder / "final_final_run" / "switch" / "switch_results.json"
    if not f.exists():
        return None
    node = json.loads(f.read_text()).get(rule_key, {})
    grid = node.get("grid")
    if not grid:
        return None
    return {s: grid[s] for s in sorted(grid, key=s_sort_key)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def figure(rule: str, metric: str, data):
    rule_key, rule_lab = RULES[rule]
    label, higher = METRICS[metric]

    # union of S values across projects, ordered, inf last
    all_s = sorted({s for d in data.values() for s in d}, key=s_sort_key)
    pos = {s: i for i, s in enumerate(all_s)}

    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]

    drawn = 0
    for i, (disp, grid) in enumerate(sorted(data.items())):
        xs, ys = [], []
        for s, block in grid.items():
            v = get_metric(block, metric)
            if v is None:
                continue
            xs.append(pos[s])
            ys.append(v)
            rows.append([disp, s, v])
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, marker=markers[i % len(markers)], markersize=4.5,
                linewidth=1.4, color=cmap(i % 20), label=disp, alpha=0.9)
        drawn += 1

    if not drawn:
        plt.close(fig)
        return None

    # mark the deployed choice S=200 and the two degenerate endpoints
    if "200" in pos:
        ax.axvline(pos["200"], color="#d62728", linestyle="--", linewidth=1.1,
                   alpha=0.7, zorder=0)
        ax.annotate("deployed\nS=200", (pos["200"], 1.005), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5, color="#d62728")

    ax.set_xticks(range(len(all_s)))
    ax.set_xticklabels([s_label(s) for s in all_s], fontsize=8.5)
    ax.set_xlabel("switch point $S$ (evaluated commits after warm-up)\n"
                  r"$S{=}0$: always $F{+}G$      $S{=}\infty$: never switch, $F$ only",
                  fontsize=9)
    ax.set_ylabel(label + ("" if higher else "  (lower is better)"))
    ax.set_title(f"Sensitivity to the switch point $S$: per-project {label}\n"
                 f"({rule_lab}, n={drawn} projects)", fontsize=10.5)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9, loc="best")
    fig.tight_layout()

    FIGD.mkdir(parents=True, exist_ok=True)
    stem = f"fig_switchS_perproject_{rule}_{metric}"
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    with (FIGD / f"{stem}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "S", metric])
        w.writerows(rows)
    return stem


def copy_sources():
    """Copy each project's switch_results.json — the sole input to these figures."""
    SRCD.mkdir(parents=True, exist_ok=True)
    entries = []
    for disp, folder in PROJECTS:
        src = OUTP / folder / "final_final_run" / "switch" / "switch_results.json"
        if not src.exists():
            continue
        out = SRCD / disp
        out.mkdir(exist_ok=True)
        tgt = out / "switch_results.json"
        shutil.copy2(src, tgt)
        assert sha(src) == sha(tgt), f"copy mismatch for {disp}"
        entries.append({
            "project": disp,
            "from": f"outputs/{folder}/final_final_run/switch/switch_results.json",
            "to": f"Source_data/{disp}/switch_results.json",
            "bytes": src.stat().st_size,
            "sha256": sha(src),
        })
    shutil.copy2(Path(__file__), SRCD / Path(__file__).name)
    return entries


def main():
    made, skipped = [], []
    per_rule = {}
    for rule, (rule_key, _lab) in RULES.items():
        data = {}
        for disp, folder in PROJECTS:
            grid = load(folder, rule_key)
            if grid:
                data[disp] = grid
        per_rule[rule] = data
        print(f"{rule:<8} projects with an S grid: {len(data)}")
        for metric in METRICS:
            stem = figure(rule, metric, data)
            (made if stem else skipped).append(stem or f"{rule}/{metric}")

    entries = copy_sources()

    prov = {
        "figures_from": "outputs/<project>/final_final_run/switch/"
                        "switch_results.json  ->  <rule>.grid[S]",
        "requires_neo4j": False,
        "run": "python kgcommit_repro/inference/make_switch_S_figs.py",
        "rules": {k: v[1] for k, v in RULES.items()},
        "metrics": {k: v[0] for k, v in METRICS.items()},
        "grid_endpoints": {
            "0": "switch immediately -> F+G for the whole stream",
            "inf": "never switch -> F alone for the whole stream",
        },
        "deployed_S": 200,
        "projects_per_rule": {r: sorted(d) for r, d in per_rule.items()},
        "s_grid_per_project": {
            disp: sorted(per_rule["perproj"][disp], key=s_sort_key)
            for disp in per_rule["perproj"]
        },
        "figures": sorted(made),
        "source_files": entries,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "MANIFEST.json").write_text(json.dumps(prov, indent=2))

    print(f"\n{len(made)} figures -> {FIGD}")
    print(f"{len(entries)} source files -> {SRCD}")
    if skipped:
        print("not drawn:", ", ".join(skipped))


if __name__ == "__main__":
    main()
