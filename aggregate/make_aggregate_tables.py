"""
Render the cross-project aggregate results (aggregate_results.pkl) into
human-readable LaTeX + CSV tables under outputs/aggregate/tables/.

Produces, for the headline metrics:
  * tab_deployed_fusion   deployed F+G fusion: Type-1 macro vs Type-2 totals
  * tab_methods_final     5 methods on the 'final' graph (macro / total)
  * tab_fusion_combos     F / F+G / F+M / F+G+M aggregated
  * tab_subgraph_variants 6 subgraph variants (Fusion leaf) aggregated
  * tab_layer_stats       summed KG size per layer across projects
  * tab_per_project        per-project deployed-fusion headline (for context)

Run:  python aggregate/make_aggregate_tables.py
"""
import pickle
import csv
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PKG_ROOT.parent / "outputs"
AGG = OUTPUTS / "aggregate"
TAB = AGG / "tables"

HEADLINE = ["Buggy_F1", "Macro_F1", "G_Mean", "AUC", "PR_AUC", "MCC"]


def _cell(c, key):
    if not c or c.get(key) is None:
        return "--"
    return f"{c[key]:.3f}"


def _write_csv(name, header, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def _write_tex(name, header, rows, caption):
    TAB.mkdir(parents=True, exist_ok=True)
    cols = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{{caption}}}", f"\\label{{tab:agg-{name}}}",
             f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(h.replace("_", r"\_") for h in header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(x) for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def table_metric_block(agg_leaf_by_row, row_labels, name, caption):
    """agg_leaf_by_row: list of dict(metric->combine) aligned with row_labels.
    Emits macro + total_N columns per headline metric."""
    header = ["row"] + [f"{m} (macro)" for m in HEADLINE] + [f"{m} (total)" for m in HEADLINE]
    rows = []
    for lbl, leaf in zip(row_labels, agg_leaf_by_row):
        macro = [_cell(leaf.get(m), "macro") for m in HEADLINE]
        total = [_cell(leaf.get(m), "total_N") for m in HEADLINE]
        rows.append([lbl] + macro + total)
    _write_csv(name, header, rows)
    _write_tex(name, header, rows, caption)


def main():
    r = pickle.load(open(AGG / "aggregate_results.pkl", "rb"))
    projs = r["meta"]["projects"]; np_ = r["meta"]["n_projects"]
    print(f"tables -> {TAB}  ({np_} projects: {', '.join(projs)})")

    # 1. deployed fusion (single row)
    if r.get("fusion"):
        table_metric_block([r["fusion"]["deployed_F"]],
                           [f"Deployed {r['fusion']['deployed_label']}"],
                           "tab_deployed_fusion",
                           f"Cross-project deployed fusion ({r['fusion']['deployed_label']}): "
                           f"Type-1 macro mean vs Type-2 commit-weighted total "
                           f"over {np_} projects.")
        # 3. fusion combos
        combos = r["fusion"]["combos"]
        order = [c for c in ["F", "F+G", "F+M", "F+G+M"] if c in combos]
        table_metric_block([combos[c] for c in order], order, "tab_fusion_combos",
                           f"Fusion-combination ablation aggregated over {np_} projects "
                           f"(macro vs total).")

    # 2. methods on final graph
    if r.get("final_experiments"):
        fe = r["final_experiments"]
        g = "final" if "final" in fe["graphs"] else fe["graphs"][-1]
        table_metric_block([fe["aggregated"][g][m] for m in fe["methods"]],
                           fe["methods"], "tab_methods_final",
                           f"Graph-inference methods on the '{g}' graph, aggregated over "
                           f"{np_} projects (macro vs total).")

    # 4. subgraph variants (Fusion leaf)
    if r.get("subgraph_rq"):
        sq = r["subgraph_rq"]
        table_metric_block([sq["aggregated"][v]["Fusion"] for v in sq["variants"]],
                           sq["variants"], "tab_subgraph_variants",
                           f"Subgraph-variant Fusion, aggregated over {np_} projects "
                           f"(macro vs total).")

    # 5. layer stats (summed)
    if r.get("layer_stats"):
        ls = r["layer_stats"]; summed = ls["summed"]
        keys = ["nodes", "delta_total", "ADDS", "REMOVES", "UPDATES", "MOVES",
                "commits_with_tokens", "n_token_types"]
        header = ["layer"] + keys
        rows = [[lyr] + [summed.get(lyr, {}).get(k, "--") for k in keys]
                for lyr in ls["layers"]]
        _write_csv("tab_layer_stats", header, rows)
        _write_tex("tab_layer_stats", header, rows,
                   f"KG size summed across {np_} projects, per structural layer.")

    # 7. baselines (if present)
    if r.get("baselines"):
        bl = r["baselines"]
        table_metric_block([bl["aggregated"][b] for b in bl["baselines"]],
                           bl["baselines"], "tab_baselines",
                           f"Within-project JIT baselines aggregated over the "
                           f"projects that have them (macro vs total).")

    # 8. effort-aware (if present)
    if r.get("effort"):
        ef = r["effort"]
        header = ["model"] + [f"{m} (macro)" for m in ["Popt", "ACC20", "Buggy_F1", "PR_AUC"]]
        rows = [[m] + [_cell(ef["aggregated"][m].get(k), "macro")
                       for k in ["Popt", "ACC20", "Buggy_F1", "PR_AUC"]]
                for m in ef["models"]]
        _write_csv("tab_effort", header, rows)
        _write_tex("tab_effort", header, rows,
                   "Effort-aware evaluation (Popt / ACC@20%LOC) of KG channels, "
                   "aggregated (macro).")

    # 9. scalability roll-up (if present)
    if r.get("scalability"):
        sc = r["scalability"]
        bc = sc.get("build_ms_per_file", {})
        if bc:
            header = ["layer", "build ms/file (macro)", "build ms/file (total-N)"]
            rows = [[lyr, _cell(bc[lyr], "macro"), _cell(bc[lyr], "total_N")]
                    for lyr in bc]
            _write_csv("tab_scal_build", header, rows)
            _write_tex("tab_scal_build", header, rows,
                       "Per-layer build cost (ms/file) aggregated across projects.")

    # 6. per-project deployed fusion (context)
    if r.get("fusion"):
        idx = r["projects_index"]
        header = ["project", "N", "n_eval", "chosen_fusion"]
        rows = []
        for p in projs:
            ip = idx.get(p, {})
            rows.append([p, ip.get("N", "--"), ip.get("n_eval", "--"),
                         str(r["fusion"]["chosen_per_project"].get(p, "--"))])
        _write_csv("tab_per_project", header, rows)
        _write_tex("tab_per_project", header, rows,
                   "Per-project commit counts and chosen fusion composition.")

    print("done. tables written:")
    for f in sorted(TAB.glob("*.tex")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
