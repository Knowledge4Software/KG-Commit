# KG-Commit — RUNBOOK (per project, light-first)

Fully-annotated command sequence to reproduce the pipeline on one project. Commands
are shown for **bash**; for **PowerShell** replace `export KGC_PROJECT=zookeeper` with
`$env:KGC_PROJECT = 'zookeeper'`.

Recommended order by size (labelled-commit count): **zookeeper (839) → zeppelin (1,451)
→ kafka (2,384) → hadoop-mapreduce (1,321) → hadoop-hdfs (2,907) → activemq (6,126) →
groovy (8,059) → cassandra (8,159) → hbase (8,730) → flink (11,691) → ignite (12,036)**.

---

## 0. One-time setup

```bash
pip install -r requirements.txt
# start Neo4j 5.x at bolt://localhost:7687 (auth neo4j/password1234, or edit projects.yaml)
```

Run every command from the **package root** (`kgcommit_repro/`).

---

## 1. Select the project

```bash
export KGC_PROJECT=zookeeper          # PowerShell: $env:KGC_PROJECT='zookeeper'
python config/project_config.py       # sanity: prints resolved paths; check they exist
```

The last command should show `REPO_PATH (exists=True)`, `CSV_PATH (exists=True)`,
`DIFF_CSV (exists=True)`. If any is False, clone the repo / obtain the CSV first (§ README 3).

---

## 2. Pick the base commit  *(new project only; groovy already has one)*

```bash
python find_base_commit.py
# -> prints a suggested SHA (first commit with >= 20 .java files). Paste it into
#    config/projects.yaml under the project's `base_commit:` field.
```

---

## 3. Preserve the current graph, then make room for this project

Neo4j Community holds **one project's graph at a time**, but you never have to
rebuild: snapshot the resident graph first, then wipe. Snapshots restore in minutes.

```bash
# (a) if a DIFFERENT project is currently resident, save it first (skippable if
#     you already have its .dump, or the DB is empty):
KGC_PROJECT=<resident-project> python snapshot_neo4j.py dump   # -> outputs/<p>/neo4j_dump/<p>.dump

# (b) wipe the DB so this project can be built (store-level wipe; handles huge graphs):
KGC_PROJECT=<this-project> python reset_neo4j.py --yes
```

**To return to a previously-built project WITHOUT rebuilding** (the whole point of
snapshots):
```bash
KGC_PROJECT=groovy python snapshot_neo4j.py restore   # loads groovy.dump back in minutes
```
`snapshot_neo4j.py list` shows all saved snapshots. The dump/restore stops the Neo4j
container briefly (Community edition cannot dump a single live DB) and uses a
transient `neo4j-admin` container over the same data volume; configure via
`KGC_NEO4J_CONTAINER` / `KGC_NEO4J_DATADIR` if your Neo4j is not the `neo4j-local`
Docker container.

---

## 4. Build the knowledge graph (Phase A → B → C)

```bash
python drivers/build_project.py
#   --limit 50     # optional: cap commits for a fast smoke test
#   --skip-cstg    # optional: build Core+AST+subgraphs only
#   --from b5      # optional: resume from a step (a1..c9)
```

This runs, in order, writing **per-commit timing logs** to `logs/<project>/`:

| Step | Script | Writes |
|---|---|---|
| A1 | `build/build_file_ast_graphs.py` | `outputs/<p>/file_ast_graphs/*.pkl` |
| A2 | `build/ingest_base_kg_and_ast.py` | Neo4j Core layer + base AST (`:Commit/:File/:Developer/:Issue`, `HAS_AST`, `AST_CHILD`) |
| A3 | `build/add_positions_to_astnodes.py` | `pos_line/pos_col` on base `:ASTNode`s |
| A4 | `build/apply_commit_labels.py` | `buggy`, JIT metrics, `author_ts`, `in_jit` on `:Commit`s |
| B5 | `build/build_online_kg.py` | AST delta edges `ADDS/REMOVES/UPDATES/MOVES`; `logs/<p>/ast_timing.csv` |
| B6 | `build/build_subgraph_online_kg.py --kind {cfg,dfg,pdg,seq}` | `:CFGNode/:DFGNode/:PDGNode/:SEQNode` layers; `logs/<p>/{kind}_timing.csv` |
| C7 | `inference/validate_cstg.py` | `outputs/<p>/cstg_bundle.pkl` |
| C8 | `inference/ingest_cstg.py --ground` | `:Term`, `MENTIONS`, `COOCCURS`, `GROUNDS_IN` |
| C9 | `inference/ingest_cstg_enrich.py` | `:Intent`, `HAS_INTENT` |

To run a single step by hand (all inherit `KGC_PROJECT`):
```bash
python build/build_online_kg.py --reset --timing-log logs/zookeeper/ast_timing.csv
python build/build_subgraph_online_kg.py --kind cfg --reset --timing-log logs/zookeeper/cfg_timing.csv
```

---

## 5. Run the final inference experiments (needs the live graph)

```bash
python drivers/run_experiments.py
```
Produces under `outputs/<p>/`: `final_experiments_results.pkl`,
`final_fusion_results.pkl`, `subgraph_rq_results.pkl`,
`subgraph_kg_methods_results.pkl`, `subgraph_layer_stats.json`, plus
`effort_results.pkl` (Popt / ACC@20%LOC of the KG channels) and
`baseline_results.pkl` (LR/RF/HGB + naive JIT baselines) — the comparison point.
The last two, and the parameter sweeps below, run from the **cached** feature
streams, so they need no live Neo4j and can be re-run any time:

```bash
python inference/run_effort_eval.py        # Popt / ACC@20%LOC (cache-only)
python baselines/run_baselines.py          # within-project JIT baselines (label CSV only)
python inference/run_param_experiments.py  # WARMUP / BLOCK / ROLL sweeps (cache-only)
```

Neo4j-dependent extra (run only when THIS project's graph is resident):
```bash
python inference/collect_ast_structure_stats.py   # node-type/change-group/depth dists
```

---

## 6. Run the scalability / complexity / significance suite

```bash
python drivers/run_scalability.py            # add --quick for faster sampling
```
Produces under `outputs/<p>/scalability/`: `kg_profile.json`, `growth.json`,
`build_complexity.json`, `prediction_latency.json`, `significance.json`,
`cstg_ablation.json`, `cstg_graph_ablation.json` (+ their `.tex` tables). The
build-complexity step reads the per-commit timing logs from `logs/<p>/`.

---

## 7. Render all tables, figures, and notebooks

```bash
python drivers/make_reports.py               # add --no-notebooks to skip nbconvert
```
Produces under `outputs/<p>/`: `tables/v4/*.tex`, `figures/v4/**/*.{png,pdf}`, and the
four executed notebooks under `outputs/<p>/notebooks/`.

---

## 8. Move to the next project (without losing the current one)

```bash
# save the project you just finished so you can restore it later in minutes:
KGC_PROJECT=zookeeper python snapshot_neo4j.py dump

export KGC_PROJECT=zeppelin
python find_base_commit.py            # paste SHA into projects.yaml (new project)
python reset_neo4j.py --yes           # wipe (zookeeper is safe in its .dump)
python drivers/build_project.py
python drivers/run_experiments.py
python drivers/run_scalability.py
python drivers/make_reports.py
KGC_PROJECT=zeppelin python snapshot_neo4j.py dump   # save zeppelin too
```

`outputs/zookeeper/` and `outputs/zeppelin/` coexist; nothing is overwritten. To put
either project's graph back in Neo4j later, `snapshot_neo4j.py restore` it — no rebuild.

---

## 9. Cross-project (general) evaluation — aggregate over ALL built projects

After building any number of projects, aggregate their results into one
**general evaluation** across all of them. This needs **no `KGC_PROJECT`** and
touches **no Neo4j** — it only reads the saved `outputs/<project>/*.pkl` files,
skipping any project that lacks them.

```bash
python drivers/run_aggregate.py                 # all projects with results
python drivers/run_aggregate.py --only activemq kafka   # restrict the set
```

Produces under `outputs/aggregate/`:
`aggregate_results.pkl` (everything, machine-readable), `aggregate_summary.json`
(headline), `projects_index.json` (which projects/files were used),
`tables/*.{tex,csv}`, `figures/*.{png,pdf}`, and the executed
`notebooks/aggregate_evaluation.ipynb`. It aggregates the ML metrics (5 methods ×
6 graphs, fusion combos, subgraph variants), and — where present per project — the
**baselines**, **effort-aware** metrics, and the **scalability** roll-up (build/
predict cost), plus a genuinely-pooled fusion metric once projects carry
`raw_stream`.

Two aggregation types per metric (see the notebook for the worked example):
- **Type 1 — Macro**: unweighted mean across projects (each project counts equally).
- **Type 2 — Total**: commit-weighted mean, reported for `w=n_eval` (scored
  commits `N-warmup`) and `w=N` (all labelled commits).

Re-run it after each new project to refresh the aggregate with that project folded
in — it is idempotent and picks up whatever result folders currently exist.

---

## Watching a long build

The AST layer of a large project can take hours. To see live progress at any time
(read-only, safe against a running build):

```bash
KGC_PROJECT=<project> python check_progress.py            # one snapshot
KGC_PROJECT=<project> python check_progress.py --watch    # refresh every 15s
```

It reports, per online-growth layer, commits-done / total with a progress bar,
a live commit-rate and ETA (sampled from the per-commit timing CSV), the running
wall-time for that layer, the Neo4j node/edge tallies landed so far, and which
pipeline artifacts already exist under `outputs/<project>/`. Add `--no-neo4j` to
skip the database round-trip, or `--interval N` to change the watch cadence.

## Troubleshooting

- **`KGC_PROJECT is not set`** — export it (§1). The package refuses to guess a project.
- **`... missing required field 'base_commit'`** — run `find_base_commit.py` and paste the
  SHA into `projects.yaml` (§2).
- **`REPO_PATH (exists=False)`** — clone the repo to `repos/apache/<project>/`.
- **Empty/duplicated results** — you likely built a new project without
  `reset_neo4j.py --yes`; wipe and rebuild.
- **Neo4j auth/connection errors** — check the DB is up and the creds in
  `config/projects.yaml → neo4j:` match.
