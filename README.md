# KG-Commit — Reproducibility Package

Code repository for the manuscript **“KG-Commit: A Dynamic Knowledge Graph for Online
Just-in-Time Software Defect Prediction”**, submitted to *Knowledge-Based Systems*
(submission **KNOSYS-D-26-21009**, under review).

> KG-Commit is a dynamic knowledge graph that incrementally maintains repository
> history, within-file code structure, and commit semantics as a project evolves. It
> tracks structural change between file edits with an AST-delta mechanism and predicts
> defect-prone commits using lightweight graph inference that runs **entirely on CPU**.

This repository is the **self-contained, publishable code artifact** for that paper. It
contains the final methodology only — the three graph layers, the inference channels and
their fusion, the six baselines, and the scalability analysis — and runs the pipeline
end-to-end (cloned repo + ApacheJIT data → Neo4j knowledge graph → all experiments →
tables, figures and notebooks) on **one Apache project at a time**.

> **One project at a time, on purpose.** The pipeline is *not* a loop over projects.
> You pick a project with the `KGC_PROJECT` environment variable, build and analyse it,
> read the results, then move to the next. Start with the **light** projects
> (zookeeper, zeppelin) and work up.

---

## 1. What is (and isn't) in here

KG-Commit organises the repository into **three layers** (paper §3.3), all grown
*online*, one commit at a time:

| Layer | Contents | Built by |
|---|---|---|
| **1 — Core** | repository entities: commits, files, developers, issues | `kg_commit/`, `build/ingest_base_kg_and_ast.py` |
| **2 — Within-file** | the four reported representations — AST, CFG, DFG, PDG — maintained by the **AST-delta** mechanism | `build_ast_features.py`, `build_cfg_features.py`, `build_dfg_features.py`, `build_cpg_features.py` (PDG), `online_ast_diff.py`, `subgraph_*.py` |
| **3 — CSTG** | Commit Semantic-Text Graph: message + diff term graph, intents, `GROUNDS_IN` bridge to the AST layer | `inference/cstg*.py`, `inference/ingest_cstg*.py` |

**In (final methodology only):**
- `build/` — the ordered knowledge-graph build steps (base snapshot, Core layer, AST
  delta layer, and the CFG/DFG/PDG representations) plus the parsers and differs.
- `inference/` — the five graph-inference channels (**RN, PPR, LP, DW, KGE**), the
  fusion, the feature-based **CSTG (G)** channel, the online evaluation protocol, and
  the renderers for the paper's tables and figures.
- `scalability/` — KG statistical profile, per-commit growth, build/update complexity,
  prediction latency, statistical-significance tests, and the CSTG ablations.
- `baselines/` — the six baselines the paper compares against (§9).
- `kg_commit/` — the library that builds the Core layer into Neo4j.
- `config/` — the per-project registry and the single source of truth for all paths.
- `drivers/` — six thin orchestrators that run the pipeline phase by phase.
- `find_base_commit.py`, `reset_neo4j.py`, `snapshot_neo4j.py`, `check_progress.py` —
  setup, teardown, snapshot/restore and live progress helpers.

**Out:** superseded earlier experimentation, legacy builders, audits, and one-off
exploratory scripts from the research repo. What remains is exactly the code reachable
from the drivers — i.e. the pipeline the paper reports.

> **A note on `seq`.** The build also constructs a token-sequence (`seq`) layer
> alongside CFG/DFG/PDG. It is retained because the build step is generic over layer
> kinds, but it is **not part of the reported methodology** — the paper's Layer 2
> results cover AST, CFG, DFG and PDG only. Pass `--skip-cstg` / select kinds if you
> want to shorten a build; nothing downstream in the paper depends on `seq`.

**Read in place, never copied:** the cloned repositories under `repos/apache/<project>/`
and the ApacheJIT CSVs under `data/apachejit/projects/`. This package references them by
path (relative to the parent repo root); it does not duplicate the multi-gigabyte data.
See §5 for exactly where each input is read from.

---

## 2. Layout

```
kg-commit/             (repository root)
  README.md            this file
  RUNBOOK.md           copy-paste command sequence, per project
  requirements.txt     Python dependencies
  _kgc_paths.py        import bootstrap (puts the package dirs on sys.path)
  config/
    project_config.py  SINGLE source of truth: resolves per-project paths from
                       projects.yaml + the KGC_PROJECT env var
    projects.template.yaml  template: copy to projects.yaml and fill in your
                       local repo/CSV paths (projects.yaml is machine-specific
                       and therefore not committed)
  find_base_commit.py  suggest a project's base commit (first substantial Java commit)
  reset_neo4j.py       full DB wipe (run before switching projects)
  snapshot_neo4j.py    dump / restore / list a project's graph (skip rebuilds)
  check_progress.py    live progress of a running build (read-only)
  build/               the 10 build-step scripts + parsers/differs (Phase A/B/C)
  inference/           inference channels, fusion, CSTG channel, protocol, renderers
  scalability/         RQ2 suite: profile, growth, complexity, latency, significance
  baselines/           the paper's six baselines (§9)
  aggregate/           cross-project roll-up over everything already built
  kg_commit/           Core-layer builder library
  drivers/
    build_project.py   Phase A→B→C for $KGC_PROJECT (with per-commit timing logs)
    run_experiments.py final inference experiments (live Neo4j)
    run_final.py       the full final run: experiments + baselines + sweeps
    run_scalability.py scalability / complexity / significance suite
    run_aggregate.py   cross-project aggregate (no Neo4j, no KGC_PROJECT)
    make_reports.py    all tables, figures, executed notebooks
  notebooks/           the four final notebooks, generated per project by
                       `drivers/make_reports.py`
  logs/<project>/      per-commit build TIMING CSVs + build stdout (for the
                       scalability RQ), created by `drivers/build_project.py`
```

> `notebooks/`, `logs/` and `outputs/` are **generated** by the pipeline and are not
> part of the repository; they appear once you run the drivers.

Generated artifacts are **namespaced per project** so builds never clobber each other
(`PROJECT_ROOT` is the clone's parent — see §5):
```
PROJECT_ROOT/outputs/<project>/                caches, result pickles, checkpoints
PROJECT_ROOT/outputs/<project>/scalability/    scalability + CSTG-ablation JSON/TeX
PROJECT_ROOT/outputs/<project>/figures/v4/…    figures
PROJECT_ROOT/outputs/<project>/tables/v4/…     LaTeX tables
PROJECT_ROOT/outputs/<project>/notebooks/…     executed notebooks
PROJECT_ROOT/outputs/<project>/neo4j_dump/…    graph snapshots (see §10)
<clone>/logs/<project>/                        per-commit build timing CSVs
```

---

## 3. Prerequisites

1. **Python 3.10+** and `pip install -r requirements.txt`.
2. **Your local configuration.** `config/projects.yaml` holds machine-specific paths and
   is therefore not committed — create it once before the first run:
   ```bash
   cp config/projects.template.yaml config/projects.yaml
   # then edit: repo paths, ApacheJIT CSV paths, base_commit, and the neo4j: block
   python config/project_config.py      # verifies every resolved path exists
   ```
3. **Neo4j 5.x** running at `bolt://localhost:7687` (default auth `neo4j` / `password1234`;
   change under `config/projects.yaml → neo4j:`). The graph store is **single-tenant** —
   it holds exactly one project's graph at a time (see §4).
4. **git** on `PATH` (the build reads the cloned repo via `git show` / `git diff`).
5. For each project you want to run: a **cloned repo** at `repos/apache/<project>/` and the
   **ApacheJIT CSVs** at `data/apachejit/projects/apache_<project>.csv` (labels) and
   `apache_<project>_diff.csv` (diff text, for the CSTG layer). The paper evaluates the
   **11 projects** of Table 4: `activemq`, `camel`, `cassandra`, `flink`, `groovy`,
   `hbase`, `hive`, `kafka`, `spark`, `zeppelin`, `zookeeper`. Register each one you
   intend to run in your `config/projects.yaml` (§3, item 2).

---

## 4. The single most important operational fact

**Neo4j is single-tenant.** Nodes are not project-tagged and the online build streams
`MATCH (c:Commit {in_jit:true})` globally, so the database holds **one project's
knowledge graph at a time**. To switch from project A to project B you must first wipe
A entirely:

```
KGC_PROJECT=<B>  python reset_neo4j.py --yes
```

Your per-project on-disk artifacts under `outputs/<A>/` are *not* touched by the wipe —
they stay for later comparison. Only the graph store is reset.

---

## 5. Where each input is read from (in place)

Large inputs are **read in place and never copied or committed**. `config/project_config.py`
resolves them against a `PROJECT_ROOT` defined as **the parent directory of this clone**
(`PKG_ROOT.parent`). So the data directories and the generated `outputs/` sit *next to* the
cloned repository, not inside it:

```
<workspace>/                     <-- PROJECT_ROOT
├── kg-commit/                   <-- this repository (clone it here)
│   ├── config/  build/  inference/  scalability/  drivers/ ...
│   └── logs/<project>/          per-commit build timing CSVs
├── repos/apache/<project>/      cloned Apache project under study
├── data/apachejit/projects/     ApacheJIT CSVs
└── outputs/<project>/           all generated caches, tables, figures, notebooks
```

| Input | Path (relative to `PROJECT_ROOT`) | Read by |
|---|---|---|
| Cloned source repo | `repos/apache/<project>/` | build engines (`git show`/`git diff`) |
| ApacheJIT labels + JIT metrics | `data/apachejit/projects/apache_<project>.csv` | `apply_commit_labels.py`, config |
| Diff text (CSTG) | `data/apachejit/projects/apache_<project>_diff.csv` | `validate_cstg.py`, `ingest_cstg.py`, `cstg_online_features.py` |

The exact paths are configurable per project in `config/projects.yaml` (the
`repo_subpath`, `label_csv` and `diff_csv` fields), so you can point them elsewhere if you
prefer a different layout. Always confirm the resolution before a run:

```bash
python config/project_config.py     # prints every resolved path + exists=True/False
```

---

## 6. Quick start (light project first)

```bash
# 1. select the project (bash; PowerShell: $env:KGC_PROJECT='zookeeper')
export KGC_PROJECT=zookeeper

# 2. NEW project only: pick its base commit, paste the printed SHA into projects.yaml
python find_base_commit.py

# 3. if a DIFFERENT project is currently in Neo4j, wipe it first
python reset_neo4j.py --yes

# 4. build the whole graph (Phase A→B→C) with per-commit timing logs
python drivers/build_project.py            # add --limit 50 for a smoke test

# 5. run the final inference experiments (needs the live graph)
python drivers/run_experiments.py

# 6. run the scalability / complexity / significance suite
python drivers/run_scalability.py

# 7. render all tables, figures, and executed notebooks
python drivers/make_reports.py
```

Everything for `zookeeper` now lives under `outputs/zookeeper/`. To do the next project,
repeat from step 1 with a new `KGC_PROJECT`, remembering the `reset_neo4j.py` in step 3.

See **RUNBOOK.md** for the fully-annotated command sequence, per-step outputs, and the
individual build steps if you want to run them one at a time.

---

## 7. Notes for the reader / reviewer

- **Fully online, leakage-free.** Every commit is predicted from a model and graph state
  built only from strictly earlier commits (prequential predict-then-grow). The protocol
  constants are defined **once**, in `inference/protocol.py`, and are identical for
  KG-Commit and every baseline:

  | Constant | Value | Meaning |
  |---|---|---|
  | `WARMUP_FRAC` (K) | `0.05` | initial-fit fraction; commits `0..W` are not scored |
  | `GAP` (g) | `50` | commits skipped between fitting and scoring (label latency) |
  | `BLOCK` (M) | `200` | the single learn/refresh granularity |
  | `REFIT_EVERY` | `1` | every component refits once per block |

  Do not hardcode these anywhere else — every module that learns or refreshes on a
  schedule imports them from `protocol.py`.
- **Deployed model.** The reported KG-Commit configuration is **Fov + G + T**:
  - `Fov = RN + PPR` — the fixed graph fusion (rationale in paper §5.4),
  - `G` — the CSTG semantic-text channel,
  - `T` — the traversal / global-context (TGC) feature block.

  Because `G` must accumulate a vocabulary, it is unreliable at cold start — it helps on
  the largest projects and hurts on the smallest. The **S@200 policy** resolves this:
  `Fov` alone scores the first 200 evaluated commits after warm-up, then the model
  switches **once** to include `G` for the rest of the stream. Adding `T` does not change
  that switch, and S@200 is *distinct* from the periodic refit interval `M = 200`. Tables
  and figures abbreviate the configuration as `F+G+T`, with `F = Fov`. The switch point
  is selected by `inference/run_switch_experiment.py`, which sweeps the grid of candidate
  switch points and reports the per-project outcome.
- **Timing logs are first-class.** `build_project.py` writes one row per commit
  (`idx, sha, files, adds/removes/updates/moves, wall_ms`) to `logs/<project>/*_timing.csv`
  for the AST and each subgraph layer. These feed the build/growth complexity analysis
  (the paper's second research question); do not delete them.
- **DB-free reruns.** After the first build+experiments, the heavy feature streams and
  results are cached under `outputs/<project>/`, so the renderers and most of the
  scalability suite run without a live database.
- **Adding a new project X.** Clone it to `repos/apache/X/`, ensure its ApacheJIT CSVs
  exist, copy the block from `config/projects.template.yaml` into `projects.yaml`, pick its
  base commit with `find_base_commit.py`, and follow §6. No code changes are needed.
- **Scope of this repository.** It contains the final methodology only: the graph build,
  the five inference channels, the fusion, the CSTG channel, the evaluation protocol, the
  baselines and the scalability suite — i.e. the code paths reachable from the drivers.
  Superseded earlier experimentation and one-off exploratory scripts are deliberately
  excluded so that what is published is exactly what the paper reports.

---

## 8. Reproducing each research question

The paper asks four research questions (§4.1). This is where each is implemented:

| RQ | Question | Run | Produces |
|---|---|---|---|
| **RQ1** | Predictive performance vs. baselines | `drivers/run_experiments.py`, then `baselines/run_baselines.py` + `run_extra_baselines.py` | `final_experiments_results.pkl`, `final_fusion_results.pkl`, `baseline_results.pkl` |
| **RQ2** | Efficiency and scalability over project evolution | `drivers/run_scalability.py` | KG profile, per-commit growth, build/update complexity, predict latency, significance tests |
| **RQ3** | Effect of within-file representations and layers | `inference/run_subgraph_rq.py`, `inference/collect_subgraph_stats.py` (via `run_experiments.py`) | `subgraph_rq_results.pkl`, `subgraph_layer_stats.json` |
| **RQ4** | Inference channels and fusion | `inference/run_final_experiments.py`, `inference/run_final_fusion.py` | five channels × six graphs, all fusion combinations |

The headline **Fov + G + T** configuration additionally needs the traversal/global-context
block. It is a two-stage design — a one-off Neo4j extraction of the tier *structure*,
then pure-numpy history-aware statistics at scoring time, so Neo4j stays off the
prediction path:

```bash
KGC_PROJECT=<project> python inference/global_context.py   # stage 1 -> global_context/tiers.pkl
KGC_PROJECT=<project> python inference/run_final_config.py # the frozen final configuration
```

Render everything for the current project with `drivers/make_reports.py`, and roll the
built projects up into the cross-project evaluation with `drivers/run_aggregate.py`.

**Statistical significance (RQ1).** The paper's paired comparison of KG-Commit against
each baseline — per project and across all 11 — is produced by:

```bash
python inference/significance_vs_baselines.py    # cache-only; no Neo4j, no rebuild
```

It aligns every model to the **common evaluation window** (the commits scored by both),
compares Macro-F1 and G-Mean with a paired percentile bootstrap (B=5000) for the 95% CI
and p-value, and reports a paired **Wilcoxon** signed-rank test on per-commit absolute
error as a threshold-free cross-check. The canonical project list lives in
`inference/paper_projects.py` (`ACTIVE` = the 11 projects of Table 4).

**Dataset.** 11 ApacheJIT projects, 78,206 evaluated commits (22,740 bug-inducing,
55,466 clean; ≈29.1% bug-inducing). Labels come from SZZ refined by issue linking and
GumTree filtering. Each commit carries 12 change metrics (Kamei et al., 2013), which
the pipeline log-transforms.

---

## 9. Baselines

The paper compares KG-Commit against **six baselines**, all evaluated under the *same*
online protocol (`inference/protocol.py`) and the same 7-metric suite:

| Baseline | Where | Notes |
|---|---|---|
| **LR** — logistic regression on change metrics | `baselines/run_baselines.py` | classical change-metric baseline |
| **HGB** — histogram gradient boosting | `baselines/run_baselines.py` | non-linear change-metric baseline |
| **RF** — random forest | `baselines/run_baselines.py` | |
| **LApredict** | `baselines/run_extra_baselines.py` | deliberately uses only lines-added |
| **JITLine** | `baselines/run_extra_baselines.py` | expert features + diff tokens; vocabulary frozen at warm-up |
| **DeepJIT** | `baselines/deepjit_kaggle.py` | CNN over message + code tokens; **runs on a Kaggle T4 GPU**, see `baselines/README_deepjit_kaggle.md` |

Run the five CPU baselines locally with `drivers/run_final.py` (steps 8–9) or directly:

```bash
python baselines/run_baselines.py          # LR, HGB, RF
python baselines/run_extra_baselines.py    # LApredict, JITLine (+ variants)
```

DeepJIT is the one baseline that needs a GPU; it is therefore run out-of-band on Kaggle
and its result pickle is copied back into `outputs/<project>/`.

These scripts additionally emit two reference points used in the analysis: a naive
**base-rate** predictor (`B_RATE`) and **JITLine-online** (`B_JITLINE_ONLINE`, which
refits vocabulary *and* model every block, unlike the frozen-vocabulary JITLine).

---

## 10. Data availability

This repository contains **code only**. The inputs and the prebuilt graphs are archived
separately (Data Availability Statement of the paper):

- **Graph dumps** — prebuilt Neo4j dumps: <https://zenodo.org/records/22348686>
  These let you skip the multi-hour build entirely. Place a dump at
  `outputs/<project>/neo4j_dump/<project>.dump`, then:
  ```bash
  KGC_PROJECT=<project> python snapshot_neo4j.py restore
  ```
  Requires Docker (dump/load runs `neo4j-admin` offline against the same data volume;
  see §4 and the script's header for the `KGC_NEO4J_*` overrides).
- **ApacheJIT dataset** — commit labels and JIT metrics:
  <https://zenodo.org/records/5907002>
- **Apache project repositories** — cloned from their public upstreams (§3, §5).

---

## 11. Citation

The manuscript is under review at *Knowledge-Based Systems*
(submission **KNOSYS-D-26-21009**). Please cite the paper once published; until then,
cite this repository:

```bibtex
@misc{kgcommit2026,
  title  = {{KG-Commit}: A Dynamic Knowledge Graph for Online Just-in-Time
            Software Defect Prediction},
  author = {Hesamolhokama, Mohsen and
            Beyrami Aghbash, Mohammad Sina and
            Rohani, Behnam and
            Fazli, MohammadAmin and
            Habibi, Jafar},
  note   = {Manuscript under review, Knowledge-Based Systems (KNOSYS-D-26-21009)},
  year   = {2026},
  howpublished = {\url{https://github.com/Knowledge4Software/kg-commit}}
}
```

## License

See `LICENSE`.
