# KG-Commit — Reproducibility Package

This folder is the **self-contained, publishable code artifact** for the paper
*KG-Commit: a multimodal, fully-online knowledge graph for just-in-time (commit-level)
defect prediction*. It contains **only the final (V4) methodology** — the graph layers,
inference methods, fusion, scalability analysis, and figures that appear in the paper —
and runs the whole pipeline end-to-end (raw repo + ApacheJIT data → Neo4j knowledge
graph → all experiments → all tables/figures/notebooks) on **one Apache project at a
time**.

> **One project at a time, on purpose.** The pipeline is *not* a loop over projects.
> You pick a project with the `KGC_PROJECT` environment variable, build and analyse it,
> read the results, then move to the next. Start with the **light** projects
> (zookeeper, zeppelin) and work up.

---

## 1. What is (and isn't) in here

**In (final methodology only):**
- `build/` — the ten-step knowledge-graph build (base snapshot + Core process layer,
  AST delta layer, CFG/DFG/PDG/Token-seq structural layers, all grown *online* per
  commit) plus the parsers/differs they use.
- `inference/` — the five graph-inference methods (RN, PPR, LP, DW, KGE), the fusion,
  the CSTG semantic-text feature channel, the evaluation protocol, and the table/figure
  renderers.
- `scalability/` — the second research question: KG statistical profile, per-commit
  growth, time/space complexity, prediction latency, statistical-significance tests, and
  the CSTG ablations.
- `kg_commit/` — the small library that builds the Core process layer (commit/file/
  developer/issue graph) into Neo4j.
- `config/` — the per-project registry and the single source of truth for all paths.
- `drivers/` — four thin orchestrators that run the pipeline phase by phase.
- `find_base_commit.py`, `reset_neo4j.py` — the two setup helpers.

**Out:** all superseded V1–V3 experimentation, legacy builders, audits, and one-off
scripts from the research repo. This package is the clean final subset.

**Read in place, never copied:** the cloned repositories under `repos/apache/<project>/`
and the ApacheJIT CSVs under `data/apachejit/projects/`. This package references them by
path (relative to the parent repo root); it does not duplicate the multi-gigabyte data.
See §5 for exactly where each input is read from.

---

## 2. Layout

```
kgcommit_repro/
  README.md            this file
  RUNBOOK.md           copy-paste command sequence, per project
  requirements.txt     Python dependencies
  _kgc_paths.py        import bootstrap (puts the package dirs on sys.path)
  config/
    project_config.py  SINGLE source of truth: resolves per-project paths from
                       projects.yaml + the KGC_PROJECT env var
    projects.yaml      registry of projects (repo path, base commit, CSVs, namespace)
    projects.template.yaml  template for adding a new project X
  find_base_commit.py  suggest a project's base commit (first substantial Java commit)
  reset_neo4j.py       full DB wipe (run before switching projects)
  build/               the 10 build-step scripts + parsers/differs (Phase A/B/C)
  inference/           final methods, fusion, CSTG channel, renderers
  scalability/         E1–E5 + CSTG ablations
  kg_commit/           Core-layer builder library
  drivers/
    build_project.py   Phase A→B→C for $KGC_PROJECT (with per-commit timing logs)
    run_experiments.py final inference experiments (live Neo4j)
    run_scalability.py scalability / complexity / significance suite
    make_reports.py    all tables, figures, executed notebooks
  notebooks/           fresh copies of the four final notebooks (regenerated per project)
  logs/<project>/      per-commit build TIMING CSVs + build stdout (for the scalability RQ)
```

Generated artifacts are **namespaced per project** so builds never clobber each other:
```
<repo-root>/outputs/<project>/                 caches, result pickles, checkpoints
<repo-root>/outputs/<project>/scalability/     E1–E5 + CSTG-ablation JSON/TeX
<repo-root>/outputs/<project>/figures/v4/…     figures
<repo-root>/outputs/<project>/tables/v4/…      LaTeX tables
<repo-root>/outputs/<project>/notebooks/…      executed notebooks
kgcommit_repro/logs/<project>/                 per-commit build timing CSVs
```

---

## 3. Prerequisites

1. **Python 3.10+** and `pip install -r requirements.txt`.
2. **Neo4j 5.x** running at `bolt://localhost:7687` (default auth `neo4j` / `password1234`;
   change under `config/projects.yaml → neo4j:`). The graph store is **single-tenant** —
   it holds exactly one project's graph at a time (see §4).
3. **git** on `PATH` (the build reads the cloned repo via `git show` / `git diff`).
4. For each project you want to run: a **cloned repo** at `repos/apache/<project>/` and the
   **ApacheJIT CSVs** at `data/apachejit/projects/apache_<project>.csv` (labels) and
   `apache_<project>_diff.csv` (diff text, for the CSTG layer). Both CSV families ship
   with the dataset for all 15 ApacheJIT projects; the 11 pre-cloned repos are listed in
   `config/projects.yaml`.

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

| Input | Path (relative to the parent repo root) | Read by |
|---|---|---|
| Cloned source repo | `repos/apache/<project>/` | build engines (`git show`/`git diff`) |
| ApacheJIT labels + JIT metrics | `data/apachejit/projects/apache_<project>.csv` | `apply_commit_labels.py`, config |
| Diff text (CSTG) | `data/apachejit/projects/apache_<project>_diff.csv` | `validate_cstg.py`, `ingest_cstg.py`, `cstg_online_features.py` |

None of these are copied into the package. `config/project_config.py` resolves them from
the parent-repo root (the directory that contains both `kgcommit_repro/` and `outputs/`,
`repos/`, `data/`).

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
  built only from strictly earlier commits (prequential predict-then-grow). Warm-up is the
  first 40% of the stream; models refresh on 200-commit blocks. These protocol constants
  are fixed across all projects (`inference/online_infer.py`).
- **Deployed model.** The final predictor is the graph-native fusion
  **F + G = RN + PPR + CSTG** (the two-method graph fusion plus the CSTG semantic channel);
  the non-graph JIT-metric channel and the commit-message channel were dropped in V4.
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
