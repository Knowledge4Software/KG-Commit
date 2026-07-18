"""
KG-Commit reproducibility package -- single source of truth for per-project paths.

Every build / inference / scalability script in this package imports the constants
it needs from HERE instead of hard-coding a project. The ONLY runtime switch is the
environment variable KGC_PROJECT (e.g. "zookeeper", "groovy"); this enforces the
"one project at a time, no loop" workflow -- if it is unset we fail loudly rather
than silently defaulting to a project.

Heavy data is READ IN PLACE and never copied:
  * cloned repo   : <repo_root>/repos/apache/<project>/
  * label CSV     : <repo_root>/data/apachejit/projects/apache_<project>.csv
  * diff CSV      : <repo_root>/data/apachejit/projects/apache_<project>_diff.csv
where <repo_root> is the parent of this kgcommit_repro/ folder (the original
KG-Commit checkout). All GENERATED artifacts are namespaced under
  <repo_root>/outputs/<project>/            (caches, results, checkpoints, figures)
  <repo_root>/outputs/<project>/scalability/(the E1-E5 + CSTG-ablation outputs)
  kgcommit_repro/logs/<project>/            (per-commit build timing CSVs, stdout)
so building one project never clobbers another's results.

Neo4j is a SINGLE shared database (nodes are not project-tagged). It therefore holds
exactly ONE project at a time; switch projects only after a full wipe
(see reset_neo4j.py). The per-project namespacing above is for the on-disk artifacts,
not for the graph store.

Usage in a copied script (example):
    from config.project_config import (REPO_PATH, BASE_COMMIT, CSV_PATH, DIFF_CSV,
                                        OUT, SCAL_OUT, NEO4J_URI, NEO4J_AUTH)
Scripts under build/ and inference/ that historically used a bare
    OUT = Path(__file__).resolve().parent[.parent] / "outputs"
now do
    from config.project_config import OUT
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for the reproducibility package config. "
        "Install it: pip install pyyaml") from e


# ---------------------------------------------------------------------------
# Locate the package and the original repo root (parent of kgcommit_repro/).
# Data (repos/, data/) and outputs/ live under the repo root and are read in place.
# ---------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent.parent          # .../kgcommit_repro
PROJECT_ROOT = PKG_ROOT.parent                              # .../KG-Commit (repo root)
REGISTRY = PKG_ROOT / "config" / "projects.yaml"


# ---------------------------------------------------------------------------
# Active project selection (env-driven; loud failure if unset/unknown).
# ---------------------------------------------------------------------------
def _load_registry() -> dict:
    if not REGISTRY.exists():
        raise FileNotFoundError(
            f"Project registry not found: {REGISTRY}. "
            f"Copy config/projects.template.yaml or edit config/projects.yaml.")
    with open(REGISTRY, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _active_name() -> str:
    name = os.environ.get("KGC_PROJECT")
    if not name:
        raise RuntimeError(
            "KGC_PROJECT is not set. This package runs ONE project at a time.\n"
            "  PowerShell:  $env:KGC_PROJECT = 'zookeeper'\n"
            "  bash:        export KGC_PROJECT=zookeeper\n"
            "Known projects: " + ", ".join(sorted(available_projects())))
    return name


def available_projects() -> list:
    reg = _load_registry()
    return list((reg.get("projects") or {}).keys())


_REG = _load_registry()
_PROJECTS = _REG.get("projects") or {}
_NEO4J_DEFAULT = _REG.get("neo4j") or {}

PROJECT = _active_name()
if PROJECT not in _PROJECTS:
    raise KeyError(
        f"KGC_PROJECT='{PROJECT}' is not in {REGISTRY}. "
        f"Known: {', '.join(sorted(_PROJECTS))}. "
        f"Add it (see config/projects.template.yaml) before running.")

_CFG = _PROJECTS[PROJECT]


def _require(field: str):
    if field not in _CFG or _CFG[field] in (None, ""):
        raise KeyError(
            f"Project '{PROJECT}' is missing required field '{field}' in "
            f"{REGISTRY}. (For 'base_commit', run find_base_commit.py to pick one.)")
    return _CFG[field]


# ---------------------------------------------------------------------------
# Resolved per-project constants (what the scripts import).
# ---------------------------------------------------------------------------
PROJECT_KEY = _CFG.get("project_key", f"apache/{PROJECT}")           # e.g. apache/zookeeper
REPO_PATH = (PROJECT_ROOT / _require("repo_subpath")).resolve()      # cloned repo (in place)
CSV_PATH = (PROJECT_ROOT / _require("label_csv")).resolve()          # ApacheJIT labels (in place)
DIFF_CSV = (PROJECT_ROOT / _require("diff_csv")).resolve()           # diff text for CSTG (in place)

# ---------------------------------------------------------------------------
# Multi-repository support (repository-split provenance).
#
# Some ApacheJIT projects (e.g. hadoop-mapreduce) have labelled commits whose
# SHAs are distributed across MORE THAN ONE git repository, because the project
# underwent a repository split/re-merge that rewrote history. A single clone can
# never resolve the whole labelled stream. A project may therefore declare extra
# repositories via `extra_repo_subpaths:` (a list). REPO_PATHS is the ordered
# search list (primary REPO_PATH first, then the extras). Single-repo projects
# are unaffected: with no extras, REPO_PATHS == [REPO_PATH] and every git access
# behaves exactly as before. See docs/Critical_notes.tex for the full rationale.
# ---------------------------------------------------------------------------
_EXTRA_REPOS = _CFG.get("extra_repo_subpaths") or []
REPO_PATHS = [REPO_PATH] + [
    (PROJECT_ROOT / p).resolve() for p in _EXTRA_REPOS
]

# cache: commit-SHA (or ref) -> repo Path that resolves it (None if unresolvable)
_REF_REPO_CACHE = {}


def git_root_for(ref: str):
    """Return the first repo in REPO_PATHS in which `ref` resolves to an object,
    or None. Result is cached. For single-repo projects this always returns
    REPO_PATH (or None) after one probe."""
    import subprocess as _sp
    if ref in _REF_REPO_CACHE:
        return _REF_REPO_CACHE[ref]
    # strip any ':path' or '^1' suffix for the existence probe of the commit-ish
    for _root in REPO_PATHS:
        r = _sp.run(["git", "-C", str(_root), "cat-file", "-e", f"{ref}^{{commit}}"],
                    capture_output=True)
        if r.returncode == 0:
            _REF_REPO_CACHE[ref] = _root
            return _root
    _REF_REPO_CACHE[ref] = None
    return None

# BASE_COMMIT is only needed by the base-snapshot BUILD steps. It may be blank for a
# freshly-registered project until find_base_commit.py picks one; so it is resolved
# LAZILY -- module import stays valid (inference/scalability on an already-built graph
# do not need it), and the build scripts that DO use it fail loudly if it is unset.
_BASE_RAW = _CFG.get("base_commit") or ""


def base_commit() -> str:
    if not _BASE_RAW:
        raise KeyError(
            f"Project '{PROJECT}' has no base_commit in {REGISTRY}. "
            f"Run: python find_base_commit.py   then paste the SHA into projects.yaml.")
    return str(_BASE_RAW)


# Backwards-compatible module attribute for scripts that read BASE_COMMIT directly.
# It is the string if set, else an empty string (build scripts should call
# base_commit() to get the loud check; reading the attribute stays import-safe).
BASE_COMMIT = str(_BASE_RAW)

_NS = _CFG.get("output_namespace", PROJECT)
OUT = (PROJECT_ROOT / "outputs" / _NS)                               # per-project generated artifacts
SCAL_OUT = OUT / "scalability"                                       # E1-E5 + CSTG-ablation outputs
FIG_DIR = OUT / "figures" / "v4"                                     # per-project figures
TIMING_DIR = PKG_ROOT / "logs" / _NS                                 # per-commit build timing CSVs
OUT.mkdir(parents=True, exist_ok=True)
SCAL_OUT.mkdir(parents=True, exist_ok=True)
TIMING_DIR.mkdir(parents=True, exist_ok=True)

# Per-project checkpoints (resumable online build).
CKPT_PATH = OUT / "online_kg_checkpoint.json"


def ckpt_path(kind: str) -> Path:
    """Per-kind checkpoint for the subgraph engine (cfg/dfg/pdg/seq)."""
    return OUT / f"online_kg_checkpoint_{kind}.json"


# ---------------------------------------------------------------------------
# Neo4j (per-project override allowed; else the shared default).
# ---------------------------------------------------------------------------
_n = {**_NEO4J_DEFAULT, **(_CFG.get("neo4j") or {})}
NEO4J_URI = _n.get("uri", "bolt://localhost:7687")
NEO4J_AUTH = (_n.get("user", "neo4j"), _n.get("password", "password1234"))


def summary() -> str:
    return (
        f"KGC_PROJECT     = {PROJECT}\n"
        f"PROJECT_KEY     = {PROJECT_KEY}\n"
        f"REPO_PATH       = {REPO_PATH}   (exists={REPO_PATH.exists()})\n"
        f"BASE_COMMIT     = {BASE_COMMIT}\n"
        f"CSV_PATH        = {CSV_PATH}   (exists={CSV_PATH.exists()})\n"
        f"DIFF_CSV        = {DIFF_CSV}   (exists={DIFF_CSV.exists()})\n"
        f"OUT             = {OUT}\n"
        f"SCAL_OUT        = {SCAL_OUT}\n"
        f"TIMING_DIR      = {TIMING_DIR}\n"
        f"CKPT_PATH       = {CKPT_PATH}\n"
        f"NEO4J_URI       = {NEO4J_URI}\n")


if __name__ == "__main__":
    print(summary())
