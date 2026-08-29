#!/usr/bin/env bash
# Chained + protected full-pipeline run for groovy.
set -o pipefail

LOCK=~/.kgc_pipeline.lock
# --- guard: refuse to start if another pipeline holds the lock or is running ---
if [ -e "$LOCK" ]; then
  echo "ABORT: lock $LOCK exists (pid $(cat $LOCK 2>/dev/null)). Another pipeline may be running."
  exit 9
fi
if ps -ef | grep -E "build_project|drivers/|build_subgraph" | grep -v grep | grep -qv "run_groovy"; then
  echo "ABORT: a pipeline process is already running."
  exit 9
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd ~/KG-Commit/kgcommit_repro || exit 1
source .venv/bin/activate
export KGC_PROJECT=groovy
export PYTHONPATH=~/KG-Commit/kgcommit_repro
export KGC_NEO4J_CONTAINER=neo4j-kgc KGC_NEO4J_IMAGE=neo4j:5

# --- guard: confirm the right project and that inputs exist ---
echo "=== PREFLIGHT ==="
python3 -m config.project_config || exit 1
python3 -m config.project_config | grep -q "KGC_PROJECT     = groovy" || { echo "ABORT: wrong project"; exit 1; }
if python3 -m config.project_config | grep -q "exists=False"; then
  echo "ABORT: an input path is missing."; exit 1
fi
free_gb=$(df -BG --output=avail ~ | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb}G"
[ "$free_gb" -lt 8 ] && { echo "ABORT: <8G free"; exit 1; }

echo "=== BUILD ===";       date; python3 -u drivers/build_project.py    || { echo "BUILD FAILED"; exit 1; }
echo "=== SNAPSHOT ===";    date; python3 -u snapshot_neo4j.py dump      || { echo "SNAPSHOT FAILED"; exit 1; }
echo "=== EXPERIMENTS ==="; date; python3 -u drivers/run_experiments.py
echo "=== SCALABILITY ==="; date; python3 -u drivers/run_scalability.py
echo "=== REPORTS ===";     date; python3 -u drivers/make_reports.py
echo "=== PIPELINE DONE ==="; date
