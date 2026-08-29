#!/usr/bin/env bash
# Camel: restore graph + run the two Neo4j-bound jobs, then re-render reports.
set -o pipefail

LOCK=~/.kgc_pipeline.lock
if [ -e "$LOCK" ]; then
  echo "ABORT: lock $LOCK exists (pid $(cat $LOCK 2>/dev/null))."; exit 9
fi
if ps -ef | grep -E "build_project|drivers/|build_subgraph|run_kg_methods" | grep -v grep | grep -qv "run_camel_kg"; then
  echo "ABORT: a pipeline process is already running."; exit 9
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd ~/KG-Commit/kgcommit_repro || exit 1
source .venv/bin/activate
export KGC_PROJECT=camel
export PYTHONPATH=~/KG-Commit/kgcommit_repro
export KGC_NEO4J_CONTAINER=neo4j-kgc KGC_NEO4J_IMAGE=neo4j:5

echo "=== PREFLIGHT ==="; date
python3 -m config.project_config || exit 1
python3 -m config.project_config | grep -q "KGC_PROJECT     = camel" || { echo "ABORT: wrong project"; exit 1; }
[ -f ~/KG-Commit/outputs/camel/neo4j_dump/camel.dump ] || { echo "ABORT: camel.dump missing"; exit 1; }
[ -f ~/KG-Commit/outputs/groovy/neo4j_dump/groovy.dump ] || { echo "ABORT: groovy.dump missing - refusing to overwrite resident graph"; exit 1; }
free_gb=$(df -BG --output=avail ~ | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb}G"; [ "$free_gb" -lt 8 ] && { echo "ABORT: <8G free"; exit 1; }

echo "=== RESTORE camel graph ==="; date
docker exec $KGC_NEO4J_CONTAINER rm -f /data/_kgc_dumps/neo4j.dump 2>/dev/null || true
python3 -u snapshot_neo4j.py restore || { echo "RESTORE FAILED"; exit 1; }

echo "=== KG METHODS RQ (the long one) ==="; date
python3 -u inference/run_kg_methods_rq.py || echo "run_kg_methods_rq returned $?"

echo "=== SUBGRAPH LAYER STATS ==="; date
python3 -u inference/collect_subgraph_stats.py || echo "collect_subgraph_stats returned $?"

echo "=== REPORTS ==="; date
python3 -u drivers/make_reports.py --no-notebooks

echo "=== CAMEL KG JOB DONE ==="; date
