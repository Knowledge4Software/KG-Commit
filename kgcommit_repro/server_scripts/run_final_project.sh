#!/usr/bin/env bash
# Final run for ONE project: restore its dump, run everything, collect.
# Usage:  bash ~/run_final_project.sh <project>
#
# Guards (an unattended multi-hour job must not corrupt another run):
#   * PID lockfile + ps check   -- never two pipelines at once
#   * project assertion         -- a mis-set KGC_PROJECT writes into the wrong folder
#   * dump existence check      -- refuse to wipe the resident graph without a target
#   * disk floor                -- abort before filling the disk mid-run
set -o pipefail

P="${1:?usage: run_final_project.sh <project>}"
LOCK=~/.kgc_pipeline.lock

if [ -e "$LOCK" ]; then
  echo "ABORT: lock $LOCK exists (pid $(cat "$LOCK" 2>/dev/null))."; exit 9
fi
# Guard against a SECOND pipeline. Deliberately narrow: match only the actual
# worker processes, never this script, its parent queue (run_final_all.sh), or the
# grep itself -- an over-broad pattern made every queued project abort with rc=9.
if pgrep -f "drivers/run_final\.py|drivers/build_project\.py|build/build_(online|subgraph_online)_kg\.py"      | grep -qv "^$$\$"; then
  echo "ABORT: a pipeline worker is already running."; exit 9
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd ~/KG-Commit/kgcommit_repro || exit 1
source .venv/bin/activate
export KGC_PROJECT="$P"
export PYTHONPATH=~/KG-Commit/kgcommit_repro
# Neo4j 2026.07.0: the dumps carry kernel v29, unopenable by the 5.26 binaries
# in the neo4j:5 image. The image supplies Java 21; NEO4J_HOME/KGC_NEO4J_DIST
# point at the newer distribution actually used for load and for serving.
export KGC_NEO4J_CONTAINER=neo4j-kgc2026 KGC_NEO4J_IMAGE=neo4j:5
export KGC_NEO4J_DIST=/home/user01/neo4j2026
export MPLBACKEND=Agg

echo "=== PREFLIGHT [$P] ==="; date
python3 -m config.project_config || exit 1
python3 -m config.project_config | grep -q "KGC_PROJECT     = $P" \
  || { echo "ABORT: KGC_PROJECT mismatch"; exit 1; }
DUMP=~/KG-Commit/outputs/$P/neo4j_dump/$P.dump
[ -f "$DUMP" ] || { echo "ABORT: $DUMP missing"; exit 1; }
free_gb=$(df -BG --output=avail ~ | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb}G"
[ "$free_gb" -lt 8 ] && { echo "ABORT: <8G free"; exit 1; }

echo "=== RESTORE [$P] ==="; date
# clear any stale staged dump owned by the neo4j uid (else copy2 -> EACCES)
docker exec $KGC_NEO4J_CONTAINER rm -f /data/_kgc_dumps/neo4j.dump 2>/dev/null || true
python3 -u snapshot_neo4j.py restore || { echo "RESTORE FAILED"; exit 1; }
sleep 20   # let Bolt finish coming up before the first query

# cstg_bundle.pkl is needed by the CSTG internal ablation (appendix). It is
# DB-free (fits from the diff CSV), and six projects did not have one, so the
# ablation failed for them. Generate it if missing.
if [ ! -f ~/KG-Commit/outputs/$P/cstg_bundle.pkl ]; then
  echo "=== CSTG BUNDLE [$P] (missing -- generating) ==="; date
  python3 -u inference/validate_cstg.py || echo "validate_cstg returned $?"
fi

echo "=== EXPERIMENTS [$P] ==="; date
python3 -u drivers/run_final.py || echo "run_final returned $?"

echo "=== SCALABILITY [$P] ==="; date
python3 -u drivers/run_scalability.py || echo "scalability returned $?"

echo "=== REPORTS [$P] ==="; date
python3 -u drivers/make_reports.py --no-notebooks || echo "reports returned $?"

echo "=== RQ3 PER-METRIC RADARS ==="; date
python3 -u inference/make_rq3_metric_radars.py || true

echo "=== COLLECT -> final_run/ [$P] ==="; date
python3 -u collect_final_run.py || echo "collect returned $?"

echo "=== DONE [$P] ==="; date
