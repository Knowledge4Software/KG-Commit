#!/usr/bin/env bash
# Full multi-project pipeline: load dump -> extract tiers -> (all projects) ->
# arm suite with early-drop -> pooled evaluation.
#
# Protected: every stage is guarded, a failure on one project does not stop the
# chain, and completed projects are skipped on re-launch (resumable).
#
# Run:  nohup bash inference/run_full_pipeline.sh > ~/pipeline.log 2>&1 &
# Watch: tail -f ~/pipeline.log ; cat ~/KG-Commit/outputs/_allproj/progress.json

set -u
cd ~/KG-Commit/kgcommit_repro
source ~/kgc_env.sh >/dev/null 2>&1

CONT=neo4j-kgc2026
PASS=password1234
MARK=~/pipeline_stage.txt

# cheapest graph first so a late failure costs least
PROJECTS="zookeeper spark zeppelin kafka groovy activemq cassandra hive camel flink hbase"

load_dump () {
  local P=$1
  if [ ! -f ~/dump/$P.dump ]; then echo "  !! no dump for $P"; return 1; fi
  cp ~/dump/$P.dump ~/dumpstage/neo4j.dump || return 1
  docker stop $CONT >/dev/null 2>&1
  sleep 5
  docker run --rm -e NEO4J_HOME=/newadmin \
    -v /home/user01/neo4j/data:/data \
    -v /home/user01/neo4j2026:/newadmin \
    -v /home/user01/dumpstage:/import \
    neo4j:5 bash -lc \
    "/newadmin/bin/neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true" \
    2>&1 | tail -2
  docker start $CONT >/dev/null 2>&1
  sleep 35
  local NAME
  NAME=$(docker exec $CONT cypher-shell -u neo4j -p $PASS --format plain \
         "MATCH (p:Project) RETURN p.name;" 2>/dev/null | tail -1)
  echo "  loaded: $NAME"
  case "$NAME" in *"$P"*) return 0 ;; *) echo "  !! mismatch, expected $P"; return 1 ;; esac
}

echo "=============================================================="
echo "STAGE 1/3  tier extraction (needs Neo4j; one dump load each)"
echo "=============================================================="
for P in $PROJECTS; do
  echo "STAGE1:$P" > $MARK
  if [ -f ~/KG-Commit/outputs/$P/global_context/tiers.pkl ]; then
    echo "[$P] tiers already present, skipping"; continue
  fi
  echo "[$P] loading dump ..."
  if ! load_dump "$P"; then echo "[$P] LOAD FAILED - skipping"; continue; fi
  export KGC_PROJECT=$P
  if python -u inference/global_context.py > ~/tiers_$P.log 2>&1; then
    echo "[$P] tiers OK"
  else
    echo "[$P] TIER EXTRACTION FAILED (see ~/tiers_$P.log)"
    tail -3 ~/tiers_$P.log
  fi
done

echo
echo "=============================================================="
echo "STAGE 2/3  arm suite over all projects (cache-only, early-drop)"
echo "=============================================================="
echo "STAGE2" > $MARK
python -u inference/run_all_projects.py 2>&1 | tail -400

echo
echo "=============================================================="
echo "STAGE 3/3  pooled cross-project evaluation (frozen design)"
echo "=============================================================="
echo "STAGE3" > $MARK
python -u inference/run_pooled_eval.py 2>&1 | tail -60

echo "PIPELINE_DONE" > $MARK
echo
echo "PIPELINE COMPLETE"
