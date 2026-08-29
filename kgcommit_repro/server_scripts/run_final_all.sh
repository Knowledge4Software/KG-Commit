#!/usr/bin/env bash
# Final run for ALL projects, sequentially, unattended.
#
# Neo4j Community holds one graph at a time, so projects MUST be sequential: each
# one restores its dump (replacing the resident graph), runs, and is collected
# before the next begins.
#
# Protection:
#   * nohup + setsid + </dev/null  -- survives SSH disconnect / laptop shutdown
#   * queue-level lockfile         -- a second queue cannot start
#   * per-project lock is released by run_final_project.sh's own trap
#   * a failing project does NOT stop the queue; it is recorded and the run moves on
#   * per-project logs + a queue-level status file, so progress is inspectable
#     without reading multi-MB logs
#
# Usage:
#   bash ~/run_final_all.sh                      # default order (small -> large)
#   bash ~/run_final_all.sh kafka spark camel    # explicit subset/order
set -o pipefail

QLOCK=~/.kgc_queue.lock
STATUS=~/final_run_status.txt

if [ -e "$QLOCK" ]; then
  echo "ABORT: queue lock $QLOCK exists (pid $(cat "$QLOCK" 2>/dev/null))."
  exit 9
fi
echo $$ > "$QLOCK"
trap 'rm -f "$QLOCK"' EXIT

# Smallest first: a systematic problem surfaces on a cheap project, not an
# expensive one. zookeeper is re-run so every project's artifacts come from one
# uninterrupted pass under the final code.
DEFAULT_ORDER="zookeeper zeppelin spark kafka activemq hive groovy cassandra hbase flink camel"
PROJECTS="${*:-$DEFAULT_ORDER}"

{
  echo "queue started : $(date)"
  echo "projects      : $PROJECTS"
  echo "-----------------------------------------------------------"
} > "$STATUS"

for P in $PROJECTS; do
  echo "$(date '+%F %T')  START   $P" >> "$STATUS"
  START=$(date +%s)

  bash ~/run_final_project.sh "$P" > ~/final_${P}.log 2>&1
  RC=$?

  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  # "DONE" only proves the script reached the end; verify the artifacts that
  # matter actually exist before calling it a success.
  FUSION=~/KG-Commit/outputs/$P/final_run/fusion/final_fusion_results.pkl
  NTAB=$(ls ~/KG-Commit/outputs/$P/final_run/tables 2>/dev/null | wc -l)
  NFIG=$(find ~/KG-Commit/outputs/$P/final_run/figures -type f 2>/dev/null | wc -l)
  if [ -f "$FUSION" ] && [ "$NTAB" -gt 0 ] && [ "$NFIG" -gt 0 ]; then
    echo "$(date '+%F %T')  OK      $P  (${ELAPSED}m, rc=$RC, ${NTAB} tables, ${NFIG} figures)" >> "$STATUS"
  else
    echo "$(date '+%F %T')  FAILED  $P  (${ELAPSED}m, rc=$RC, fusion=$([ -f "$FUSION" ] && echo y || echo n), tables=$NTAB, figures=$NFIG)" >> "$STATUS"
  fi
done

echo "-----------------------------------------------------------" >> "$STATUS"
echo "queue finished: $(date)" >> "$STATUS"
