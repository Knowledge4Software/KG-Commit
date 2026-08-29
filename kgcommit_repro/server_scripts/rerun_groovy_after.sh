#!/usr/bin/env bash
# Re-run groovy once the main queue finishes.
#
# WHY: groovy's fusion stage aborted on a stale stream cache (8,039 cached commits
# vs 8,059 in the restored graph), which skipped stages 2-10 -- fusion, subgraph RQ,
# baselines, seeds and parameter sweeps -- while the run still printed DONE and the
# status line said OK (the artifact check saw stale tables from an earlier pull).
#
# The stale cache is quarantined in ~/stale_groovy_2026-08-23/ and
# run_final_fusion.py now REBUILDS a mismatched cache instead of aborting, so this
# re-run starts from a clean slate.
#
# Waits for run_final_all.sh to exit so the two never contend for Neo4j.

LOG=~/rerun_groovy_after.log
echo "$(date '+%F %T')  waiting for the main queue to finish ..." > "$LOG"

# Wait on the QUEUE only. Match the full command line so an unrelated shell that
# merely mentions the script name cannot keep this waiting forever.
while pgrep -f '^bash /home/user01/run_final_all\.sh' >/dev/null; do
  sleep 120
done
echo "$(date '+%F %T')  queue finished; waiting for the per-project lock to clear" >> "$LOG"

# The last project's own trap releases the lock; give it a moment.
for _ in $(seq 1 30); do
  [ -e ~/.kgc_pipeline.lock ] || break
  sleep 10
done
sleep 20

echo "$(date '+%F %T')  starting groovy re-run" >> "$LOG"
echo "$(date '+%F %T')  START   groovy (re-run)" >> ~/final_run_status.txt

bash ~/run_final_project.sh groovy > ~/final_groovy.log 2>&1
RC=$?

# "DONE" only proves the script reached the end. Verify the artifacts that the
# first attempt silently lacked -- and that the fusion pkl is NEW, not the stale
# one that made the first attempt look successful.
FUS=~/KG-Commit/outputs/groovy/final_run/fusion/final_fusion_results.pkl
NT=$(ls ~/KG-Commit/outputs/groovy/final_run/tables 2>/dev/null | wc -l)
NF=$(find ~/KG-Commit/outputs/groovy/final_run/figures -type f 2>/dev/null | wc -l)
FRESH=no
if [ -f "$FUS" ]; then
  # fresh = modified within the last 6 hours
  [ -n "$(find "$FUS" -mmin -360 2>/dev/null)" ] && FRESH=yes
fi

if [ "$FRESH" = "yes" ] && [ "$NT" -gt 0 ] && [ "$NF" -gt 0 ]; then
  echo "$(date '+%F %T')  OK      groovy (re-run) (rc=$RC, $NT tables, $NF figures, fusion fresh)" >> ~/final_run_status.txt
else
  echo "$(date '+%F %T')  FAILED  groovy (re-run) (rc=$RC, tables=$NT, figures=$NF, fusion_fresh=$FRESH)" >> ~/final_run_status.txt
fi
echo "$(date '+%F %T')  done (rc=$RC, fusion_fresh=$FRESH)" >> "$LOG"
