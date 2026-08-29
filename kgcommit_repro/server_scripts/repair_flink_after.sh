#!/usr/bin/env bash
# Repair flink's four stages that died on the +/-inf bug, then re-collect.
#
# WHY: np.nan_to_num(x, nan=...) leaves +/-inf untouched. An inf in a score array
# makes sklearn's FPR non-monotonic ("x is neither increasing nor decreasing") and
# breaks RandomForest's bootstrap ("probabilities do not sum to 1"). That killed:
#     7 effort-aware eval, 8 baselines, 9 extra baselines, 10 parameter sweeps
# cum_metrics clips to [0,1], which is why fusion/subgraph survived. online_jit.py
# now guards posinf/neginf at all five call sites.
#
# All four stages are CACHE-ONLY (no Neo4j), so this cannot disturb the graph or
# contend with a restore. It still waits for the queue and the groovy re-run so it
# never competes for CPU.
#
# flink's fusion + subgraph results are FRESH from this run and are NOT touched.

LOG=~/repair_flink_after.log
echo "$(date '+%F %T')  waiting for queue + groovy re-run ..." > "$LOG"

while pgrep -f '^bash /home/user01/run_final_all\.sh' >/dev/null \
   || pgrep -f '^bash /home/user01/rerun_groovy_after\.sh' >/dev/null \
   || pgrep -f '^bash /home/user01/run_final_project\.sh' >/dev/null; do
  sleep 120
done
sleep 20

cd ~/KG-Commit/kgcommit_repro || exit 1
source .venv/bin/activate
export KGC_PROJECT=flink
export PYTHONPATH=~/KG-Commit/kgcommit_repro
export MPLBACKEND=Agg
# no KGC_NEO4J_* on purpose: every stage below reads caches only.

echo "$(date '+%F %T')  starting flink repair (4 cache-only stages)" >> "$LOG"
echo "$(date '+%F %T')  START   flink (repair)" >> ~/final_run_status.txt

run () {   # run <label> <script> [subdir]
  echo "$(date '+%F %T')  -> $1" >> "$LOG"
  if [ -n "$3" ]; then s="$3/$2"; else s="inference/$2"; fi
  nice -n 5 python3 -u "$s" >> ~/final_flink_repair.log 2>&1
  echo "$(date '+%F %T')     $1 rc=$?" >> "$LOG"
}

: > ~/final_flink_repair.log
run "effort eval"      run_effort_eval.py
run "baselines"        run_baselines.py       baselines
run "extra baselines"  run_extra_baselines.py baselines
run "param sweeps"     run_param_experiments.py

echo "$(date '+%F %T')  re-rendering reports + collecting" >> "$LOG"
nice -n 5 python3 -u drivers/make_reports.py --no-notebooks >> ~/final_flink_repair.log 2>&1
nice -n 5 python3 -u collect_final_run.py                   >> ~/final_flink_repair.log 2>&1

# Verify the artifacts that were stale/missing are now FRESH (<6h old).
fresh () { [ -n "$(find "$1" -mmin -360 2>/dev/null)" ] && echo yes || echo no; }
O=~/KG-Commit/outputs/flink
E=$(fresh $O/effort_results.pkl)
B=$(fresh $O/baseline_results.pkl)
X=$(fresh $O/baseline_extra_results.pkl)
K=$(fresh $O/param_experiments/K.json)

if [ "$E" = yes ] && [ "$B" = yes ] && [ "$X" = yes ] && [ "$K" = yes ]; then
  echo "$(date '+%F %T')  OK      flink (repair) (effort/baselines/extra/params all fresh)" >> ~/final_run_status.txt
else
  echo "$(date '+%F %T')  FAILED  flink (repair) (effort=$E baselines=$B extra=$X params=$K)" >> ~/final_run_status.txt
fi
echo "$(date '+%F %T')  done" >> "$LOG"
