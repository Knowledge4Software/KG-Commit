#!/usr/bin/env bash
# Overnight auto-resume build for one project, robust to transient Bolt drops.
# Strategy (matches the proven manual cassandra recovery):
#   Phase A + each online layer is run individually; the FIRST attempt of an online
#   layer uses --reset, RETRIES resume from the layer's checkpoint (NO --reset, MERGE
#   is idempotent). Then experiments -> scalability -> reports -> probe -> dump.
set -u
PROJ="${KGC_PROJECT:?set KGC_PROJECT}"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PKG:$PKG/inference:$PKG/baselines${PYTHONPATH:+:$PYTHONPATH}"
LOG="$PKG/logs/${PROJ}_overnight.log"; mkdir -p "$PKG/logs" "$PKG/logs/$PROJ"
echo "===== OVERNIGHT $PROJ start $(date) =====" | tee -a "$LOG"
B="$PKG/build"; TL="$PKG/logs/$PROJ"

wait_bolt () {
  for i in $(seq 1 80); do
    docker exec neo4j-local cypher-shell -u neo4j -p password1234 "RETURN 1" >/dev/null 2>&1 && return 0
    docker start neo4j-local >/dev/null 2>&1; sleep 5
  done; return 1
}

# run a command with up to $1 attempts; $2 label; rest = cmd. resumes rely on the
# script's own checkpoint (we pass --reset only on attempt 1 via $RESET_ON_FIRST).
retry () {
  local max="$1" label="$2"; shift 2
  for a in $(seq 1 "$max"); do
    echo ">>> [$label] attempt $a/$max $(date)" | tee -a "$LOG"
    wait_bolt || { echo "!! bolt down" | tee -a "$LOG"; return 1; }
    if "$@" >>"$LOG" 2>&1; then echo ">>> [$label] OK" | tee -a "$LOG"; return 0; fi
    echo "!! [$label] fail $a; retry in 20s" | tee -a "$LOG"; sleep 20
  done
  echo "!! [$label] GAVE UP" | tee -a "$LOG"; return 1
}

# online layer with reset-on-first-attempt-only
layer () {   # layer <max> <label> <engine.py> <kind-args...>
  local max="$1" label="$2" eng="$3"; shift 3
  for a in $(seq 1 "$max"); do
    echo ">>> [$label] attempt $a/$max $(date)" | tee -a "$LOG"
    wait_bolt || { echo "!! bolt down" | tee -a "$LOG"; return 1; }
    local rst=(); [ "$a" -eq 1 ] && rst=(--reset)
    if python "$B/$eng" "$@" "${rst[@]}" --timing-log "$TL/${label}_timing.csv" >>"$LOG" 2>&1; then
      echo ">>> [$label] OK" | tee -a "$LOG"; return 0
    fi
    echo "!! [$label] fail $a; resume (no reset) in 20s" | tee -a "$LOG"; sleep 20
  done
  echo "!! [$label] GAVE UP" | tee -a "$LOG"; return 1
}

# --- Phase A (base + Core), idempotent, retry as a block ---
retry 4 "PHASE_A" bash -c "python '$B/build_file_ast_graphs.py' && python '$B/ingest_base_kg_and_ast.py' && python '$B/add_positions_to_astnodes.py' && python '$B/apply_commit_labels.py'"
# --- Phase B: online layers, each resumable ---
layer 6 "AST" build_online_kg.py
layer 6 "CFG" build_subgraph_online_kg.py --kind cfg
layer 6 "DFG" build_subgraph_online_kg.py --kind dfg
layer 6 "PDG" build_subgraph_online_kg.py --kind pdg
layer 6 "SEQ" build_subgraph_online_kg.py --kind seq
# --- Phase C: CSTG ---
retry 3 "CSTG" bash -c "python '$PKG/inference/validate_cstg.py' && python '$PKG/inference/ingest_cstg.py' --ground && python '$PKG/inference/ingest_cstg_enrich.py'"
# --- inference / scalability / reports / probe / baselines / headabl / dump ---
retry 3 "EXPERIMENTS" python "$PKG/drivers/run_experiments.py"
retry 3 "SCALABILITY" python "$PKG/drivers/run_scalability.py"
retry 2 "REPORTS"     python "$PKG/drivers/make_reports.py" --no-notebooks
retry 2 "PROBE"       python "$PKG/inference/probe_ast_cfg.py"
retry 2 "XBASE"       python "$PKG/baselines/run_extra_baselines.py"
retry 2 "HEADABL"     python "$PKG/inference/fusion_head_ablation.py"
retry 2 "DUMP"        python "$PKG/snapshot_neo4j.py" dump
echo "===== OVERNIGHT $PROJ done $(date) =====" | tee -a "$LOG"
