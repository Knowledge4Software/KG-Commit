#!/usr/bin/env bash
set -o pipefail
LOCK=~/.kgc_pipeline.lock
if [ -e "$LOCK" ]; then echo "ABORT: lock exists (pid $(cat $LOCK))"; exit 9; fi
echo $$ > "$LOCK"; trap "rm -f $LOCK" EXIT
cd ~/KG-Commit/kgcommit_repro || exit 1
source .venv/bin/activate
export KGC_PROJECT=camel PYTHONPATH=~/KG-Commit/kgcommit_repro
export KGC_NEO4J_CONTAINER=neo4j-kgc KGC_NEO4J_IMAGE=neo4j:5
echo "=== KG METHODS RQ (retry, KGE clip fix) ==="; date
python3 -u inference/run_kg_methods_rq.py || { echo "STILL FAILED"; exit 1; }
echo "=== REPORTS ==="; date
python3 -u drivers/make_reports.py --no-notebooks
echo "=== RETRY DONE ==="; date
