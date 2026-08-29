cd ~/KG-Commit/kgcommit_repro
source .venv/bin/activate
export KGC_PROJECT=groovy
export PYTHONPATH=~/KG-Commit/kgcommit_repro
echo "python : $(which python3)"
echo "project: $KGC_PROJECT"
export KGC_NEO4J_CONTAINER=neo4j-kgc
export KGC_NEO4J_IMAGE=neo4j:5
