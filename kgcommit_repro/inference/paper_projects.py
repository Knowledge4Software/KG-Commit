"""
Single source of truth for the ACTIVE paper project set used by all make_* artifact
generators. Edit here to add/drop a project everywhere at once.

hadoop-hdfs and hadoop-mapreduce are the two multi-repository projects; they are
DROPPED from the paper (editorial decision, re-dropped 2026-07-25) but their Neo4j
dumps, result pickles, and full outputs/ artifacts are retained, so they are fully
C3-recoverable on demand -- to re-include them, move them from DROPPED back into
ACTIVE and re-run the generators.

Drop rationale (2026-07-25): both are multi-repository (labels span a spin-off repo
+ the hadoop monorepo), which yields extreme class imbalance and degenerate metrics
on several graphs (e.g. G-Mean 0 on MapReduce structural layers). Including them
materially changed the headline performance averages and the per-project artifacts
had recurring issues. Deferred open decision (see docs/Critical_notes.tex): after
all other projects are settled, either (a) drop them permanently WITH a documented
justification, or (b) fix the multi-repo issues and include them in the final paper.
"""

# (display name, output-folder name)
ACTIVE = [
    ("ActiveMQ", "activemq"),
    ("Cassandra", "cassandra"),
    ("Groovy", "groovy"),
    ("Kafka", "kafka"),
    ("Spark", "spark"),
    ("Zeppelin", "zeppelin"),
    ("Zookeeper", "zookeeper"),
]

# retained on disk (dumps + pickles + outputs/) but excluded from the paper. The two
# multi-repository projects have extreme class imbalance and degenerate metrics that
# skewed the aggregates; dropped 2026-07-25 pending the keep/fix decision above.
DROPPED = [
    ("HDFS", "hadoop-hdfs"),
    ("MapReduce", "hadoop-mapreduce"),
]

# convenience: all folders that exist on disk (for scripts that want everything)
ALL = ACTIVE + DROPPED
