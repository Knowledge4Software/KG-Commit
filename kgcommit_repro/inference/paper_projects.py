"""
Single source of truth for the ACTIVE paper project set used by all make_* artifact
generators. Edit here to add/drop a project everywhere at once.

hadoop-hdfs and hadoop-mapreduce are the two multi-repository projects; they are
currently DROPPED from the paper (editorial decision) but their Neo4j dumps and
result pickles are retained, so they are fully C3-recoverable on demand -- to
re-include them, move them from DROPPED back into ACTIVE and re-run the generators.
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
    ("HDFS", "hadoop-hdfs"),
    ("MapReduce", "hadoop-mapreduce"),
]

# retained on disk (dumps + pickles); re-included into ACTIVE above. The two
# multi-repository projects have extreme class imbalance, so their per-project
# metrics read weaker than the single-repo set -- expected, not an error.
DROPPED = []

# convenience: all folders that exist on disk (for scripts that want everything)
ALL = ACTIVE + DROPPED
