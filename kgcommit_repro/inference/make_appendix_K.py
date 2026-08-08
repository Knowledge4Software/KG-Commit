"""
Appendix K: data-curation table -- per project, the number of labelled commits in
the ApacheJIT dataset and which repository/repositories were used to build the graph
(single-repo vs the two multi-repo Hadoop spin-offs).
Sources: config/projects.yaml, data/apachejit/projects/apache_<p>.csv (row count).
"""
from pathlib import Path
import csv

import _kgc_paths  # noqa: F401
from config.project_config import _load_registry as load_registry  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
PM = ROOT / "Paper" / "paper_material" / "appendices" / "K_data_curation"
DATA = ROOT / "data" / "apachejit" / "projects"

DISP = {"activemq": "ActiveMQ", "groovy": "Groovy", "hadoop-hdfs": "HDFS",
        "hadoop-mapreduce": "MapReduce", "kafka": "Kafka", "spark": "Spark",
        "zeppelin": "Zeppelin", "zookeeper": "Zookeeper", "cassandra": "Cassandra"}
# active paper set (hdfs/mapreduce dropped from the paper; still C3-recoverable)
ORDER = ["activemq", "cassandra", "groovy", "kafka", "spark", "zeppelin", "zookeeper"]


def n_commits(proj):
    f = DATA / f"apache_{proj}.csv"
    if not f.exists():
        return None
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh) - 1


def main():
    PM.mkdir(parents=True, exist_ok=True)
    try:
        reg = load_registry().get("projects", {})
    except Exception:
        reg = {}
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{Appendix K: data curation. Labelled commits per project and the "
         r"repositories used to build the knowledge graph. All projects in the paper's "
         r"active set are single-repository.}",
         r"\label{tab:data_curation}",
         r"\begin{tabular}{llrl}", r"\toprule",
         r"Project & ApacheJIT key & \#Labelled commits & Repositories used \\ \midrule"]
    for p in ORDER:
        n = n_commits(p)
        entry = reg.get(p, {}) if isinstance(reg, dict) else {}
        primary = entry.get("repo_subpath", f"repos/apache/{p}")
        extra = entry.get("extra_repo_subpaths") or []
        repos = primary.split("/")[-1]
        if extra:
            repos += " + " + ", ".join(e.split("/")[-1] for e in extra) + " (multi-repo)"
        L.append(f"{DISP.get(p, p)} & apache/{p} & {n if n is not None else '--':,} & "
                 f"\\texttt{{{repos}}} \\\\" if isinstance(n, int)
                 else f"{DISP.get(p, p)} & apache/{p} & -- & \\texttt{{{repos}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PM / "appK_data_curation.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote appK_data_curation.tex -> {PM}")


if __name__ == "__main__":
    main()
