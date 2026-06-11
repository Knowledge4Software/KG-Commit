import os
import sys
import json
from pathlib import Path
import random
import inspect
from pprint import pprint

from dotenv import load_dotenv, find_dotenv
# 1. Locate and load the environment variables from the .env configuration file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# 2. Extract configuration variables from the environment
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
REPOS_DIR = os.getenv("REPOS_DIR")
DATA_DIR = os.getenv("DATA_DIR")

project_root = Path(PROJECT_ROOT).resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import your newly structured module
from kg_commit.knowledge.dataloader import CommitDataLoader, JITDatasetAdapter
from kg_commit.knowledge.parsers import FilteredCommitParser, IdentityCommitParser
from kg_commit.knowledge.utils import CommitPayloadPrinter
from kg_commit.knowledge.graph import JITCommitKnowledgeGraph

def generate_repo_map(base_dir: str, prefix: str = "apache/") -> dict[str, str]:
    """
    Scans a base directory for valid git repositories and constructs
    a REPO_MAP dictionary compatible with the CommitDataLoader mapping schema.
    """
    repo_map = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"⚠️ Warning: Base directory '{base_dir}' does not exist.")
        return repo_map

    # Iterate through all direct items in the repos folder
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it contains a hidden .git directory to verify it's a real repo
            git_dir = item / ".git"
            if git_dir.exists():
                # Reconstruct the project key name (e.g., "apache/groovy")
                project_key = f"{prefix}{item.name.lower()}"
                
                # Assign the absolute string path as the value
                repo_map[project_key] = str(item.resolve())
                
    return repo_map

# Run the auto-generation mapping
generated_map = generate_repo_map(REPOS_DIR, prefix="apache/")

TARGET_PROJECTS = [
    # 'apache/activemq',
    'apache/camel',
    'apache/cassandra',
    'apache/flink',
    'apache/groovy',
    'apache/hadoop',
    'apache/hadoop-hdfs',
    'apache/hadoop-mapreduce',
    'apache/hbase',
    'apache/hive',
    'apache/ignite',
    'apache/kafka',
    'apache/spark',
    'apache/zeppelin',
    'apache/zookeeper',
]

REPO_MAP = {
    project: path 
    for project, path in generated_map.items() 
    if project in TARGET_PROJECTS
}

# Print out your freshly discovered mappings
print("📂 Automatically generated REPO_MAP mappings:")
print("-" * 50)
for project, local_path in REPO_MAP.items():
    print(f"  '{project}'")
print("-" * 50)

# Neo4j Local Instance Credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password1234"

# =====================================================================
# INITIALIZATION
# =====================================================================
data_loader = CommitDataLoader(repo_map=REPO_MAP)

# Instantiate the structural graph engine using our default parser schema
parser_instance = FilteredCommitParser()
kg = JITCommitKnowledgeGraph(
    uri=NEO4J_URI,
    auth_user=NEO4J_USER,
    auth_pass=NEO4J_PASSWORD,
    parser=parser_instance
)

from tqdm import tqdm
import time

# 1. Wipe historical graph database state for a fresh chronological run
print("Wiping historical graph database state...")
counters = kg.purge_database()
print(f"Database wiped successfully. Nodes/Relationships removed: {counters}\n")

from typing import Iterable, Any, Dict, Generator
from tqdm import tqdm

def make_tracked_stream(
    commit_stream: Iterable[Dict[str, Any]], 
    project_label: str
) -> Generator[Dict[str, Any], None, None]:
    """
    Wraps a flat raw commit stream generator with a clean, dynamic tqdm progress bar.
    Tracks throughput and updates terminal UI frames cleanly on-the-fly.
    """
    with tqdm(
        desc=f" [{project_label}] Ingest", 
        unit=" commit", 
        dynamic_ncols=True
    ) as pbar:
        
        for current_count, parsed_commit in enumerate(commit_stream, start=1):
            commit_id = parsed_commit.get("commit_id", "Unknown")
            
            # Update visual diagnostics side-cards
            pbar.set_description(f"Processing {commit_id[:8]}")
            pbar.set_postfix(total_ingested=current_count)
            pbar.update(1)
            
            yield parsed_commit

# Global portfolio metrics collection definitions
portfolio_ingest_stats = {}
global_pipeline_start = time.time()

print("🚀 COMMENCING PORTFOLIO-WIDE CHRONOLOGICAL INGESTION STREAMS...")
print("=" * 85)

# 2. Sequential execution over discovered repositories
for project_key in REPO_MAP.keys():
    print(f"\n⏳ Activating target stream: '{project_key}'")
    print("-" * 85)
    
    project_start_time = time.time()
    project_short_name = project_key.split('/')[-1]
    
    try:
        # A. Initialize the fast underlying process stream generator
        raw_stream = data_loader.fetch_all_commits_fast(project=project_key, limit=-1)
        
        # B. Intercept the generator stream with our tracking telemetry
        tracked_stream = make_tracked_stream(raw_stream, project_label=project_short_name)
        
        # C. Pipe the tracked stream directly into the reusable Neo4j session engine
        actual_ingested = kg.ingest_fast(tracked_stream)
        
        # D. Calculate throughput performance metrics
        project_duration = time.time() - project_start_time
        throughput = actual_ingested / project_duration if project_duration > 0 else 0
        
        # Save records for the portfolio-wide overview report
        portfolio_ingest_stats[project_key] = {
            "ingested": actual_ingested,
            "duration": project_duration,
            "throughput": throughput
        }
        
        print(f"✅ Finished '{project_key}': Ingested {actual_ingested} commits in {project_duration:.3f}s ({throughput:.2f} commits/sec)")
        print("-" * 85)
        
    except Exception as e:
        print(f"❌ Critical Pipeline Failure on project '{project_key}': {str(e)}")
        portfolio_ingest_stats[project_key] = {"ingested": 0, "duration": 0.0, "throughput": 0.0}
        continue

global_pipeline_duration = time.time() - global_pipeline_start

# =====================================================================
# FINAL PORTFOLIO INGESTION REPORT DASHBOARD
# =====================================================================
print("\n" + "=" * 90)
print("                    PORTFOLIO GRAPH INGESTION ANALYTICS SUMMARY                 ")
print("=" * 90)
print(f"{'PROJECT KEY':<25} | {'INGESTED COMMITS':<18} | {'TIME ELAPSED':<14} | {'INGEST SPEED (C/s)'}")
print("-" * 90)

grand_total_ingested = 0
for proj, metrics in portfolio_ingest_stats.items():
    grand_total_ingested += metrics["ingested"]
    print(f"{proj:<25} | {metrics['ingested']:<18} | {metrics['duration']:11.3f}s | {metrics['throughput']:.2f} commits/sec")

print("-" * 90)
print(f"Portfolio Summary metrics:")
print(f" ├── Total Projects Successfully Processed : {len(portfolio_ingest_stats)}")
print(f" ├── Grand Total Ingested Graph Commits    : {grand_total_ingested}")
print(f" └── Total Pipeline Execution Wall Time     : {global_pipeline_duration:.3f} seconds")
print("=" * 90)