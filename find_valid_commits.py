"""
Find commits where the BEFORE file content == BASE_COMMIT content.
These are the first commits (after BASE_COMMIT) that touched each Java file,
where the parent state still matches BASE_COMMIT exactly.
"""
import subprocess
from pathlib import Path
from neo4j import GraphDatabase

REPO      = Path("repos/apache/groovy")
BASE      = "408b29851d7bbe4d343340832297e4be7e0c5578"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password1234")

def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True)

def file_blob(commit, path):
    """Return the raw bytes of a file at a commit, or None."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
with driver.session() as s:
    files = [r["file"] for r in s.run(
        "MATCH (f:File)-[:HAS_AST]->() RETURN DISTINCT f.id AS file ORDER BY f.id"
    ).data()]
driver.close()

print(f"Files with ASTNodes in Neo4j: {len(files)}\n")
print(f"Finding first-touch commits after BASE_COMMIT where parent state = BASE...\n")

valid = []  # (commit_id, file)
base_blobs = {}  # cache

for file in files:
    # First commit after BASE that modified this file (oldest = last of --reverse)
    r = git("log", "--reverse", "--format=%H", f"{BASE}..HEAD", "--", file)
    lines = r.stdout.strip().splitlines()
    if not lines:
        continue
    first_commit = lines[0]  # oldest commit that touched this file after BASE

    # Cache BASE blob
    if file not in base_blobs:
        base_blobs[file] = file_blob(BASE, file)
    base_blob = base_blobs[file]
    if base_blob is None:
        continue

    # Check parent's content == BASE content
    parent_blob = file_blob(f"{first_commit}^", file)
    if parent_blob is None:
        continue

    if parent_blob == base_blob:
        valid.append((first_commit, file))
        print(f"  VALID  {first_commit[:8]}  {file.split('/')[-1]}")
    else:
        pass  # parent differs from BASE — skip

print(f"\nFound {len(valid)} valid (commit, file) pairs where parent == BASE")
# Save for use in build_delta_graph.py
import json
Path("outputs/valid_delta_commits.json").write_text(json.dumps(valid, indent=2))
print("Saved to outputs/valid_delta_commits.json")
