## Setting Up Neo4j (Windows)

### Step 1: Install WSL 2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

Restart your computer if prompted.

To verify the installation:

```powershell
wsl --status
```
---

### Step 2: Install Docker Desktop

1. Download Docker Desktop from Docker's official website.
2. Run the installer and complete the installation.
3. During installation, ensure **Use WSL 2 instead of Hyper-V** is enabled.
4. Launch Docker Desktop after installation.
5. Wait until Docker reports that the engine is running.

Verify the installation:

```powershell
docker --version
```

---

### Step 3: Start a Local Neo4j Instance

Open PowerShell and run:

```powershell
docker run ^
    --name neo4j-local ^
    -p 7474:7474 -p 7687:7687 ^
    -d ^
    -v "%USERPROFILE%\neo4j\data:/data" ^
    -v "%USERPROFILE%\neo4j\logs:/logs" ^
    --env NEO4J_AUTH=neo4j/password1234 ^
    neo4j:latest
```

You can verify that the container is running:

```powershell
docker ps
```

You should see a container named `neo4j-local`.

---

### Step 4: Access Neo4j

Open your browser and navigate to:

```
http://localhost:7474
```

Login with:

* **Username:** `neo4j`
* **Password:** `password1234`

If the login page appears, your Neo4j instance is ready to use.

---

### Common Docker Commands

Stop Neo4j:

```powershell
docker stop neo4j-local
```

Start Neo4j again:

```powershell
docker start neo4j-local
```

View logs:

```powershell
docker logs neo4j-local
```

Remove the container:

```powershell
docker rm -f neo4j-local
```

## Usage

The ingestion pipeline is composed of three independent layers:

1. **CommitDataLoader** retrieves raw commit data from repositories.
2. **CommitParser** transforms or filters commit payloads into a desired schema.
3. **JITCommitKnowledgeGraph** consumes parsed commits and persists them into Neo4j.

### Architecture

```text
CommitDataLoader
       │
       ▼
 Raw Commit Stream
       │
       ▼
 CommitParser
       │
       ▼
 Parsed Commit Stream
       │
       ▼
JITCommitKnowledgeGraph
       │
       ▼
     Neo4j
```

### Initialization

Create a data loader, parser, and knowledge graph instance:

```python
from jitdp.data.dataloader import CommitDataLoader
from jitdp.graph.graph import JITCommitKnowledgeGraph
from jitdp.graph.parsers import FilteredCommitParser

data_loader = CommitDataLoader(repo_map=REPO_MAP)

# Select the parser schema used during ingestion
parser_instance = FilteredCommitParser()

kg = JITCommitKnowledgeGraph(
    uri=NEO4J_URI,
    auth_user=NEO4J_USER,
    auth_pass=NEO4J_PASSWORD,
    parser=parser_instance
)
```

### Parser Layer

Parsers define how raw commit payloads are transformed before entering the graph.

#### IdentityCommitParser

Passes the original commit payload through unchanged:

```python
parser = IdentityCommitParser()
```

#### FilteredCommitParser

Keeps only a selected subset of fields:

```python
parser = FilteredCommitParser(
    allowed_keys=[
        "commit_id",
        "project",
        "author_name",
        "committed_datetime",
        "files_modified_list",
    ]
)
```

If no schema is provided, the parser uses the framework's default knowledge graph schema.

### Ingesting Commits

The data loader produces a generator of raw commits. This stream can be passed directly into the knowledge graph ingestion engine.

```python
# A. Initialize the fast underlying process stream generator
raw_stream = data_loader.fetch_all_commits_fast(
    project=project_key,
    limit=-1
)

# B. Optionally wrap the stream with tracking or telemetry
tracked_stream = make_tracked_stream(
    raw_stream,
    project_label=project_short_name
)

# C. Ingest into Neo4j
actual_ingested = kg.ingest_fast(tracked_stream)

print(f"Ingested {actual_ingested:,} commits")
```

During ingestion:

1. The `CommitDataLoader` yields raw commit dictionaries.
2. The configured parser processes each commit.
3. `JITCommitKnowledgeGraph` converts parsed entities into graph nodes and relationships.
4. The resulting structure is persisted to Neo4j using a reusable session engine.

### Example Notebook

A complete end-to-end example is available in:

```text
neo4j.ipynb
```

The notebook demonstrates repository loading, parser configuration, graph ingestion, and Neo4j querying workflows.

