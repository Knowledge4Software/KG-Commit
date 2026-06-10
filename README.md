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
