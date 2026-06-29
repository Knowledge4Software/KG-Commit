"""Diagnose position mismatch between GumTree and javalang Neo4j nodes."""
import subprocess, json, os, sys, tempfile
from pathlib import Path
from neo4j import GraphDatabase

GT_LIB    = Path("tools/gumtree/gumtree-2.1.2/lib")
REPO      = Path("repos/apache/groovy")
BASE      = "408b29851d7bbe4d343340832297e4be7e0c5578"
FILE      = "src/main/org/codehaus/groovy/ast/MethodNode.java"

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","password1234"))

# 1. Get Neo4j positions for this file (only real-positioned nodes)
with driver.session() as s:
    neo4j = s.run("""
        MATCH (a:ASTNode {file: $f})
        WHERE a.pos_line > 0
        RETURN a.ast_type AS t, a.value AS v, a.pos_line AS line, a.pos_col AS col
        ORDER BY a.pos_line, a.pos_col
        LIMIT 20
    """, f=FILE).data()

print("Neo4j positions (javalang):")
for r in neo4j:
    print(f"  {r['t']:35s}  line={r['line']:4d}  col={r['col']:3d}  val={str(r['v'] or '')[:20]}")

# 2. Get GumTree positions for the same file
src_bytes = subprocess.run(
    ["git","-C",str(REPO),"show",f"{BASE}:{FILE}"], capture_output=True
).stdout
src_text = src_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")

with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
    f.write(src_text.encode("utf-8"))
    tmp = f.name

def build_post_order(root):
    nodes = []
    def walk(n):
        for c in n.get("children", []):
            walk(c)
        nodes.append(n)
    walk(root)
    return nodes

def offset_to_lc(text, offset):
    before = text[:offset]
    lines  = before.split("\n")
    return len(lines), len(lines[-1]) + 1

lib = str(GT_LIB / "*")
raw = subprocess.run(
    ["java","-cp",lib,"com.github.gumtreediff.client.Run",
     "parse","-g","java-javaparser", tmp],
    capture_output=True, text=True, timeout=30
).stderr + subprocess.run(
    ["java","-cp",lib,"com.github.gumtreediff.client.Run",
     "parse","-g","java-javaparser", tmp],
    capture_output=True, text=True, timeout=30
).stdout
os.unlink(tmp)

tree = json.loads(raw[raw.find("{"):])
post = build_post_order(tree["root"])

# Print first 20 named/interesting nodes (with a label or specific types)
interesting_types = {
    "MethodDeclaration","FieldDeclaration","ClassOrInterfaceDeclaration",
    "ConstructorDeclaration","VariableDeclarator","Parameter","SimpleName"
}
print(f"\nGumTree positions (JavaParser) — first interesting nodes:")
shown = 0
for i, n in enumerate(post):
    tl = n["typeLabel"]
    if tl in interesting_types or n.get("label",""):
        line, col = offset_to_lc(src_text, int(n["pos"]))
        print(f"  [{i:4d}] {tl:35s}  line={line:4d}  col={col:3d}  label={n.get('label','')[:20]}")
        shown += 1
        if shown >= 25:
            break

# 3. Compare: for "MethodDeclaration" nodes, what positions do both give?
print("\n── MethodDeclaration comparison ──")
neo4j_meth = {(r['line'], r['col']): r['v']
              for r in neo4j if r['t'] == 'MethodDeclaration'}
print(f"  Neo4j: {list(neo4j_meth.items())[:5]}")

# Find GumTree MethodDeclarations
tmp2 = tempfile.mktemp(suffix=".java")
src_bytes2 = subprocess.run(
    ["git","-C",str(REPO),"show",f"{BASE}:{FILE}"], capture_output=True
).stdout
Path(tmp2).write_bytes(src_text.encode("utf-8"))
raw2 = subprocess.run(
    ["java","-cp",lib,"com.github.gumtreediff.client.Run",
     "parse","-g","java-javaparser", tmp2],
    capture_output=True, text=True, timeout=30
).stderr
os.unlink(tmp2)
# already have post - use it
gt_meth = []
for n in post:
    if n["typeLabel"] == "MethodDeclaration":
        line, col = offset_to_lc(src_text, int(n["pos"]))
        gt_meth.append((line, col, n.get("label","")))
print(f"  GumTree MethodDeclaration positions: {gt_meth[:5]}")

driver.close()
