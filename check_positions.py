"""Quick check: what position info does javalang give per node type?"""
import subprocess, sys, json
from pathlib import Path

REPO = Path("repos/apache/groovy")
BASE = "408b29851d7bbe4d343340832297e4be7e0c5578"
FILE = "src/main/org/codehaus/groovy/ant/Groovyc.java"

script = r"""
import javalang, sys, json

src = open(sys.argv[1], encoding='utf-8', errors='replace').read()
tree = javalang.parse.parse(src)

samples = {}
for path, node in tree:
    t = type(node).__name__
    if t not in samples:
        pos = getattr(node, 'position', None)
        raw_val = getattr(node, 'value', None) or getattr(node, 'name', None)
        samples[t] = {
            'has_position': pos is not None,
            'pos': [int(pos.line), int(pos.column)] if pos else None,
            'value': str(raw_val)[:40] if raw_val is not None else ''
        }
    if len(samples) >= 35:
        break

print(json.dumps(samples))
"""

src = subprocess.run(
    ["git", "-C", str(REPO), "show", f"{BASE}:{FILE}"],
    capture_output=True
).stdout

tmp = Path("outputs/_pos_test.java")
tmp.parent.mkdir(exist_ok=True)
tmp.write_bytes(src)

result = subprocess.run(
    [sys.executable, "-c", script, str(tmp)],
    capture_output=True, text=True, timeout=30
)
tmp.unlink()

if result.returncode == 0:
    data = json.loads(result.stdout)
    with_pos    = [(t, v) for t, v in data.items() if v['has_position']]
    without_pos = [t for t, v in data.items() if not v['has_position']]
    print(f"Nodes WITH position ({len(with_pos)}):")
    for t, v in sorted(with_pos):
        print(f"  {t:40s}  line={v['pos'][0]:4d}  col={v['pos'][1]:3d}  value={v['value']}")
    print(f"\nNodes WITHOUT position ({len(without_pos)}):")
    for t in sorted(without_pos):
        print(f"  {t}")
else:
    print("ERROR:", result.stderr[:800])
