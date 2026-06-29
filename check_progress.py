"""Quick progress check for the online KG build. Run: python check_progress.py"""
import json, time
from pathlib import Path

CK = Path("outputs/online_kg_checkpoint.json")

def read():
    return json.loads(CK.read_text())["next_index"]

a = read()
print(f"Progress: {a} / 8059  ({100*a//8059}%)   files tracked: "
      f"{len(json.loads(CK.read_text())['file_state'])}")
# is it advancing? sample twice
time.sleep(8)
b = read()
if b > a:
    rate = (b - a) / 8.0
    remain = (8059 - b) / rate if rate else 0
    print(f"RUNNING — {rate:.1f} commits/sec, ~{remain/60:.0f} min left")
else:
    print("NOT advancing — rerun 'python build_online_kg.py' to resume from checkpoint")
