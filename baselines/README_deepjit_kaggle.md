# DeepJIT baseline on Kaggle (T4) — run instructions

DeepJIT (Hoang et al., ICSE 2019) evaluated under the **exact** KG-Commit online
prequential protocol, so its numbers drop straight into the paper's baseline tables.
No local GPU needed; runs on a free Kaggle T4. Each project's input is 18–74 MB, well
under Kaggle's 200 MB/file limit.

## 1. Build the Kaggle Dataset
Upload these two files **per project** to a Kaggle Dataset (e.g. named
`apachejit-diffs`):

```
apache_<project>.csv          # from data/apachejit/projects/
apache_<project>_diff.csv     # from data/apachejit/projects/  (has diff_text)
```

You can put several projects' files in one Dataset — only the per-run `PROJECT`
file is read. The paper evaluates all **11** ApacheJIT projects (Table 4):
`zookeeper`, `zeppelin`, `spark`, `kafka`, `activemq`, `hive`, `groovy`,
`cassandra`, `hbase`, `flink`, `camel`.

## 2. Notebook setup
1. New Notebook → **Add Data** → your `apachejit-diffs` Dataset.
2. **Settings → Accelerator → GPU T4**.
3. Paste `deepjit_kaggle.py` into a single notebook cell.
4. In the **CONFIG** cell set:
   - `PROJECT = "zookeeper"` (change per run)
   - `DATA_DIR = "/kaggle/input/apachejit-diffs"` (match your dataset's mount path —
     check the right-hand "Input" panel for the exact folder name).
5. **Run All**.

## 3. Run order (small → large)
Run one project per notebook execution, light-first:

```
zookeeper → zeppelin → spark → kafka → activemq → hive
          → groovy → cassandra → hbase → flink → camel
```

Faithful online refit (retraining once per block on the expanding past) means the
larger projects take substantially longer; start with `zookeeper` to validate the
setup before committing GPU time to `flink` or `camel`.

## 4. Collect results
Each run writes `deepjit_<project>_results.pkl` to `/kaggle/working/`. Download it
and place it here locally:

```
outputs/<project>/deepjit_<project>_results.pkl
```

The pickle holds DeepJIT's scored stream and its metrics under the same 7-metric suite
as every other baseline, so its numbers are directly comparable with the locally-run
baselines produced by `baselines/run_extra_baselines.py`.

> DeepJIT is scored **out-of-band**: it runs on Kaggle rather than through the local
> drivers, so the local reporting pipeline does not read this pickle automatically.
> Load it directly (`pickle.load`) to place DeepJIT beside the other baselines.

## Protocol parity (do not change)
The script mirrors the constants defined in `inference/protocol.py` so results are
directly comparable with KG-Commit and the other baselines:

| Constant | Value | |
|---|---|---|
| `WARMUP_FRAC` (K) | `0.05` | warm-up window; these commits are not scored |
| `BLOCK` (M) | `200` | predict-then-learn granularity |
| `GAP` (g) | `50` | commits skipped between fitting and scoring |
| `REFIT_EVERY` | `1` | faithful online refit: retrain once per block |

It also uses the same leakage-free online-tuned threshold (`init=300, step=150`,
maximising buggy-F1 on the past) and the identical 7-metric suite, with trajectories
at window 150 and 800 so the stream figures line up. The metric functions were
verified to match `inference/online_jit.py` to 1e-9.

Because this script runs on Kaggle it cannot import `protocol.py`, so these values are
restated in its CONFIG block. **If `protocol.py` ever changes, update this script to
match** — otherwise the DeepJIT column is no longer comparable.

> Note: the module docstring at the top of `deepjit_kaggle.py` still describes an
> earlier configuration (`WARMUP_FRAC = 0.40`, refit every 3 blocks). The executable
> CONFIG block is authoritative and uses the values in the table above.
