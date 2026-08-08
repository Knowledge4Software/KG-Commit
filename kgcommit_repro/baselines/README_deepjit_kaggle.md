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

You can put all five projects' files in one Dataset — only the per-run `PROJECT`
file is read. Files:
`apache_zookeeper*.csv`, `apache_zeppelin*.csv`, `apache_kafka*.csv`,
`apache_activemq*.csv`, `apache_groovy*.csv`.

## 2. Notebook setup
1. New Notebook → **Add Data** → your `apachejit-diffs` Dataset.
2. **Settings → Accelerator → GPU T4**.
3. Import `deepjit_kaggle.ipynb` (or paste `deepjit_kaggle.py` into one cell).
4. In the **CONFIG** cell set:
   - `PROJECT = "zookeeper"` (change per run)
   - `DATA_DIR = "/kaggle/input/apachejit-diffs"` (match your dataset's mount path —
     check the right-hand "Input" panel for the exact folder name).
5. **Run All**.

## 3. Run order (small → large)
Run one project per notebook execution, in this order:

```
zookeeper  →  zeppelin  →  kafka  →  activemq  →  groovy
```

Faithful online refit (retrain every 3 blocks on the expanding past) means bigger
projects take longer; each is well within a single Kaggle GPU session.

## 4. Collect results
Each run writes `deepjit_<project>_results.pkl` to `/kaggle/working/`. Download it
and place it here locally:

```
outputs/<project>/deepjit_<project>_results.pkl
```

Then regenerate the baseline document — `make_baselines_doc.py` auto-detects DeepJIT
and adds it to the per-project tables, the aggregate, and the KG-vs-all arrow table.

## Protocol parity (do not change)
The notebook hard-codes the same constants as the repo so results are comparable:
`WARMUP_FRAC=0.40`, `BLOCK=200`, refit every 3 blocks, online-tuned threshold
(`init=300, step=150`, maximise buggy-F1 on the past), and the identical 7-metric
suite. The metric functions were byte-verified to match `inference/online_jit.py`
to 1e-9. Changing these breaks comparability with KG-Commit and the other baselines.
