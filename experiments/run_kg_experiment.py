"""
Vanilla KG Experiment
---------------------
Runs two back-to-back experiments on the same dataset:
  1. Baseline  – standard handcrafted features (la, ld, nf, nd, ns, ent, ...)
  2. KG        – same features + 5 knowledge-graph features derived from
                 project-level commit history

Prints a side-by-side comparison table and saves both result JSONs to
outputs/results/.

Usage:
    python experiments/run_kg_experiment.py
    python experiments/run_kg_experiment.py --data data/apachejit/apachejit_train.csv
    python experiments/run_kg_experiment.py --window-size 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_commit.data.dataset import CSVCommitDataset
from kg_commit.data.preprocess import SimplePreprocessor
from kg_commit.data.stream import CommitStream
from kg_commit.evaluation.evaluator import Evaluator
from kg_commit.knowledge import CommitKnowledgeGraph, KGPreprocessor
from kg_commit.model.online_model import OnlineModel
from kg_commit.persistence.serializer import JSONSerializer
from kg_commit.training.incremental_trainer import IncrementalTrainer
from kg_commit.utils.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_pipeline(commits, window_size, window_step, use_kg: bool):
    """Build and run one full incremental pipeline. Returns the evaluator."""
    stream = CommitStream(commits, window_size=window_size, step=window_step)

    if use_kg:
        graph = CommitKnowledgeGraph(recent_window=10)
        preprocessor = KGPreprocessor(graph=graph, label_column="buggy")
    else:
        graph = None
        preprocessor = SimplePreprocessor(label_column="buggy")

    model = OnlineModel(classes=[0, 1])
    evaluator = Evaluator()
    trainer = IncrementalTrainer(
        model=model,
        preprocessor=preprocessor,
        evaluator=evaluator,
        graph=graph,
    )

    n_windows = 0
    for _window, _preds in trainer.run(stream):
        n_windows += 1

    print(f"  -> processed {n_windows} windows")
    return evaluator


def print_comparison(baseline: dict, kg: dict):
    metrics = ["accuracy", "precision", "recall", "f1"]
    col = 12
    header = f"{'Metric':<14}" + f"{'Baseline':>{col}}" + f"{'KG':>{col}}" + f"{'Delta':>{col}}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        b = baseline.get(m, 0.0)
        k = kg.get(m, 0.0)
        delta = k - b
        sign = "+" if delta >= 0 else ""
        print(f"{m:<14}{b:>{col}.4f}{k:>{col}.4f}{sign + f'{delta:.4f}':>{col}}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline vs KG-augmented JIT defect prediction.")
    parser.add_argument("--data", default="data/apachejit/apachejit_train.csv",
                        help="Path to dataset CSV (default: apachejit_train.csv)")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-step", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/results",
                        help="Directory to save result JSONs")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data once; both runs share the same commit list
    # ------------------------------------------------------------------
    print(f"Loading dataset: {data_path}")
    dataset = CSVCommitDataset(source=data_path, label_column="buggy", timestamp_column="author_date")
    commits = dataset.load()
    print(f"Loaded {len(commits)} commits  |  window_size={args.window_size}  step={args.window_step}\n")

    # ------------------------------------------------------------------
    # Run 1: Baseline
    # ------------------------------------------------------------------
    print("[ 1 / 2 ]  Baseline (handcrafted features only)")
    evaluator_base = run_pipeline(commits, args.window_size, args.window_step, use_kg=False)
    summary_base = evaluator_base.summarize()

    # ------------------------------------------------------------------
    # Run 2: KG-augmented
    # ------------------------------------------------------------------
    print("\n[ 2 / 2 ]  KG-augmented (+ project history features)")
    evaluator_kg = run_pipeline(commits, args.window_size, args.window_step, use_kg=True)
    summary_kg = evaluator_kg.summarize()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print_comparison(summary_base, summary_kg)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    serializer = JSONSerializer()

    base_file = output_dir / "baseline_results.json"
    kg_file = output_dir / "kg_results.json"

    serializer.save({"mode": "baseline", "config": vars(args), "summary": summary_base}, str(base_file))
    serializer.save({"mode": "kg", "config": vars(args), "summary": summary_kg}, str(kg_file))

    print(f"\nResults saved:")
    print(f"  Baseline -> {base_file}")
    print(f"  KG       -> {kg_file}")


if __name__ == "__main__":
    main()
