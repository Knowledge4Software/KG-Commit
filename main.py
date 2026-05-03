from __future__ import annotations
import argparse
from pathlib import Path
from kg_commit.data.dataset import CSVCommitDataset
from kg_commit.data.preprocess import SimplePreprocessor
from kg_commit.data.stream import CommitStream
from kg_commit.evaluation.evaluator import Evaluator
from kg_commit.evaluation.report import SimpleReport
from kg_commit.model.online_model import OnlineModel
from kg_commit.training.incremental_trainer import IncrementalTrainer
from kg_commit.utils.config import Config
from kg_commit.persistence.serializer import JSONSerializer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a commit-level incremental defect prediction experiment.")
    parser.add_argument(
        "--config",
        default="experiments/configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to the commit dataset CSV file (overrides config).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Name of the label column in the dataset (overrides config).",
    )
    parser.add_argument(
        "--timestamp-column",
        default=None,
        help="Name of the timestamp column in the dataset (overrides config).",
    )
    parser.add_argument("--window-size", type=int, default=None, help="Number of commits in each window (overrides config).")
    parser.add_argument("--window-step", type=int, default=None, help="Number of commits to advance between windows (overrides config).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config from YAML
    config = Config.from_yaml(args.config)

    # Get parameters from config, with CLI args taking precedence
    data_path = args.data_path or config.get("data.path")
    label_column = args.label_column or config.get("preprocessing.label_column")
    timestamp_column = args.timestamp_column or config.get("data.timestamp_column")
    window_size = args.window_size or config.get("streaming.window_size")
    window_step = args.window_step or config.get("streaming.window_step")
    output_dir = Path(config.get("output.results_dir", "outputs/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = CSVCommitDataset(
        source=data_path,
        label_column=label_column,
        timestamp_column=timestamp_column,
    )
    commits = dataset.load()
    print(f"Loaded {len(commits)} commits from {data_path}")

    # Create stream, preprocessor, model, and trainer
    stream = CommitStream(commits, window_size=window_size, step=window_step)
    preprocessor = SimplePreprocessor(label_column=label_column)
    model_classes = config.get("model.classes", [0, 1])
    model = OnlineModel(classes=model_classes)
    evaluator = Evaluator()
    trainer = IncrementalTrainer(model=model, preprocessor=preprocessor, evaluator=evaluator)

    # Run experiment
    print(f"Running experiment with window_size={window_size}, step={window_step}")
    for index, (window, preds) in enumerate(trainer.run(stream), start=1):
        if index % 10 == 0:
            print(f"Processed {index} windows")

    # Print final summary
    summary = evaluator.summarize()
    print("\n" + "="*50)
    print("Experiment Summary")
    print("="*50)
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall: {summary['recall']:.4f}")
    print(f"F1 Score: {summary['f1']:.4f}")
    print("\nClassification Report:")
    print(summary['report'])

    # Save results
    results_data = {
        "experiment_config": args.config,
        "data_path": str(data_path),
        "window_size": window_size,
        "window_step": window_step,
        "summary": summary
    }
    serializer = JSONSerializer()
    results_file = output_dir / "results.json"
    serializer.save(results_data, str(results_file))
    print(f"\nResults saved to {results_file}")
    print(summary["report"])


if __name__ == "__main__":
    main()