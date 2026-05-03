from __future__ import annotations
import argparse
from kg_commit.data.dataset import CSVCommitDataset
from kg_commit.data.preprocess import SimplePreprocessor
from kg_commit.data.stream import CommitStream
from kg_commit.evaluation.evaluator import Evaluator
from kg_commit.model.online_model import OnlineModel
from kg_commit.training.incremental_trainer import IncrementalTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a commit-level incremental defect prediction experiment.")
    parser.add_argument(
        "--data-path",
        default="data/apachejit/apachejit_train.csv",
        help="Path to the commit dataset CSV file.",
    )
    parser.add_argument(
        "--label-column",
        default="buggy",
        help="Name of the label column in the dataset.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="commit_time",
        help="Name of the timestamp column in the dataset.",
    )
    parser.add_argument("--window-size", type=int, default=50, help="Number of commits in each window.")
    parser.add_argument("--window-step", type=int, default=50, help="Number of commits to advance between windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = CSVCommitDataset(
        source=args.data_path,
        label_column=args.label_column,
        timestamp_column=args.timestamp_column,
    )
    commits = dataset.load()

    stream = CommitStream(commits, window_size=args.window_size, step=args.window_step)
    preprocessor = SimplePreprocessor(label_column=args.label_column)
    model = OnlineModel(classes=[0, 1])
    evaluator = Evaluator()
    trainer = IncrementalTrainer(model=model, preprocessor=preprocessor, evaluator=evaluator)

    for index, (window, preds) in enumerate(trainer.run(stream), start=1):
        print(f"Window {index}: size={window.size()}, accuracy so far not computed")

    summary = evaluator.summarize()
    print("\nExperiment summary")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(summary["report"])


if __name__ == "__main__":
    main()