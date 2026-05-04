from __future__ import annotations
from pathlib import Path
from kg_commit.utils.config import Config, PROJECT_ROOT
from kg_commit.data.dataset import CSVCommitDataset
from kg_commit.data.preprocess import SimplePreprocessor
from kg_commit.data.stream import CommitStream
from kg_commit.evaluation.evaluator import Evaluator
from kg_commit.evaluation.report import SimpleReport
from kg_commit.model.online_model import OnlineModel
from kg_commit.training.incremental_trainer import IncrementalTrainer
from kg_commit.persistence.serializer import JSONSerializer
from kg_commit.utils.logging import SimpleLogger


class Experiment:
    """Configurable experiment runner using YAML config."""

    def __init__(self, config_path: str | Path):
        self.config = Config.from_yaml(config_path)
        self.logger = SimpleLogger()
        self.dataset = None
        self.model = None
        self.preprocessor = None
        self.stream = None
        self.evaluator = None
        self.trainer = None
        self.results = []

    def initialize(self):
        """Initialize all components from config."""
        # Load data
        data_path = self.config.get("data.path")
        self.dataset = CSVCommitDataset(
            source=data_path,
            label_column=self.config.get("preprocessing.label_column"),
            timestamp_column=self.config.get("data.timestamp_column")
        )
        commits = self.dataset.load()
        self.logger.log_metrics({"loaded_commits": len(commits)})

        # Create stream
        window_size = self.config.get("streaming.window_size")
        window_step = self.config.get("streaming.window_step")
        self.stream = CommitStream(commits, window_size=window_size, step=window_step)

        # Create preprocessor
        self.preprocessor = SimplePreprocessor(
            label_column=self.config.get("preprocessing.label_column"),
            include_text=self.config.get("preprocessing.include_text", True)
        )

        # Create model
        classes = self.config.get("model.classes", [0, 1])
        self.model = OnlineModel(classes=classes)

        # Create evaluator
        self.evaluator = Evaluator()

        # Create trainer
        self.trainer = IncrementalTrainer(
            model=self.model,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator
        )

    def run(self):
        """Run the experiment."""
        if self.trainer is None:
            self.initialize()

        for window, preds in self.trainer.run(self.stream):
            self.results.append((window, preds))

    def save_results(self):
        """Save results to file."""
        output_dir = Path(self.config.get("output.results_dir", "outputs/results"))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.evaluator.summarize()
        results_data = {
            "experiment_name": self.config.get("experiment.name"),
            "description": self.config.get("experiment.description"),
            "config_path": str(self.config),
            "num_windows": len(self.results),
            "summary": summary
        }

        serializer = JSONSerializer()
        results_file = output_dir / "results.json"
        serializer.save(results_data, str(results_file))
        return results_file

    def get_summary(self):
        """Get evaluation summary."""
        return self.evaluator.summarize()