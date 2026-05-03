import csv
from pathlib import Path
from kg_commit.data.dataset import CSVCommitDataset


def test_csv_commit_dataset_loads_records_sorted(tmp_path):
    path = tmp_path / "commits.csv"
    rows = [
        {"commit_time": "3", "buggy": "0", "files_changed": "2"},
        {"commit_time": "1", "buggy": "1", "files_changed": "5"},
        {"commit_time": "2", "buggy": "0", "files_changed": "1"},
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["commit_time", "buggy", "files_changed"])
        writer.writeheader()
        writer.writerows(rows)

    dataset = CSVCommitDataset(source=path)
    records = dataset.load()

    assert len(records) == 3
    assert records[0]["commit_time"] == 1.0
    assert records[-1]["commit_time"] == 3.0
    assert dataset.get_metadata()["rows"] == 3
    assert "buggy" in dataset.get_metadata()["columns"]


def test_csv_commit_dataset_files_not_found_raises(tmp_path):
    missing = tmp_path / "missing.csv"
    dataset = CSVCommitDataset(source=missing)

    try:
        dataset.load()
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
