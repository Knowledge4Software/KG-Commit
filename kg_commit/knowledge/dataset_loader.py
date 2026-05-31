"""
Dataset loader module for flexible data source handling.

Supports loading commits from CSV files with customizable column mappings.
Extensible to other data sources (Parquet, JSON, database, etc.).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


class DatasetLoader:
    """
    Flexible dataset loader for commit data.
    
    Supports multiple formats and customizable column mappings.
    All data is normalized to a consistent internal format.
    """

    def __init__(self, source: Path | str, format: str = "csv"):
        """
        Initialize loader.
        
        Parameters
        ----------
        source : Path | str
            Path to data file
        format : str
            Data format ('csv', 'parquet', 'json')
        """
        self.source = Path(source)
        self.format = format
        self._data = None

    def load(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        sort_by: str = "author_date",
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Load and normalize dataset.
        
        Parameters
        ----------
        column_mapping : dict, optional
            Rename columns to standard names.
            E.g., {'sha': 'commit_id', 'timestamp': 'author_date'}
        sort_by : str
            Column to sort by (default: author_date)
        filters : dict, optional
            Column filters {col: value} to apply
            
        Returns
        -------
        pd.DataFrame
            Normalized dataframe
        """
        if self._data is None:
            self._read_file()
        
        df = self._data.copy()
        
        # Apply column mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist with defaults
        required_cols = ["commit_id", "author_date", "project", "buggy"]
        for col in required_cols:
            if col not in df.columns:
                if col == "buggy":
                    df[col] = False
                elif col == "project":
                    df[col] = "unknown"
                else:
                    raise ValueError(f"Required column '{col}' not found")
        
        # Apply filters
        if filters:
            for col, val in filters.items():
                if col in df.columns:
                    if isinstance(val, (list, tuple)):
                        df = df[df[col].isin(val)]
                    else:
                        df = df[df[col] == val]
        
        # Sort by timestamp
        if sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)
        
        return df

    def _read_file(self):
        """Read file in specified format."""
        if self.format == "csv":
            self._data = pd.read_csv(self.source)
        elif self.format == "parquet":
            self._data = pd.read_parquet(self.source)
        elif self.format == "json":
            self._data = pd.read_json(self.source)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def peek(self, n: int = 5) -> pd.DataFrame:
        """Preview first n rows."""
        if self._data is None:
            self._read_file()
        return self._data.head(n)


class DiffTextSource:
    """
    Manages diff_text data loading from various sources.
    
    Supports:
    - CSV column directly
    - Separate CSV file keyed by commit_id
    - Git repository (via GitPython)
    """

    def __init__(self, source: Optional[Path | str] = None, source_type: str = "csv_column"):
        """
        Initialize diff source.
        
        Parameters
        ----------
        source : Path | str, optional
            Path to diff data source
        source_type : str
            'csv_column': diff_text as CSV column
            'csv_file': separate CSV with commit_id + diff_text columns
            'git_repo': git repository path
        """
        self.source = Path(source) if source else None
        self.source_type = source_type
        self._cache = {}

    def get_diff(self, commit_id: str, source_df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """
        Get diff text for a commit.
        
        Parameters
        ----------
        commit_id : str
            Commit hash
        source_df : pd.DataFrame, optional
            DataFrame to extract from (for csv_column mode)
            
        Returns
        -------
        str | None
            Diff text or None if not found
        """
        if commit_id in self._cache:
            return self._cache[commit_id]
        
        diff_text = None
        
        if self.source_type == "csv_column" and source_df is not None:
            # Get from dataframe column
            mask = source_df["commit_id"] == commit_id
            if mask.any() and "diff_text" in source_df.columns:
                diff_text = source_df.loc[mask, "diff_text"].iloc[0]
        
        elif self.source_type == "csv_file" and self.source:
            # Load from separate CSV
            if "diff_df" not in self._cache:
                try:
                    self._cache["diff_df"] = pd.read_csv(
                        self.source, dtype={"commit_id": str}
                    )
                except Exception:
                    return None
            
            diff_df = self._cache["diff_df"]
            mask = diff_df["commit_id"] == commit_id
            if mask.any() and "diff_text" in diff_df.columns:
                diff_text = diff_df.loc[mask, "diff_text"].iloc[0]
        
        elif self.source_type == "git_repo" and self.source:
            # Load from git repository
            try:
                from git import Repo
                repo = Repo(self.source)
                diff_text = repo.git.show(commit_id, "--format=", "-p")
            except Exception:
                return None
        
        if diff_text and isinstance(diff_text, str):
            self._cache[commit_id] = diff_text
        
        return diff_text
    
    def load_batch(self, commit_ids: List[str], **kwargs) -> Dict[str, Optional[str]]:
        """Load diff text for multiple commits."""
        return {cid: self.get_diff(cid, **kwargs) for cid in commit_ids}
