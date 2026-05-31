"""
Knowledge Graph Action Logger.

Logs all KG operations (node adds, edge adds, counter updates) to CSV.
Provides an audit trail and enables analysis of KG construction patterns.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import csv
import json


class KGActionLogger:
    """
    Logs all KG building actions to a CSV file.
    
    Action types:
    - ADD_NODE: Add a node to the graph
    - ADD_EDGE: Add an edge between two nodes
    - UPDATE_ATTR: Update node/edge attribute
    - CREATE_INTERVAL: Create interval node
    - UPDATE_INTERVAL: Update interval bounds
    """

    def __init__(self, output_path: Optional[Path | str] = None):
        """
        Initialize logger.
        
        Parameters
        ----------
        output_path : Path | str, optional
            Path to CSV file. If None, logs to memory only.
        """
        self.output_path = Path(output_path) if output_path else None
        self.actions: List[Dict[str, Any]] = []
        self._file_handle = None
        self._writer = None
        self._headers = [
            "timestamp",
            "action_type",
            "commit_id",
            "entity_type",
            "entity_id",
            "target_entity_id",
            "relation_type",
            "attribute_name",
            "attribute_value",
            "tier",
            "graph_state",
        ]
        
        if self.output_path:
            self._init_csv()

    def _init_csv(self):
        """Initialize CSV file."""
        if not self.output_path:
            return
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file_handle, fieldnames=self._headers)
        self._writer.writeheader()

    def log_action(
        self,
        action_type: str,
        commit_id: str,
        entity_type: str,
        entity_id: str,
        target_entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        tier: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
        graph_state: Optional[Dict[str, int]] = None,
    ):
        """
        Log an action.
        
        Parameters
        ----------
        action_type : str
            ADD_NODE, ADD_EDGE, UPDATE_ATTR, CREATE_INTERVAL, UPDATE_INTERVAL
        commit_id : str
            Commit being processed
        entity_type : str
            COMMIT, AUTHOR, FILE, FUNCTION, etc.
        entity_id : str
            Node ID
        target_entity_id : str, optional
            Target node for edges
        relation_type : str, optional
            Edge relation type (by, contains, parent, etc.)
        tier : int, optional
            KG tier (1-5)
        attributes : dict, optional
            Node/edge attributes being set
        graph_state : dict, optional
            Graph metrics (n_nodes, n_edges, etc.)
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "commit_id": commit_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "target_entity_id": target_entity_id or "",
            "relation_type": relation_type or "",
            "attribute_name": "",
            "attribute_value": "",
            "tier": tier or 0,
            "graph_state": json.dumps(graph_state or {}),
        }
        
        # Log each attribute separately
        if attributes:
            base_record = record.copy()
            for attr_name, attr_value in attributes.items():
                record = base_record.copy()
                record["attribute_name"] = attr_name
                record["attribute_value"] = str(attr_value)
                self._write_record(record)
        else:
            self._write_record(record)

    def _write_record(self, record: Dict[str, Any]):
        """Write a single record to CSV and memory."""
        self.actions.append(record)
        
        if self._writer:
            self._writer.writerow(record)
            if self._file_handle:
                self._file_handle.flush()

    def log_commit_start(self, commit_id: str, graph_state: Optional[Dict] = None):
        """Log start of commit processing."""
        self.log_action(
            action_type="COMMIT_START",
            commit_id=commit_id,
            entity_type="COMMIT",
            entity_id=commit_id,
            graph_state=graph_state,
        )

    def log_commit_end(self, commit_id: str, graph_state: Optional[Dict] = None):
        """Log end of commit processing."""
        self.log_action(
            action_type="COMMIT_END",
            commit_id=commit_id,
            entity_type="COMMIT",
            entity_id=commit_id,
            graph_state=graph_state,
        )

    def log_node_add(
        self,
        commit_id: str,
        entity_id: str,
        entity_type: str,
        tier: int,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Log adding a node."""
        self.log_action(
            action_type="ADD_NODE",
            commit_id=commit_id,
            entity_type=entity_type,
            entity_id=entity_id,
            tier=tier,
            attributes=attributes,
        )

    def log_edge_add(
        self,
        commit_id: str,
        from_id: str,
        to_id: str,
        relation_type: str,
        tier: int,
    ):
        """Log adding an edge."""
        self.log_action(
            action_type="ADD_EDGE",
            commit_id=commit_id,
            entity_type="EDGE",
            entity_id=from_id,
            target_entity_id=to_id,
            relation_type=relation_type,
            tier=tier,
        )

    def log_attr_update(
        self,
        commit_id: str,
        entity_id: str,
        entity_type: str,
        attribute_name: str,
        old_value: Any,
        new_value: Any,
    ):
        """Log updating an attribute."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "action_type": "UPDATE_ATTR",
            "commit_id": commit_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "attribute_name": attribute_name,
            "attribute_value": f"{old_value} → {new_value}",
        }
        self._write_record(record)

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged actions.
        
        Returns
        -------
        dict
            Counts by action type, entity type, tier, etc.
        """
        if not self.actions:
            return {}
        
        import pandas as pd
        df = pd.DataFrame(self.actions)
        
        return {
            "total_actions": len(df),
            "by_action_type": df["action_type"].value_counts().to_dict(),
            "by_entity_type": df["entity_type"].value_counts().to_dict(),
            "by_tier": df[df["tier"] > 0]["tier"].value_counts().to_dict(),
            "commits_processed": df["commit_id"].nunique(),
        }

    def to_dataframe(self):
        """Export logged actions to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.actions)

    def close(self):
        """Close file handle."""
        if self._file_handle:
            self._file_handle.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class KGBuildMetrics:
    """
    Tracks metrics during KG construction.
    
    Records:
    - Graph growth (nodes, edges over time)
    - Entity type distribution
    - Tier coverage
    - Processing time per commit
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.snapshots: List[Dict[str, Any]] = []
        self.timings: Dict[str, List[float]] = {}

    def snapshot(
        self,
        commit_id: str,
        commit_idx: int,
        graph_nodes: int,
        graph_edges: int,
        node_types: Dict[str, int],
        elapsed_ms: float,
    ):
        """Record a snapshot at a commit."""
        self.snapshots.append({
            "commit_id": commit_id,
            "commit_idx": commit_idx,
            "n_nodes": graph_nodes,
            "n_edges": graph_edges,
            "avg_degree": graph_edges / max(graph_nodes, 1),
            "elapsed_ms": elapsed_ms,
            **{f"type_{t}": c for t, c in node_types.items()},
        })

    def record_timing(self, operation: str, elapsed_ms: float):
        """Record timing for an operation."""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(elapsed_ms)

    def summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        import statistics
        
        summary = {
            "total_snapshots": len(self.snapshots),
            "operations": {},
        }
        
        for op, times in self.timings.items():
            summary["operations"][op] = {
                "count": len(times),
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0,
            }
        
        if self.snapshots:
            last = self.snapshots[-1]
            summary["final_state"] = {
                "n_nodes": last["n_nodes"],
                "n_edges": last["n_edges"],
                "avg_degree": last["avg_degree"],
            }
        
        return summary

    def to_dataframe(self):
        """Export snapshots to DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.snapshots)
