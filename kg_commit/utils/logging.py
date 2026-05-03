from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any
from kg_commit.core.window import Window


class Logger(ABC):
    """Abstract base class for logging utilities."""

    @abstractmethod
    def log_window(self, window: Window):
        """Log window information."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log evaluation metrics."""
        pass


class SimpleLogger(Logger):
    """Simple logger using Python's logging module."""

    def __init__(self, level: int = logging.INFO):
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def log_window(self, window: Window):
        """Log window details."""
        self.logger.info(f"Window: size={window.size()}, time=({window.start_time} → {window.end_time})")

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics."""
        self.logger.info(f"Metrics: {metrics}")