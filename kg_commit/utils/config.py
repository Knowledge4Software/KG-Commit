from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None