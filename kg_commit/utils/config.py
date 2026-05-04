from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List

import yaml
from dotenv import load_dotenv

load_dotenv()

def find_root(current_path: Path) -> Path:
    for parent in current_path.parents:
        # Look for a marker that only exists at the root
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current_path.parents[-1] if current_path.parents else current_path


def get_project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser()
        env_root_path = env_path.resolve(strict=False)
        if env_root_path.exists():
            return env_root_path

    return find_root(Path(__file__).resolve())


PROJECT_ROOT = get_project_root()

class Config:
    PATH_SECTIONS = ["data", "output"]

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = self._process_config(config_dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from an absolute path or relative to project root."""
        input_path = Path(path)
        
        # If relative, anchor to root; if absolute, use as is
        yaml_path = input_path if input_path.is_absolute() else (PROJECT_ROOT / input_path)
        yaml_path = yaml_path.resolve()
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Config file not found.\n"
                f"Attempted path: {yaml_path}\n"
                f"Project root:   {PROJECT_ROOT}"
            )
            
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            return cls(data)

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve paths found within the config content."""
        for section in self.PATH_SECTIONS:
            if section in config and isinstance(config[section], dict):
                for key, value in config[section].items():
                    if isinstance(value, str):
                        p = Path(value)
                        # Only anchor to root if the YAML value isn't already absolute
                        abs_path = p if p.is_absolute() else (PROJECT_ROOT / p)
                        config[section][key] = abs_path.resolve()
                        
                        if section == "output":
                            self._ensure_exists(config[section][key])
        return config

    def _ensure_exists(self, path: Path):
        """Creates directory or parent directory for a file."""
        target = path.parent if path.suffix else path
        target.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Fetch values using dot notation: config.get('data.path')"""
        parts = key.split('.')
        val = self._config
        for part in parts:
            if not isinstance(val, dict):
                return default
            val = val.get(part, default)
        return val

    def __getitem__(self, key: str) -> Any:
        return self.get(key)