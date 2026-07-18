"""
Import-path bootstrap for the KG-Commit reproducibility package.

Every copied script begins with:

    import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
    from config.project_config import OUT, REPO_PATH, ...

Importing this module ensures the package root and its build/, inference/, and
scalability/ subdirectories are on sys.path, so the copied scripts can resolve
(a) their sibling modules by bare name -- `import kg_methods`,
    `from advanced_infer import load_kg`, `import run_final_experiments` -- exactly
    as they did in the original flat repo, and
(b) the `config.project_config` module at the package root.

This mirrors what scalability/_common.py already did in the original repo (it
injected inference/ and the repo root onto sys.path); here we centralise it so a
single import line at the top of each script is enough, and every script runs both
standalone (`python build/build_online_kg.py ...`) and when launched by the
drivers/ orchestrators.
"""
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
for _d in (_PKG, _PKG / "build", _PKG / "inference", _PKG / "scalability"):
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)
