"""Configuration helpers.

This module keeps YAML loading intentionally small and transparent.
Each script accepts CLI arguments, but `--config configs/default.yaml` can be used
for repeatable experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: str | None) -> Dict[str, Any]:
    """Load a YAML config file.

    Parameters
    ----------
    path:
        Path to a YAML file. If None, an empty dict is returned.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file does not exist: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get_nested(config: Dict[str, Any], keys: list[str], default: Any) -> Any:
    """Safely read nested values from a dictionary.

    Example
    -------
    >>> get_nested(cfg, ["training", "epochs"], 30)
    """
    cur = config
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
