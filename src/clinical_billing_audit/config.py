"""Pipeline configuration management.

Loads YAML configuration files and provides a typed, dot-accessible
config object used by every pipeline stage.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass
class PipelineConfig:
    """Central configuration object for the pipeline.

    Attributes
    ----------
    raw : dict
        The raw (merged) YAML dictionary.
    config_path : Path | None
        Path the config was loaded from.
    """

    raw: dict = field(default_factory=dict)
    config_path: Path | None = None

    # ── Accessors ───────────────────────────────────────────

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a value using dot-separated keys.

        Example
        -------
        >>> cfg.get("baseline.tfidf.max_features", 10000)
        15000
        """
        keys = dotted_key.split(".")
        node = self.raw
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw

    # ── Convenience Properties ──────────────────────────────

    @property
    def seed(self) -> int:
        return self.get("project.random_seed", 42)

    @property
    def data(self) -> dict:
        return self.raw.get("data", {})

    @property
    def quality(self) -> dict:
        return self.raw.get("quality", {})

    @property
    def nlp(self) -> dict:
        return self.raw.get("nlp", {})

    @property
    def baseline(self) -> dict:
        return self.raw.get("baseline", {})

    @property
    def transformer(self) -> dict:
        return self.raw.get("transformer", {})

    @property
    def anomaly(self) -> dict:
        return self.raw.get("anomaly", {})

    @property
    def evaluation(self) -> dict:
        return self.raw.get("evaluation", {})

    @property
    def dashboard(self) -> dict:
        return self.raw.get("dashboard", {})


def load_config(
    config_path: str | Path,
    overrides: dict | None = None,
) -> PipelineConfig:
    """Load a YAML config and optionally merge overrides.

    Parameters
    ----------
    config_path : str | Path
        Path to the primary YAML configuration file.
    overrides : dict | None
        Optional dictionary of overrides to merge on top.

    Returns
    -------
    PipelineConfig
        Fully resolved configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    logger.info("Loaded config from %s", config_path)
    return PipelineConfig(raw=raw, config_path=config_path)
