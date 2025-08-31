"""Configuration management for the preprocessing pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str = MISSING  # Dataset name or repo ID
    repo_id: Optional[str] = None  # Optional specific repo ID (e.g., "username/repo")
    files: Optional[List[str]] = None  # Files to download (single file or list)
    subset: Optional[str] = None  # Dataset subset (if using predefined datasets)
    cache_dir: str = "data/cache"
    split: str = "train"
    revision: Optional[str] = None  # Git revision (branch, tag, or commit hash)


@dataclass
class Config:
    """Main configuration class."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# Register config schema with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# Make config classes available at package level
__all__ = [
    'Config',
    'DatasetConfig',
] 
