"""YAML -> RAGConfig loader. Ported from rag-foundry's rag/config/loader.py (trimmed)."""

from pathlib import Path
from typing import Union

import yaml

from rag_agent.config import RAGConfig


class ConfigLoader:
    """Load a pinned RAGConfig from a YAML file."""

    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> RAGConfig:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return RAGConfig.from_dict(data)

    @staticmethod
    def load(filepath: Union[str, Path]) -> RAGConfig:
        filepath = Path(filepath)
        if filepath.suffix in (".yaml", ".yml"):
            return ConfigLoader.load_yaml(filepath)
        raise ValueError(f"Unsupported format: {filepath.suffix}")
