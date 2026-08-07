"""Domain registry: loads the pinned per-domain config + data-loading
metadata from configs/domains.yaml and configs/<domain>.yaml.

Replaces rag-foundry's dynamic "auto-pick best config from comparison.csv"
step: the winning config per domain was computed once (see README.md /
the implementation plan) and is pinned as a literal YAML file here, so
this package needs zero rag-foundry files at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from rag_agent.config import RAGConfig
from rag_agent.loader import ConfigLoader

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@dataclass
class DomainSpec:
    """Everything needed to build and query one domain's RAG pipeline."""

    key: str
    description: str
    rag_config: RAGConfig
    data_loader: Dict[str, Any]
    data_parser: Any  # str or {type: ..., config: {...}}
    data_processing: Optional[Dict[str, Any]] = None

    # Populated at RagAgent construction time (eager build).
    pipeline: Any = field(default=None, repr=False)


def load_domains(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Dict[str, DomainSpec]:
    """Load all domains declared in ``configs/domains.yaml``."""
    config_dir = Path(config_dir)

    with open(config_dir / "domains.yaml", "r") as f:
        domains_data = yaml.safe_load(f)["domains"]

    domains: Dict[str, DomainSpec] = {}
    for domain_key, domain_data in domains_data.items():
        rag_config = ConfigLoader.load(config_dir / domain_data["config_file"])
        domains[domain_key] = DomainSpec(
            key=domain_key,
            description=domain_data["description"].strip(),
            rag_config=rag_config,
            data_loader=domain_data["data_loader"],
            data_parser=domain_data["data_parser"],
            data_processing=domain_data.get("data_processing"),
        )
    return domains


def load_evaluation_config(config_dir: Path | str = DEFAULT_CONFIG_DIR):
    """Load the single shared TRACe evaluation config from configs/evaluation.yaml."""
    from rag_agent.config import EvaluationConfig

    config_dir = Path(config_dir)
    with open(config_dir / "evaluation.yaml", "r") as f:
        data = yaml.safe_load(f)["evaluation"]
    return EvaluationConfig(**data)


@dataclass
class RGBEvaluationSectionConfig:
    """Thin ``{type, provider, config}`` wrapper mirroring EvaluationConfig's
    shape, for the supplementary RGB scoring section of evaluation.yaml."""

    type: str
    provider: str
    config: Dict[str, Any]


def load_rgb_evaluation_config(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> RGBEvaluationSectionConfig:
    """Load the shared RGB-benchmark-style evaluation config from
    configs/evaluation.yaml's ``rgb:`` section."""
    config_dir = Path(config_dir)
    with open(config_dir / "evaluation.yaml", "r") as f:
        data = yaml.safe_load(f)["rgb"]
    return RGBEvaluationSectionConfig(**data)
