"""HuggingFace dataset loader. Ported (trimmed) from rag-foundry's
data_sources/loaders/huggingface_loader.py. All 4 domains load from the
same HF dataset (galileo-ai/ragbench), differing only by subset/limit.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class DatasetLoadingConfig:
    use_cache: bool = True
    limit: Optional[int] = None


class HuggingFaceLoader:
    """Loader for datasets from HuggingFace Hub."""

    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
        config: DatasetLoadingConfig = None,
        hf_token: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.config = config or DatasetLoadingConfig()
        self.hf_token = hf_token
        self._data = None

    def load(self) -> List[Dict[str, Any]]:
        if self._data is not None and self.config.use_cache:
            return self._data

        subset_str = f"/{self.subset}" if self.subset else ""
        print(f"Loading HuggingFace dataset: {self.dataset_name}{subset_str} ({self.split})...")

        try:
            load_args = {
                "path": self.dataset_name,
                "split": self.split,
                "token": self.hf_token,
            }
            if self.subset:
                load_args["name"] = self.subset

            dataset = load_dataset(**load_args)
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset {self.dataset_name} "
                f"(subset: {self.subset}, split: {self.split}): {str(e)}"
            )

        data = [dict(sample) for sample in dataset]

        if self.config.limit:
            data = data[:self.config.limit]

        print(f"Loaded {len(data)} samples")

        if self.config.use_cache:
            self._data = data

        return data
