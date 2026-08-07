"""Config round-trip tests: load all 4 pinned domain configs and assert
their fields match the original winning yaml's values (no drift during
the port from rag-foundry). No network/provider calls required.
"""

from pathlib import Path

from rag_agent.agent.domain_registry import (
    load_domains,
    load_evaluation_config,
    load_rgb_evaluation_config,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_all_domains():
    domains = load_domains(CONFIG_DIR)
    assert set(domains.keys()) == {"pubmedqa", "covidqa", "finqa", "tatqa"}
    for key, spec in domains.items():
        assert spec.rag_config is not None
        assert spec.description
        assert spec.data_loader["dataset_name"] == "galileo-ai/ragbench"


def test_pubmedqa_pinned_values():
    domains = load_domains(CONFIG_DIR)
    cfg = domains["pubmedqa"].rag_config
    assert cfg.name == "pubmedqa_title_aware_v22_precise_citation"
    assert cfg.chunking.type.value == "sentence"
    assert cfg.chunking.config.max_words == 200
    assert cfg.embedding.config.model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    assert cfg.retrieval.query_transform.type.value == "multi_query"
    assert cfg.retrieval.query_transform.config.num_queries == 5
    assert cfg.retrieval.fusion.type.value == "weighted_sum"
    assert cfg.retrieval.rerank.type.value == "cross_encoder"
    assert cfg.retrieval.rerank.config.top_k == 25
    assert cfg.generation.provider == "groq"
    assert cfg.generation.config.model == "llama-3.3-70b-versatile"


def test_covidqa_pinned_values():
    domains = load_domains(CONFIG_DIR)
    cfg = domains["covidqa"].rag_config
    assert cfg.name == "covidqa_title_aware_v3"
    assert cfg.embedding.config.model_name == "BAAI/bge-large-en-v1.5"
    assert cfg.retrieval.fusion.type.value == "rrf"
    assert cfg.retrieval.fusion.config.k == 60
    assert cfg.retrieval.rerank.config.top_k == 7


def test_finqa_pinned_values():
    domains = load_domains(CONFIG_DIR)
    cfg = domains["finqa"].rag_config
    assert cfg.name == "finqa_v8_strict_meta_free"
    assert cfg.chunking.config.max_words == 300
    assert cfg.retrieval.query_transform.type.value == "multi_query"
    assert cfg.retrieval.query_transform.provider == "groq"
    assert cfg.retrieval.query_transform.config.model == "llama-3.1-8b-instant"
    assert cfg.generation.provider == "groq"
    assert cfg.generation.config.model == "openai/gpt-oss-20b"
    assert cfg.generation.config.reasoning_effort == "low"
    assert "ollama" not in cfg.providers


def test_tatqa_pinned_values():
    domains = load_domains(CONFIG_DIR)
    cfg = domains["tatqa"].rag_config
    assert cfg.name == "tatqa_v1"
    assert cfg.chunking.type.value == "finance_table_aware"
    assert cfg.chunking.config.max_rows_per_table_chunk == 6
    assert cfg.retrieval.query_transform is None
    assert cfg.retrieval.expansion.type.value == "sibling"
    assert cfg.data_processing is not None
    assert cfg.data_processing["steps"][0]["type"] == "sibling_entity_context"
    assert cfg.generation.provider == "groq"
    assert cfg.generation.config.model == "openai/gpt-oss-20b"
    assert "ollama" not in cfg.providers


def test_load_evaluation_config():
    eval_config = load_evaluation_config(CONFIG_DIR)
    assert eval_config.type.value == "trace"
    assert eval_config.provider == "groq"
    assert eval_config.config.model == "llama-3.3-70b-versatile"


def test_load_rgb_evaluation_config():
    rgb_config = load_rgb_evaluation_config(CONFIG_DIR)
    assert rgb_config.type == "rgb"
    assert rgb_config.provider == "groq"
    assert rgb_config.config["judge_model"] == "llama-3.3-70b-versatile"
