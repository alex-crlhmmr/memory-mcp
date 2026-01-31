"""Shared test fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_mcp.config import Settings, _settings
import memory_mcp.config as config_module


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path):
    """Override settings to use a temporary directory for all tests."""
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        lance_db_path=tmp_path / "data" / "lancedb",
        profile_path=tmp_path / "data" / "user_profile.json",
        anthropic_api_key="test-key-not-real",
    )
    original = config_module._settings
    config_module._settings = settings
    yield tmp_path
    config_module._settings = original


@pytest.fixture
def mock_embedder():
    """Mock embedding manager that returns deterministic vectors."""
    embedder = MagicMock()

    def make_vector(texts):
        """Create simple deterministic vectors from text."""
        vectors = []
        for text in texts:
            # Use hash of text to create a pseudo-random but deterministic vector
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h]
            # Pad/truncate to 1024 dims
            vec = (vec * 32)[:1024]
            # Normalize
            norm = sum(v * v for v in vec) ** 0.5
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    async def embed(texts):
        return make_vector(texts)

    async def embed_query(text):
        return make_vector([text])[0]

    async def embed_documents(texts):
        return make_vector(texts) if texts else []

    embedder.embed = AsyncMock(side_effect=embed)
    embedder.embed_query = AsyncMock(side_effect=embed_query)
    embedder.embed_documents = AsyncMock(side_effect=embed_documents)
    return embedder


@pytest.fixture
def sample_extraction_response():
    """Sample Claude API extraction response."""
    return {
        "observations": [
            {
                "category": "coding_preference",
                "content": "User prefers Python with type hints",
                "confidence": 0.95,
                "tags": ["python", "typing"],
                "supersedes": None,
            },
            {
                "category": "technical_context",
                "content": "User works on a Jetson Orin with CUDA",
                "confidence": 0.9,
                "tags": ["jetson", "cuda", "gpu"],
                "supersedes": None,
            },
            {
                "category": "code_pattern",
                "content": "User uses asyncio patterns extensively",
                "confidence": 0.85,
                "tags": ["asyncio", "async"],
                "supersedes": None,
            },
        ],
        "conversation_summary": "Discussion about setting up a memory system",
        "topics": ["memory", "MCP", "embeddings"],
        "profile_updates": [
            {
                "key": "preferred_language",
                "value": "Python with strict typing",
                "confidence": 0.95,
            }
        ],
    }
