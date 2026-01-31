"""Tests for the embedding manager."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_mock_embedder_produces_vectors(mock_embedder):
    """Test that the mock embedder produces normalized vectors."""
    vectors = await mock_embedder.embed_documents(["hello world", "test text"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    # Check normalization (should be close to 1.0)
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 0.01


@pytest.mark.asyncio
async def test_mock_embedder_deterministic(mock_embedder):
    """Test that the mock embedder is deterministic."""
    v1 = await mock_embedder.embed_query("test query")
    v2 = await mock_embedder.embed_query("test query")
    assert v1 == v2


@pytest.mark.asyncio
async def test_mock_embedder_different_texts_different_vectors(mock_embedder):
    """Test that different texts produce different vectors."""
    v1 = await mock_embedder.embed_query("hello")
    v2 = await mock_embedder.embed_query("world")
    assert v1 != v2


@pytest.mark.asyncio
async def test_embed_documents_empty(mock_embedder):
    """Test embedding empty list returns empty."""
    result = await mock_embedder.embed_documents([])
    assert result == []
