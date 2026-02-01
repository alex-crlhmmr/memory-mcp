"""Benchmark: streaming extraction + single-batch embedding.

Simulates realistic latency:
- Claude API: ~50ms per chunk (simulating token-by-token streaming)
- Embedding: ~200ms per batch (simulating model inference)

Verifies the pipeline completes correctly and measures wall-clock time.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.models import ExtractionResult, Observation, ObservationCategory
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore
from memory_mcp.tools.save import save_conversation_memory


# ---------------------------------------------------------------------------
# Test data: 12 observations
# ---------------------------------------------------------------------------

def _make_response(n_observations: int = 12) -> dict:
    categories = list(ObservationCategory)
    return {
        "observations": [
            {
                "category": categories[i % len(categories)].value,
                "content": f"Observation number {i}: user prefers pattern {i}",
                "confidence": round(0.7 + 0.02 * i, 2),
                "tags": [f"tag{i}"],
                "supersedes": None,
            }
            for i in range(n_observations)
        ],
        "conversation_summary": "A detailed technical discussion",
        "topics": ["architecture", "performance", "testing"],
        "profile_updates": [
            {"key": "bench_pref", "value": "fast code", "confidence": 0.9}
        ],
    }


RESPONSE_DATA = _make_response(12)
RESPONSE_JSON = json.dumps(RESPONSE_DATA)

# Latency parameters (seconds)
CHUNK_DELAY = 0.05       # delay per streamed chunk (~network/token latency)
EMBED_DELAY = 0.30       # delay per embedding batch (~model inference)
CHUNK_SIZE = 60           # characters per chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


class _SlowTextStream:
    """Simulates streaming with per-chunk delay."""

    def __init__(self, chunks: list[str], delay: float):
        self._chunks = chunks
        self._delay = delay

    def __iter__(self):
        for chunk in self._chunks:
            time.sleep(self._delay)
            yield chunk


class _SlowStreamContext:
    def __init__(self, chunks: list[str], delay: float):
        self._stream = _SlowTextStream(chunks, delay)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def text_stream(self):
        return self._stream


def _make_slow_embedder(delay: float = EMBED_DELAY):
    """Mock embedder that sleeps to simulate model work."""
    import hashlib

    def _vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h]
        vec = (vec * 32)[:1024]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    async def embed_documents(texts):
        await asyncio.sleep(delay)
        return [_vec(t) for t in texts]

    async def embed_query(text):
        await asyncio.sleep(delay)
        return _vec(text)

    embedder = MagicMock()
    embedder.embed_documents = AsyncMock(side_effect=embed_documents)
    embedder.embed_query = AsyncMock(side_effect=embed_query)
    embedder.embed = AsyncMock(side_effect=embed_documents)
    return embedder


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_pipeline_completes(tmp_data_dir):
    """Verify the save pipeline completes correctly with simulated latency."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()
    embedder = _make_slow_embedder(EMBED_DELAY)

    chunks = _make_chunks(RESPONSE_JSON)

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _SlowStreamContext(chunks, CHUNK_DELAY)
        mock_client_fn.return_value = mock_client

        t0 = time.perf_counter()
        result = await save_conversation_memory(
            conversation_text="benchmark test conversation",
            extractor=extractor,
            embedder=embedder,
            store=store,
            profile_store=profile_store,
        )
        total = time.perf_counter() - t0

    assert result["status"] == "saved"
    assert result["observations_count"] == 12

    # Embedding should be called exactly once (single batch)
    assert embedder.embed_documents.call_count == 1

    n_chunks = len(chunks)
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: 12 observations, {n_chunks} stream chunks")
    print(f"  Chunk delay: {CHUNK_DELAY*1000:.0f}ms | Embed delay: {EMBED_DELAY*1000:.0f}ms")
    print(f"{'='*60}")
    print(f"  Total:       {total:.3f}s")
    print(f"  Embed calls: {embedder.embed_documents.call_count}")
    print(f"{'='*60}")
