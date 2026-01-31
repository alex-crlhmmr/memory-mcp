"""Benchmark: streaming+parallel vs sequential extraction+embedding.

Simulates realistic latency:
- Claude API: ~50ms per chunk (simulating token-by-token streaming)
- Embedding: ~200ms per batch (simulating GPU inference)

Measures wall-clock time for both approaches to show overlap benefit.
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
# Test data: 12 observations to exercise multiple batches
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

# Latency parameters (seconds) — tuned to reflect real-world conditions:
# Claude API streams ~30 tokens/s, each chunk ≈ a few tokens.
# Qwen3-Embedding-8B on Jetson Orin takes ~300-500ms per small batch.
CHUNK_DELAY = 0.05       # delay per streamed chunk (~network/token latency)
EMBED_DELAY = 0.30       # delay per embedding batch (~GPU inference)
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
    """Mock embedder that sleeps to simulate GPU work."""
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
# Sequential baseline (old approach)
# ---------------------------------------------------------------------------

async def _run_sequential(extractor, embedder, store, profile_store):
    """Simulate the old sequential approach: full extract then full embed."""
    current_profile = profile_store.load()

    # Full blocking extraction (simulate full stream wait)
    t0 = time.perf_counter()
    chunks = _make_chunks(RESPONSE_JSON)
    full_text = ""
    for chunk in chunks:
        await asyncio.sleep(CHUNK_DELAY)  # simulate network
        full_text += chunk

    data = json.loads(full_text)
    observations = []
    for obs_data in data["observations"]:
        observations.append(Observation(
            conversation_id="seq_test",
            category=ObservationCategory(obs_data["category"]),
            content=obs_data["content"],
            confidence=obs_data["confidence"],
            tags=obs_data.get("tags", []),
        ))
    extraction_time = time.perf_counter() - t0

    # Full embedding (one big batch)
    t1 = time.perf_counter()
    contents = [obs.content for obs in observations]
    vectors = await embedder.embed_documents(contents)
    embed_time = time.perf_counter() - t1

    total = time.perf_counter() - t0
    return {
        "total": total,
        "extraction_time": extraction_time,
        "embed_time": embed_time,
        "n_observations": len(observations),
    }


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_faster_than_sequential(tmp_data_dir):
    """Streaming+parallel should have lower wall time than sequential."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()
    embedder = _make_slow_embedder(EMBED_DELAY)

    # --- Sequential baseline ---
    seq_result = await _run_sequential(extractor, embedder, store, profile_store)

    # --- Streaming approach ---
    chunks = _make_chunks(RESPONSE_JSON)
    stream_embedder = _make_slow_embedder(EMBED_DELAY)

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _SlowStreamContext(chunks, CHUNK_DELAY)
        mock_client_fn.return_value = mock_client

        t0 = time.perf_counter()
        result = await save_conversation_memory(
            conversation_text="benchmark test conversation",
            extractor=extractor,
            embedder=stream_embedder,
            store=store,
            profile_store=profile_store,
        )
        streaming_total = time.perf_counter() - t0

    assert result["status"] == "saved"
    assert result["observations_count"] == 12

    # Report
    n_chunks = len(chunks)
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: 12 observations, {n_chunks} stream chunks")
    print(f"  Chunk delay: {CHUNK_DELAY*1000:.0f}ms | Embed delay: {EMBED_DELAY*1000:.0f}ms")
    print(f"{'='*60}")
    print(f"  Sequential:  {seq_result['total']:.3f}s  (extract {seq_result['extraction_time']:.3f}s + embed {seq_result['embed_time']:.3f}s)")
    print(f"  Streaming:   {streaming_total:.3f}s")
    speedup = seq_result["total"] / streaming_total if streaming_total > 0 else float("inf")
    saved = seq_result["total"] - streaming_total
    print(f"  Speedup:     {speedup:.2f}x  ({saved:.3f}s saved)")
    print(f"{'='*60}")

    # The streaming approach should be meaningfully faster because embedding
    # overlaps with continued streaming. With 3 embed batches of 4 observations
    # at 150ms each = 450ms of embedding. In streaming mode, most of that
    # overlaps with the stream, so wall time should be notably less.
    assert streaming_total < seq_result["total"], (
        f"Streaming ({streaming_total:.3f}s) should be faster than "
        f"sequential ({seq_result['total']:.3f}s)"
    )
