"""Tests for the extraction pipeline."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from memory_mcp.extraction.extractor import (
    ConversationExtractor,
    _IncrementalObservationParser,
)
from memory_mcp.models import ObservationCategory


# ---------------------------------------------------------------------------
# Helpers for mocking the streaming API
# ---------------------------------------------------------------------------

class _FakeTextStream:
    """Simulates anthropic stream.text_stream yielding chunks."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeStreamContext:
    """Context manager returned by client.messages.stream(...)."""

    def __init__(self, chunks: list[str]):
        self._stream = _FakeTextStream(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def text_stream(self):
        return self._stream


def _make_stream_chunks(response_dict: dict, chunk_size: int = 40) -> list[str]:
    """Split a JSON response dict into small string chunks."""
    full = json.dumps(response_dict)
    return [full[i:i + chunk_size] for i in range(0, len(full), chunk_size)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_response(sample_extraction_response):
    """Create a mock Anthropic API response."""
    content_block = MagicMock()
    content_block.text = json.dumps(sample_extraction_response)
    response = MagicMock()
    response.content = [content_block]
    return response


# ---------------------------------------------------------------------------
# Tests for the original (non-streaming) extract method
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_observations(mock_anthropic_response, sample_extraction_response):
    """Test that extraction correctly parses Claude API response."""
    extractor = ConversationExtractor()

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_client_fn.return_value = mock_client

        result = await extractor.extract("User: Hello\nAssistant: Hi!", {})

    assert len(result.observations) == 3
    assert result.observations[0].category == ObservationCategory.CODING_PREFERENCE
    assert result.observations[0].confidence == 0.95
    assert "python" in result.observations[0].tags
    assert result.conversation_summary == "Discussion about setting up a memory system"
    assert len(result.topics) == 3
    assert len(result.profile_updates) == 1
    assert result.profile_updates[0].key == "preferred_language"


@pytest.mark.asyncio
async def test_extract_handles_markdown_code_blocks(sample_extraction_response):
    """Test extraction handles responses wrapped in markdown code blocks."""
    extractor = ConversationExtractor()

    content_block = MagicMock()
    content_block.text = f"```json\n{json.dumps(sample_extraction_response)}\n```"
    response = MagicMock()
    response.content = [content_block]

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        mock_client_fn.return_value = mock_client

        result = await extractor.extract("test conversation", {})

    assert len(result.observations) == 3


@pytest.mark.asyncio
async def test_extract_skips_unknown_categories():
    """Test that unknown categories are skipped gracefully."""
    extractor = ConversationExtractor()

    data = {
        "observations": [
            {
                "category": "unknown_category",
                "content": "something",
                "confidence": 0.5,
                "tags": [],
            },
            {
                "category": "coding_preference",
                "content": "valid observation",
                "confidence": 0.8,
                "tags": [],
            },
        ],
        "conversation_summary": "test",
        "topics": [],
        "profile_updates": [],
    }

    content_block = MagicMock()
    content_block.text = json.dumps(data)
    response = MagicMock()
    response.content = [content_block]

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        mock_client_fn.return_value = mock_client

        result = await extractor.extract("test", {})

    assert len(result.observations) == 1
    assert result.observations[0].content == "valid observation"


# ---------------------------------------------------------------------------
# Tests for the streaming extract method
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_streaming_basic(sample_extraction_response):
    """Test streaming extraction parses all observations and final result."""
    extractor = ConversationExtractor()
    chunks = _make_stream_chunks(sample_extraction_response, chunk_size=30)

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _FakeStreamContext(chunks)
        mock_client_fn.return_value = mock_client

        queue, future = await extractor.extract_streaming("test conversation", {})

        # Drain the queue
        streamed_obs = []
        while True:
            obs = await asyncio.wait_for(queue.get(), timeout=5.0)
            if obs is None:
                break
            streamed_obs.append(obs)

        result = await asyncio.wait_for(future, timeout=5.0)

    # Streamed observations should match the 3 in sample data
    assert len(streamed_obs) == 3
    assert streamed_obs[0].category == ObservationCategory.CODING_PREFERENCE
    assert streamed_obs[1].category == ObservationCategory.TECHNICAL_CONTEXT
    assert streamed_obs[2].category == ObservationCategory.CODE_PATTERN

    # Final result should have full data
    assert len(result.observations) == 3
    assert result.conversation_summary == "Discussion about setting up a memory system"
    assert len(result.topics) == 3
    assert len(result.profile_updates) == 1


@pytest.mark.asyncio
async def test_extract_streaming_single_char_chunks(sample_extraction_response):
    """Test streaming with one character per chunk (worst-case fragmentation)."""
    extractor = ConversationExtractor()
    chunks = _make_stream_chunks(sample_extraction_response, chunk_size=1)

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _FakeStreamContext(chunks)
        mock_client_fn.return_value = mock_client

        queue, future = await extractor.extract_streaming("test", {})

        streamed_obs = []
        while True:
            obs = await asyncio.wait_for(queue.get(), timeout=5.0)
            if obs is None:
                break
            streamed_obs.append(obs)

        result = await asyncio.wait_for(future, timeout=5.0)

    assert len(streamed_obs) == 3
    assert len(result.observations) == 3


@pytest.mark.asyncio
async def test_extract_streaming_empty_observations():
    """Test streaming with no observations."""
    extractor = ConversationExtractor()
    data = {
        "observations": [],
        "conversation_summary": "Empty",
        "topics": [],
        "profile_updates": [],
    }
    chunks = _make_stream_chunks(data)

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _FakeStreamContext(chunks)
        mock_client_fn.return_value = mock_client

        queue, future = await extractor.extract_streaming("test", {})

        streamed_obs = []
        while True:
            obs = await asyncio.wait_for(queue.get(), timeout=5.0)
            if obs is None:
                break
            streamed_obs.append(obs)

        result = await asyncio.wait_for(future, timeout=5.0)

    assert len(streamed_obs) == 0
    assert len(result.observations) == 0
    assert result.conversation_summary == "Empty"


@pytest.mark.asyncio
async def test_extract_streaming_skips_unknown_categories():
    """Test that streaming skips unknown categories like non-streaming does."""
    extractor = ConversationExtractor()
    data = {
        "observations": [
            {"category": "bad_cat", "content": "skip me", "confidence": 0.5, "tags": []},
            {"category": "coding_preference", "content": "keep me", "confidence": 0.8, "tags": []},
        ],
        "conversation_summary": "test",
        "topics": [],
        "profile_updates": [],
    }
    chunks = _make_stream_chunks(data, chunk_size=20)

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _FakeStreamContext(chunks)
        mock_client_fn.return_value = mock_client

        queue, future = await extractor.extract_streaming("test", {})

        streamed_obs = []
        while True:
            obs = await asyncio.wait_for(queue.get(), timeout=5.0)
            if obs is None:
                break
            streamed_obs.append(obs)

        result = await asyncio.wait_for(future, timeout=5.0)

    # Streamed parser skips unknown categories
    assert len(streamed_obs) == 1
    assert streamed_obs[0].content == "keep me"
    # Final parse also skips unknown
    assert len(result.observations) == 1


# ---------------------------------------------------------------------------
# Tests for _IncrementalObservationParser directly
# ---------------------------------------------------------------------------

class TestIncrementalParser:
    """Unit tests for the bracket-matching incremental parser."""

    def test_single_chunk(self):
        parser = _IncrementalObservationParser("conv123")
        data = json.dumps({
            "observations": [
                {"category": "decision", "content": "chose REST", "confidence": 0.9, "tags": ["api"]},
            ],
            "conversation_summary": "test",
        })
        results = parser.feed(data)
        assert len(results) == 1
        assert results[0].content == "chose REST"
        assert results[0].category == ObservationCategory.DECISION
        assert results[0].conversation_id == "conv123"

    def test_multi_chunk(self):
        parser = _IncrementalObservationParser("conv456")
        data = json.dumps({
            "observations": [
                {"category": "coding_preference", "content": "likes Rust", "confidence": 0.8, "tags": []},
                {"category": "subtle_signal", "content": "dislikes ORM", "confidence": 0.7, "tags": []},
            ],
            "other": "stuff",
        })
        # Feed one char at a time
        all_obs = []
        for ch in data:
            all_obs.extend(parser.feed(ch))
        assert len(all_obs) == 2
        assert all_obs[0].content == "likes Rust"
        assert all_obs[1].content == "dislikes ORM"

    def test_handles_strings_with_braces(self):
        """Braces inside JSON strings should not confuse the parser."""
        parser = _IncrementalObservationParser("conv789")
        data = json.dumps({
            "observations": [
                {
                    "category": "code_pattern",
                    "content": "Uses dict comprehension {k: v for k, v in items}",
                    "confidence": 0.85,
                    "tags": ["python"],
                },
            ],
            "conversation_summary": "test",
        })
        # Feed in small chunks
        all_obs = []
        for i in range(0, len(data), 5):
            all_obs.extend(parser.feed(data[i:i + 5]))
        assert len(all_obs) == 1
        assert "{k: v for k, v in items}" in all_obs[0].content

    def test_no_observations_key(self):
        """Parser returns nothing if observations key is absent."""
        parser = _IncrementalObservationParser("conv")
        data = json.dumps({"summary": "no observations here"})
        results = parser.feed(data)
        assert len(results) == 0

    def test_handles_escaped_quotes(self):
        """Parser handles escaped quotes inside strings."""
        parser = _IncrementalObservationParser("conv")
        data = json.dumps({
            "observations": [
                {
                    "category": "decision",
                    "content": 'Prefers "double quotes" in code',
                    "confidence": 0.7,
                    "tags": [],
                },
            ],
            "conversation_summary": "test",
        })
        all_obs = []
        for i in range(0, len(data), 3):
            all_obs.extend(parser.feed(data[i:i + 3]))
        assert len(all_obs) == 1
        assert '"double quotes"' in all_obs[0].content
