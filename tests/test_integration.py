"""Integration tests: save -> recall -> update flow."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore
from memory_mcp.tools.recall import recall_memories
from memory_mcp.tools.save import save_conversation_memory
from memory_mcp.tools.update import update_memory


# ---------------------------------------------------------------------------
# Helpers for mocking streaming
# ---------------------------------------------------------------------------

class _FakeTextStream:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks
    def __iter__(self):
        return iter(self._chunks)

class _FakeStreamContext:
    def __init__(self, chunks: list[str]):
        self._stream = _FakeTextStream(chunks)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    @property
    def text_stream(self):
        return self._stream

def _make_stream_chunks(response_dict: dict, chunk_size: int = 50) -> list[str]:
    full = json.dumps(response_dict)
    return [full[i:i + chunk_size] for i in range(0, len(full), chunk_size)]


def _patch_extractor_streaming(extractor, response_dict):
    """Return a patch context that mocks the streaming API for an extractor."""
    chunks = _make_stream_chunks(response_dict)
    return patch.object(
        extractor, "_ensure_client",
        return_value=_make_mock_client(chunks),
    )


def _make_mock_client(chunks):
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _FakeStreamContext(chunks)
    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_and_recall_flow(
    tmp_data_dir, mock_embedder, sample_extraction_response
):
    """Test the full save -> recall pipeline."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()

    with _patch_extractor_streaming(extractor, sample_extraction_response):
        result = await save_conversation_memory(
            conversation_text="User: I prefer Python with type hints\nAssistant: Noted!",
            extractor=extractor,
            embedder=mock_embedder,
            store=store,
            profile_store=profile_store,
        )

    assert result["status"] == "saved"
    assert result["observations_count"] == 3
    assert result["profile_updates_count"] == 1

    # Verify profile was updated
    profile = profile_store.load()
    assert profile["preferred_language"] == "Python with strict typing"

    # Recall
    recall_result = await recall_memories(
        query="Python preferences",
        embedder=mock_embedder,
        store=store,
        profile_store=profile_store,
    )

    assert recall_result["user_profile"]["preferred_language"] == "Python with strict typing"
    assert recall_result["total_found"] >= 0


@pytest.mark.asyncio
async def test_save_recall_update_flow(
    tmp_data_dir, mock_embedder, sample_extraction_response
):
    """Test save -> recall -> update -> recall flow."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()

    with _patch_extractor_streaming(extractor, sample_extraction_response):
        save_result = await save_conversation_memory(
            conversation_text="test conversation",
            extractor=extractor,
            embedder=mock_embedder,
            store=store,
            profile_store=profile_store,
        )

    # Update profile
    update_result = await update_memory(
        embedder=mock_embedder,
        store=store,
        profile_store=profile_store,
        profile_key="preferred_language",
        new_value="Rust",
    )
    assert update_result["profile"]["action"] == "updated"
    assert update_result["profile"]["value"] == "Rust"

    # Verify profile changed
    profile = profile_store.load()
    assert profile["preferred_language"] == "Rust"


@pytest.mark.asyncio
async def test_delete_observation(
    tmp_data_dir, mock_embedder, sample_extraction_response
):
    """Test deleting a specific observation."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()

    with _patch_extractor_streaming(extractor, sample_extraction_response):
        save_result = await save_conversation_memory(
            conversation_text="test",
            extractor=extractor,
            embedder=mock_embedder,
            store=store,
            profile_store=profile_store,
        )

    # Get an observation ID from the store to delete
    vec = await mock_embedder.embed_query("test")
    results = store.search(vec, limit=10, relevance_threshold=0.0)

    if results:
        obs_id = results[0].observation_id
        delete_result = await update_memory(
            embedder=mock_embedder,
            store=store,
            profile_store=profile_store,
            observation_id=obs_id,
            delete=True,
        )
        assert delete_result["observation"]["action"] == "deleted"
        assert delete_result["observation"]["success"] is True

        # Verify it's gone
        assert store.get_observation(obs_id) is None


@pytest.mark.asyncio
async def test_save_no_observations(tmp_data_dir, mock_embedder):
    """Test save with empty extraction result."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()

    empty_response = {
        "observations": [],
        "conversation_summary": "Empty",
        "topics": [],
        "profile_updates": [],
    }

    with _patch_extractor_streaming(extractor, empty_response):
        result = await save_conversation_memory(
            conversation_text="hi",
            extractor=extractor,
            embedder=mock_embedder,
            store=store,
            profile_store=profile_store,
        )

    assert result["status"] == "no_observations"
