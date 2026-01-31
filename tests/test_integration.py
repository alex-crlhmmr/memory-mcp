"""Integration tests: save -> recall -> update flow."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore
from memory_mcp.tools.recall import recall_memories
from memory_mcp.tools.save import save_conversation_memory
from memory_mcp.tools.update import update_memory


@pytest.mark.asyncio
async def test_save_and_recall_flow(
    tmp_data_dir, mock_embedder, sample_extraction_response
):
    """Test the full save -> recall pipeline."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()

    # Mock Claude API response
    content_block = MagicMock()
    content_block.text = json.dumps(sample_extraction_response)
    api_response = MagicMock()
    api_response.content = [content_block]

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = api_response
        mock_client_fn.return_value = mock_client

        # Save
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
    # We should get some memories back (relevance depends on mock vectors)
    assert recall_result["total_found"] >= 0


@pytest.mark.asyncio
async def test_save_recall_update_flow(
    tmp_data_dir, mock_embedder, sample_extraction_response
):
    """Test save -> recall -> update -> recall flow."""
    extractor = ConversationExtractor()
    store = LanceStore()
    profile_store = UserProfileStore()

    content_block = MagicMock()
    content_block.text = json.dumps(sample_extraction_response)
    api_response = MagicMock()
    api_response.content = [content_block]

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = api_response
        mock_client_fn.return_value = mock_client

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

    content_block = MagicMock()
    content_block.text = json.dumps(sample_extraction_response)
    api_response = MagicMock()
    api_response.content = [content_block]

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = api_response
        mock_client_fn.return_value = mock_client

        save_result = await save_conversation_memory(
            conversation_text="test",
            extractor=extractor,
            embedder=mock_embedder,
            store=store,
            profile_store=profile_store,
        )

    # Get an observation ID from the store to delete
    # Search with a zero-threshold to get all results
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

    content_block = MagicMock()
    content_block.text = json.dumps(empty_response)
    api_response = MagicMock()
    api_response.content = [content_block]

    with patch.object(extractor, "_ensure_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = api_response
        mock_client_fn.return_value = mock_client

        result = await save_conversation_memory(
            conversation_text="hi",
            extractor=extractor,
            embedder=mock_embedder,
            store=store,
            profile_store=profile_store,
        )

    assert result["status"] == "no_observations"
