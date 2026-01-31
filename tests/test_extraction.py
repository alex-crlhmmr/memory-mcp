"""Tests for the extraction pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.models import ObservationCategory


@pytest.fixture
def mock_anthropic_response(sample_extraction_response):
    """Create a mock Anthropic API response."""
    content_block = MagicMock()
    content_block.text = json.dumps(sample_extraction_response)
    response = MagicMock()
    response.content = [content_block]
    return response


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
