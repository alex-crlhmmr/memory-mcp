"""save_conversation_memory tool implementation."""

from __future__ import annotations

import logging

from memory_mcp.embeddings.manager import EmbeddingManager
from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.models import ConversationRecord
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore

logger = logging.getLogger(__name__)


async def save_conversation_memory(
    conversation_text: str,
    extractor: ConversationExtractor,
    embedder: EmbeddingManager,
    store: LanceStore,
    profile_store: UserProfileStore,
) -> dict:
    """Extract, embed, and store observations from a conversation.

    Returns summary of what was saved.
    """
    # Load current profile for context
    current_profile = profile_store.load()

    # Extract observations via Claude API
    logger.info("Extracting observations from conversation...")
    extraction = await extractor.extract(conversation_text, current_profile)
    logger.info(
        "Extracted %d observations, %d profile updates",
        len(extraction.observations),
        len(extraction.profile_updates),
    )

    if not extraction.observations:
        return {
            "status": "no_observations",
            "message": "No observations extracted from the conversation.",
        }

    # Embed all observation contents
    contents = [obs.content for obs in extraction.observations]
    logger.info("Embedding %d observations...", len(contents))
    vectors = await embedder.embed_documents(contents)

    # Store observations with vectors
    store.store_observations(extraction.observations, vectors)

    # Store conversation metadata
    conversation_id = extraction.observations[0].conversation_id
    record = ConversationRecord(
        conversation_id=conversation_id,
        summary=extraction.conversation_summary,
        topics=extraction.topics,
        observation_count=len(extraction.observations),
    )
    store.store_conversation(record)

    # Apply profile updates
    if extraction.profile_updates:
        profile_store.apply_updates(extraction.profile_updates)

    return {
        "status": "saved",
        "conversation_id": conversation_id,
        "observations_count": len(extraction.observations),
        "profile_updates_count": len(extraction.profile_updates),
        "summary": extraction.conversation_summary,
        "topics": extraction.topics,
        "categories": {
            cat: sum(1 for o in extraction.observations if o.category.value == cat)
            for cat in set(o.category.value for o in extraction.observations)
        },
    }
