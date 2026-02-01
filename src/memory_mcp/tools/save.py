"""save_conversation_memory tool implementation."""

from __future__ import annotations

import logging

from memory_mcp.embeddings.manager import EmbeddingManager
from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.models import ConversationRecord, Observation
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

    Uses streaming extraction so observations arrive incrementally,
    then embeds everything in a single batch.

    Returns summary of what was saved.
    """
    # Load current profile for context
    current_profile = await profile_store.load()

    # Start streaming extraction
    logger.info("Starting streaming extraction from conversation...")
    queue, result_future = await extractor.extract_streaming(
        conversation_text, current_profile
    )

    # Collect all observations from the streaming queue
    all_observations: list[Observation] = []
    while True:
        obs = await queue.get()
        if obs is None:
            break
        all_observations.append(obs)

    if not all_observations:
        return {
            "status": "no_observations",
            "message": "No observations extracted from the conversation.",
        }

    logger.info("Streamed %d observations", len(all_observations))

    # Wait for the full extraction result (summary, topics, profile_updates)
    extraction = await result_future

    # Embed all observations in a single batch
    contents = [obs.content for obs in extraction.observations]
    vectors = await embedder.embed_documents(contents)

    # Store observations with vectors
    await store.store_observations(extraction.observations, vectors)

    # Store conversation metadata
    conversation_id = extraction.observations[0].conversation_id
    record = ConversationRecord(
        conversation_id=conversation_id,
        summary=extraction.conversation_summary,
        topics=extraction.topics,
        observation_count=len(extraction.observations),
    )
    await store.store_conversation(record)

    # Apply profile updates
    if extraction.profile_updates:
        await profile_store.apply_updates(extraction.profile_updates)

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
