"""save_conversation_memory tool implementation."""

from __future__ import annotations

import asyncio
import logging

from memory_mcp.embeddings.manager import EmbeddingManager
from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.models import ConversationRecord, Observation
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore

logger = logging.getLogger(__name__)

# How many observations to batch before kicking off an embedding task
_BATCH_SIZE = 4


async def save_conversation_memory(
    conversation_text: str,
    extractor: ConversationExtractor,
    embedder: EmbeddingManager,
    store: LanceStore,
    profile_store: UserProfileStore,
) -> dict:
    """Extract, embed, and store observations from a conversation.

    Uses streaming extraction so embedding batches can overlap with
    continued Claude API generation.

    Returns summary of what was saved.
    """
    # Load current profile for context
    current_profile = profile_store.load()

    # Start streaming extraction
    logger.info("Starting streaming extraction from conversation...")
    queue, result_future = await extractor.extract_streaming(
        conversation_text, current_profile
    )

    # Consume observations from queue, batch-embed concurrently
    embed_tasks: list[asyncio.Task] = []
    batch: list[Observation] = []
    all_observations: list[Observation] = []

    while True:
        obs = await queue.get()
        if obs is None:
            break
        batch.append(obs)
        all_observations.append(obs)

        if len(batch) >= _BATCH_SIZE:
            embed_tasks.append(_launch_embed(batch, embedder))
            batch = []

    # Embed any remaining observations in a final batch
    if batch:
        embed_tasks.append(_launch_embed(batch, embedder))

    if not all_observations:
        return {
            "status": "no_observations",
            "message": "No observations extracted from the conversation.",
        }

    logger.info(
        "Streamed %d observations in %d embedding batches",
        len(all_observations),
        len(embed_tasks),
    )

    # Wait for the full extraction result (summary, topics, profile_updates)
    extraction = await result_future

    # Collect all embeddings in order
    batch_results = await asyncio.gather(*embed_tasks)
    vectors: list[list[float]] = []
    for batch_vectors in batch_results:
        vectors.extend(batch_vectors)

    # Use the authoritative observation list from the final parse so IDs match
    # between observations and vectors. The streamed observations were used only
    # for early embedding — the final parse is the source of truth.
    # Re-embed if count differs (shouldn't happen, but defensive).
    if len(extraction.observations) != len(vectors):
        logger.warning(
            "Observation count mismatch: final=%d streamed=%d. Re-embedding from final.",
            len(extraction.observations),
            len(vectors),
        )
        contents = [obs.content for obs in extraction.observations]
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


def _launch_embed(
    batch: list[Observation], embedder: EmbeddingManager
) -> asyncio.Task:
    """Launch an async task to embed a batch of observations."""
    contents = [obs.content for obs in batch]
    logger.info("Launching embedding task for %d observations", len(contents))
    return asyncio.create_task(embedder.embed_documents(contents))
