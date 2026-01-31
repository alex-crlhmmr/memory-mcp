"""recall_memories tool implementation."""

from __future__ import annotations

import logging
from typing import Optional

from memory_mcp.config import get_settings
from memory_mcp.embeddings.manager import EmbeddingManager
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore

logger = logging.getLogger(__name__)


async def recall_memories(
    query: str,
    embedder: EmbeddingManager,
    store: LanceStore,
    profile_store: UserProfileStore,
    limit: Optional[int] = None,
    category: Optional[str] = None,
) -> dict:
    """Search past observations and return user profile + relevant memories.

    Args:
        query: Natural language search query.
        embedder: Embedding manager for query encoding.
        store: LanceDB store for vector search.
        profile_store: User profile store.
        limit: Max number of memories to return.
        category: Filter by observation category.

    Returns:
        Dict with user_profile, memories, and total_found.
    """
    settings = get_settings()
    limit = limit or settings.default_recall_limit

    # Embed the query
    logger.info("Embedding recall query: %s", query[:100])
    query_vector = await embedder.embed_query(query)

    # Search
    memories = store.search(
        query_vector=query_vector,
        limit=limit,
        category=category,
        relevance_threshold=settings.relevance_threshold,
    )

    # Load profile
    user_profile = profile_store.load()

    return {
        "user_profile": user_profile,
        "memories": [m.model_dump() for m in memories],
        "total_found": len(memories),
    }
