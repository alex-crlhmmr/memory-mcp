"""update_memory tool implementation."""

from __future__ import annotations

import logging
from typing import Optional

from memory_mcp.embeddings.manager import EmbeddingManager
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore

logger = logging.getLogger(__name__)


async def update_memory(
    embedder: EmbeddingManager,
    store: LanceStore,
    profile_store: UserProfileStore,
    observation_id: Optional[str] = None,
    new_content: Optional[str] = None,
    profile_key: Optional[str] = None,
    new_value: Optional[str] = None,
    delete: bool = False,
) -> dict:
    """Correct or remove specific memories.

    Can update/delete an observation by ID, or update/delete a profile key.

    Args:
        embedder: Embedding manager for re-embedding updated content.
        store: LanceDB store.
        profile_store: User profile store.
        observation_id: ID of observation to update/delete.
        new_content: New content for the observation (if updating).
        profile_key: Profile key to update/delete.
        new_value: New value for the profile key (if updating).
        delete: If True, delete the observation or profile key.

    Returns:
        Status dict.
    """
    results = {}

    # Handle observation updates
    if observation_id:
        if delete:
            success = store.delete_observation(observation_id)
            results["observation"] = {
                "action": "deleted",
                "observation_id": observation_id,
                "success": success,
            }
        elif new_content:
            # Re-embed the new content
            new_vector = await embedder.embed_query(new_content)
            success = store.update_observation_content(
                observation_id, new_content, new_vector
            )
            results["observation"] = {
                "action": "updated",
                "observation_id": observation_id,
                "success": success,
            }
        else:
            # Just retrieve it
            obs = store.get_observation(observation_id)
            results["observation"] = {
                "action": "retrieved",
                "data": obs,
            }

    # Handle profile updates
    if profile_key:
        if delete:
            existed = profile_store.delete_key(profile_key)
            results["profile"] = {
                "action": "deleted",
                "key": profile_key,
                "existed": existed,
            }
        elif new_value is not None:
            profile = profile_store.set_key(profile_key, new_value)
            results["profile"] = {
                "action": "updated",
                "key": profile_key,
                "value": new_value,
                "profile": profile,
            }
        else:
            profile = profile_store.load()
            results["profile"] = {
                "action": "retrieved",
                "key": profile_key,
                "value": profile.get(profile_key),
            }

    if not results:
        return {"error": "No observation_id or profile_key provided."}

    return results
