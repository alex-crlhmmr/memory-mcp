"""MCP server entry point using FastMCP with stdio transport."""

from __future__ import annotations

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from memory_mcp.embeddings.manager import EmbeddingManager
from memory_mcp.extraction.extractor import ConversationExtractor
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore
from memory_mcp.tools.recall import recall_memories as _recall
from memory_mcp.tools.save import save_conversation_memory as _save
from memory_mcp.tools.update import update_memory as _update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Shared instances (created once, reused across calls)
_extractor = ConversationExtractor()
_embedder = EmbeddingManager()
_store = LanceStore()
_profile_store = UserProfileStore()

mcp = FastMCP("memory")


@mcp.tool()
async def save_conversation_memory(conversation_text: str) -> dict:
    """Extract observations from a conversation and store them in long-term memory.

    Call this at the end of every conversation to build persistent memory.
    Extracts coding preferences, patterns, decisions, and subtle signals.

    Args:
        conversation_text: The full conversation text to analyze.
    """
    return await _save(
        conversation_text=conversation_text,
        extractor=_extractor,
        embedder=_embedder,
        store=_store,
        profile_store=_profile_store,
    )


@mcp.tool()
async def recall_memories(
    query: str,
    limit: Optional[int] = None,
    category: Optional[str] = None,
) -> dict:
    """Search past observations and return the user profile with relevant memories.

    Call this at the start of conversations or when you need context about the user.

    Args:
        query: Natural language search query describing what you want to remember.
        limit: Maximum number of memories to return (default: 20).
        category: Filter by category: coding_preference, communication_pattern,
                  code_pattern, technical_context, decision, subtle_signal, project_context.
    """
    return await _recall(
        query=query,
        embedder=_embedder,
        store=_store,
        profile_store=_profile_store,
        limit=limit,
        category=category,
    )


@mcp.tool()
async def update_memory(
    observation_id: Optional[str] = None,
    new_content: Optional[str] = None,
    profile_key: Optional[str] = None,
    new_value: Optional[str] = None,
    delete: bool = False,
) -> dict:
    """Correct or remove specific memories or profile entries.

    Use this when the user says something is wrong in their memories,
    or when a preference has changed.

    Args:
        observation_id: ID of a specific observation to update or delete.
        new_content: New content for the observation (triggers re-embedding).
        profile_key: A user profile key to update or delete.
        new_value: New value for the profile key.
        delete: If True, delete the specified observation or profile key.
    """
    return await _update(
        embedder=_embedder,
        store=_store,
        profile_store=_profile_store,
        observation_id=observation_id,
        new_content=new_content,
        profile_key=profile_key,
        new_value=new_value,
        delete=delete,
    )


def main():
    """Run the MCP server with stdio transport."""
    logger.info("Starting memory MCP server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
