"""CLI entry point for saving conversation memory from a Claude Code transcript.

Designed to be called as a SessionEnd hook:
    memory-save

Reads JSON from stdin (provided by the hook) with a `transcript_path` field
pointing to the .jsonl transcript file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    """Extract plain text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    # Skip tool results — they're internal
                    continue
        return "\n".join(parts)
    return ""


def parse_transcript(path: str) -> str:
    """Read a .jsonl transcript file and extract conversation text.

    Claude Code transcripts contain streaming entries: multiple JSONL lines
    for the same assistant message (identified by message.id). We keep only
    the last (most complete) entry per message ID for assistant messages.
    User messages are keyed by their entry uuid.
    """
    # Collect entries, grouping streamed assistant blocks by message id
    # Each streaming entry for the same msg id carries a different content
    # block (text, thinking, tool_use). We accumulate text blocks.
    grouped: dict[str, tuple[str, list[str]]] = {}  # key -> (role, [texts])
    order: list[str] = []  # insertion-order keys

    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", entry)
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            text = _extract_text(msg.get("content", ""))
            # Filter out empty or placeholder text
            if not text.strip() or text.strip() == "(no content)":
                continue

            # Group by message id for assistant (streaming), uuid for user
            if role == "assistant":
                key = msg.get("id") or entry.get("uuid", "")
            else:
                key = entry.get("uuid", "")

            if not key:
                continue

            if key not in grouped:
                order.append(key)
                grouped[key] = (role, [])
            grouped[key][1].append(text.strip())

    lines = []
    for key in order:
        role, texts = grouped[key]
        # Deduplicate identical text fragments from streaming
        seen_texts: list[str] = []
        for t in texts:
            if t not in seen_texts:
                seen_texts.append(t)
        combined = "\n".join(seen_texts)
        if not combined.strip():
            continue
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {combined}")

    return "\n\n".join(lines)


async def _run(transcript_path: str) -> dict:
    from memory_mcp.embeddings.manager import EmbeddingManager
    from memory_mcp.extraction.extractor import ConversationExtractor
    from memory_mcp.storage.lance_store import LanceStore
    from memory_mcp.storage.profile import UserProfileStore
    from memory_mcp.tools.save import save_conversation_memory

    conversation_text = parse_transcript(transcript_path)
    if not conversation_text:
        return {"status": "empty", "message": "Transcript contained no conversation text."}

    extractor = ConversationExtractor()
    embedder = EmbeddingManager()
    store = LanceStore()
    profile_store = UserProfileStore()

    return await save_conversation_memory(
        conversation_text=conversation_text,
        extractor=extractor,
        embedder=embedder,
        store=store,
        profile_store=profile_store,
    )


def main():
    """Entry point: read hook JSON from stdin, parse transcript, save memory."""
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to parse JSON from stdin")
        sys.exit(1)

    transcript_path = hook_input.get("transcript_path")
    if not transcript_path:
        logger.error("No transcript_path in hook input")
        sys.exit(1)

    try:
        result = asyncio.run(_run(transcript_path))
    except Exception:
        logger.exception("Failed to save conversation memory")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
