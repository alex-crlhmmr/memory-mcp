"""Claude API conversation analysis and observation extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import anthropic

from memory_mcp.config import get_settings
from memory_mcp.models import (
    ExtractionResult,
    Observation,
    ObservationCategory,
    ProfileUpdate,
)

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction system. Analyze the conversation and extract observations about the user.

Extract 5-30 observations across these categories:
- coding_preference: Language, framework, style, tooling preferences
- communication_pattern: How they communicate, explain, ask questions
- code_pattern: Recurring code patterns, architectures, conventions they use/prefer
- technical_context: Their tech stack, environment, project details
- decision: Specific technical decisions made and rationale
- subtle_signal: Implicit preferences, frustrations, working style hints
- project_context: Current projects, goals, deadlines, team structure

For each observation provide:
- category: one of the categories above
- content: clear, specific description of the observation
- confidence: 0.0-1.0 (how certain you are)
- tags: relevant keywords

Also provide:
- conversation_summary: 1-2 sentence summary of the conversation
- topics: list of main topics discussed
- profile_updates: list of stable, high-confidence (>=0.7) preferences to add to the user profile
  Each profile_update has: key (snake_case identifier), value (concise description), confidence

Current user profile (use to detect changes/contradictions):
{user_profile}

Respond with valid JSON matching this schema:
{{
  "observations": [
    {{
      "category": "coding_preference",
      "content": "User prefers...",
      "confidence": 0.9,
      "tags": ["python", "typing"],
      "supersedes": null
    }}
  ],
  "conversation_summary": "...",
  "topics": ["topic1", "topic2"],
  "profile_updates": [
    {{
      "key": "preferred_language",
      "value": "Python with strict typing",
      "confidence": 0.95
    }}
  ]
}}
"""


class ConversationExtractor:
    """Extracts structured observations from conversations using Claude."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: anthropic.Anthropic | None = None

    def _ensure_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = self._settings.anthropic_api_key
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY or MEMORY_MCP_ANTHROPIC_API_KEY must be set"
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    async def extract(
        self, conversation_text: str, user_profile: dict
    ) -> ExtractionResult:
        """Extract observations from conversation text."""
        client = self._ensure_client()
        conversation_id = uuid.uuid4().hex[:16]

        system = EXTRACTION_SYSTEM_PROMPT.format(
            user_profile=json.dumps(user_profile, indent=2) if user_profile else "{}"
        )

        message = client.messages.create(
            model=self._settings.extraction_model,
            max_tokens=8192,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this conversation and extract observations:\n\n{conversation_text}",
                }
            ],
        )

        raw = message.content[0].text
        # Handle markdown code blocks in response
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        data = json.loads(raw)

        observations = []
        for obs_data in data.get("observations", []):
            try:
                category = ObservationCategory(obs_data["category"])
            except ValueError:
                logger.warning("Unknown category: %s", obs_data.get("category"))
                continue

            observations.append(Observation(
                conversation_id=conversation_id,
                category=category,
                content=obs_data["content"],
                confidence=min(max(float(obs_data.get("confidence", 0.5)), 0.0), 1.0),
                tags=obs_data.get("tags", []),
                supersedes=obs_data.get("supersedes"),
            ))

        profile_updates = []
        for pu_data in data.get("profile_updates", []):
            profile_updates.append(ProfileUpdate(
                key=pu_data["key"],
                value=pu_data["value"],
                confidence=min(max(float(pu_data.get("confidence", 0.5)), 0.0), 1.0),
            ))

        return ExtractionResult(
            observations=observations,
            conversation_summary=data.get("conversation_summary", ""),
            topics=data.get("topics", []),
            profile_updates=profile_updates,
        )

    async def extract_streaming(
        self, conversation_text: str, user_profile: dict
    ) -> tuple[asyncio.Queue[Observation | None], asyncio.Future[ExtractionResult]]:
        """Stream extraction: yield observations as they're parsed, return final result.

        Returns a tuple of:
          - queue: observations are put here as they're parsed; None sentinel signals end
          - future: resolves to the full ExtractionResult once the stream is done
        """
        client = self._ensure_client()
        conversation_id = uuid.uuid4().hex[:16]

        system = EXTRACTION_SYSTEM_PROMPT.format(
            user_profile=json.dumps(user_profile, indent=2) if user_profile else "{}"
        )

        queue: asyncio.Queue[Observation | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ExtractionResult] = loop.create_future()

        async def _run_stream() -> None:
            try:
                raw_text = await self._stream_and_parse(
                    client, system, conversation_text, conversation_id, queue
                )
                result = self._parse_final_result(raw_text, conversation_id)
                future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                # Always signal consumer to stop (exactly once)
                await queue.put(None)

        asyncio.create_task(_run_stream())
        return queue, future

    async def _stream_and_parse(
        self,
        client: anthropic.Anthropic,
        system: str,
        conversation_text: str,
        conversation_id: str,
        queue: asyncio.Queue[Observation | None],
    ) -> str:
        """Run the streaming API call, parse observations incrementally, return full text.

        The synchronous stream iteration runs in a thread so the event loop
        stays free to execute embedding tasks concurrently.
        """
        parser = _IncrementalObservationParser(conversation_id)
        loop = asyncio.get_running_loop()

        def _iterate_stream() -> str:
            """Synchronous: open stream, iterate chunks, push observations."""
            buffer = ""
            ctx = client.messages.stream(
                model=self._settings.extraction_model,
                max_tokens=8192,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze this conversation and extract observations:\n\n{conversation_text}",
                    }
                ],
            )
            with ctx as stream:
                for text_chunk in stream.text_stream:
                    buffer += text_chunk
                    new_obs = parser.feed(text_chunk)
                    for obs in new_obs:
                        loop.call_soon_threadsafe(queue.put_nowait, obs)
            return buffer

        raw = await loop.run_in_executor(None, _iterate_stream)

        # Handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        return raw

    def _parse_final_result(
        self, raw_text: str, conversation_id: str
    ) -> ExtractionResult:
        """Parse the complete response into an ExtractionResult."""
        data = json.loads(raw_text)

        observations = []
        for obs_data in data.get("observations", []):
            try:
                category = ObservationCategory(obs_data["category"])
            except ValueError:
                logger.warning("Unknown category: %s", obs_data.get("category"))
                continue
            observations.append(Observation(
                conversation_id=conversation_id,
                category=category,
                content=obs_data["content"],
                confidence=min(max(float(obs_data.get("confidence", 0.5)), 0.0), 1.0),
                tags=obs_data.get("tags", []),
                supersedes=obs_data.get("supersedes"),
            ))

        profile_updates = []
        for pu_data in data.get("profile_updates", []):
            profile_updates.append(ProfileUpdate(
                key=pu_data["key"],
                value=pu_data["value"],
                confidence=min(max(float(pu_data.get("confidence", 0.5)), 0.0), 1.0),
            ))

        return ExtractionResult(
            observations=observations,
            conversation_summary=data.get("conversation_summary", ""),
            topics=data.get("topics", []),
            profile_updates=profile_updates,
        )


class _IncrementalObservationParser:
    """Bracket-matching parser that extracts complete observation objects from streamed JSON.

    Detects when we're inside the "observations" array, tracks brace depth to find
    complete {...} objects, and parses each one with json.loads().
    """

    def __init__(self, conversation_id: str) -> None:
        self._conversation_id = conversation_id
        self._full_buffer = ""
        self._in_observations = False
        self._brace_depth = 0
        self._current_object_start: int | None = None
        self._in_string = False
        self._escape_next = False
        self._scan_pos = 0

    def feed(self, chunk: str) -> list[Observation]:
        """Feed a text chunk, return any newly completed observations."""
        self._full_buffer += chunk
        results: list[Observation] = []

        while self._scan_pos < len(self._full_buffer):
            ch = self._full_buffer[self._scan_pos]

            if not self._in_observations:
                # Look for "observations" key followed by [
                idx = self._full_buffer.find('"observations"', self._scan_pos)
                if idx == -1:
                    self._scan_pos = max(0, len(self._full_buffer) - 20)
                    break
                bracket_idx = self._full_buffer.find("[", idx + len('"observations"'))
                if bracket_idx == -1:
                    self._scan_pos = idx
                    break
                self._in_observations = True
                self._scan_pos = bracket_idx + 1
                continue

            # Handle string state (inside an object)
            if self._in_string:
                if self._escape_next:
                    self._escape_next = False
                elif ch == "\\":
                    self._escape_next = True
                elif ch == '"':
                    self._in_string = False
                self._scan_pos += 1
                continue

            # Outside a string, inside the observations array
            if ch == '"' and self._brace_depth > 0:
                self._in_string = True
            elif ch == "{":
                if self._brace_depth == 0:
                    self._current_object_start = self._scan_pos
                self._brace_depth += 1
            elif ch == "}":
                self._brace_depth -= 1
                if self._brace_depth == 0 and self._current_object_start is not None:
                    obj_str = self._full_buffer[self._current_object_start:self._scan_pos + 1]
                    obs = self._try_parse(obj_str)
                    if obs is not None:
                        results.append(obs)
                    self._current_object_start = None
            elif ch == "]" and self._brace_depth == 0:
                self._in_observations = False
                self._scan_pos += 1
                break

            self._scan_pos += 1

        return results

    def _try_parse(self, obj_str: str) -> Observation | None:
        """Try to parse a JSON string into an Observation."""
        try:
            obs_data = json.loads(obj_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse observation object: %s", obj_str[:100])
            return None

        try:
            category = ObservationCategory(obs_data["category"])
        except (ValueError, KeyError):
            logger.warning("Unknown/missing category in streamed observation")
            return None

        return Observation(
            conversation_id=self._conversation_id,
            category=category,
            content=obs_data.get("content", ""),
            confidence=min(max(float(obs_data.get("confidence", 0.5)), 0.0), 1.0),
            tags=obs_data.get("tags", []),
            supersedes=obs_data.get("supersedes"),
        )
