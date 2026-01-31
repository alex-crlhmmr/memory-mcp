"""Claude API conversation analysis and observation extraction."""

from __future__ import annotations

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
