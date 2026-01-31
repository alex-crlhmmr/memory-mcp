"""Data models for the memory MCP server."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ObservationCategory(str, Enum):
    CODING_PREFERENCE = "coding_preference"
    COMMUNICATION_PATTERN = "communication_pattern"
    CODE_PATTERN = "code_pattern"
    TECHNICAL_CONTEXT = "technical_context"
    DECISION = "decision"
    SUBTLE_SIGNAL = "subtle_signal"
    PROJECT_CONTEXT = "project_context"


class Observation(BaseModel):
    """A single extracted observation from a conversation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    conversation_id: str
    category: ObservationCategory
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    supersedes: Optional[str] = None


class ConversationRecord(BaseModel):
    """Metadata for a processed conversation."""

    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    summary: str
    topics: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observation_count: int = 0


class ProfileUpdate(BaseModel):
    """An update to the aggregated user profile."""

    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Result from the Claude API extraction pipeline."""

    observations: list[Observation]
    conversation_summary: str
    topics: list[str]
    profile_updates: list[ProfileUpdate] = Field(default_factory=list)


class MemoryResult(BaseModel):
    """A single memory returned from recall."""

    observation_id: str
    category: str
    content: str
    confidence: float
    tags: list[str]
    timestamp: str
    relevance_score: float


class RecallResponse(BaseModel):
    """Response from the recall_memories tool."""

    user_profile: dict
    memories: list[MemoryResult]
    total_found: int
