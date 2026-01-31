"""LanceDB vector storage for observations and conversations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import lancedb
import pyarrow as pa

from memory_mcp.config import get_settings
from memory_mcp.models import ConversationRecord, MemoryResult, Observation

logger = logging.getLogger(__name__)

def _observations_schema() -> pa.Schema:
    dim = get_settings().embedding_dim
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("category", pa.string()),
        pa.field("content", pa.string()),
        pa.field("confidence", pa.float32()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("timestamp", pa.string()),
        pa.field("supersedes", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


CONVERSATIONS_SCHEMA = pa.schema([
    pa.field("conversation_id", pa.string()),
    pa.field("summary", pa.string()),
    pa.field("topics", pa.list_(pa.string())),
    pa.field("timestamp", pa.string()),
    pa.field("observation_count", pa.int32()),
])


class LanceStore:
    """LanceDB-backed vector store for memory observations."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._db: lancedb.DBConnection | None = None
        self._obs_table: lancedb.table.Table | None = None
        self._conv_table: lancedb.table.Table | None = None

    def _ensure_db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._settings.lance_db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self._settings.lance_db_path))
        return self._db

    def _table_names(self) -> list[str]:
        """Get table names, handling both old and new lancedb API."""
        db = self._ensure_db()
        result = db.list_tables()
        # lancedb >= 0.27 returns ListTablesResponse with a .tables attribute
        if hasattr(result, "tables"):
            return result.tables
        return list(result)

    def _ensure_observations_table(self) -> lancedb.table.Table:
        if self._obs_table is None:
            db = self._ensure_db()
            if "observations" in self._table_names():
                self._obs_table = db.open_table("observations")
            else:
                schema = _observations_schema()
                empty = pa.table(
                    {f.name: pa.array([], type=f.type) for f in schema},
                    schema=schema,
                )
                self._obs_table = db.create_table("observations", data=empty)
        return self._obs_table

    def _ensure_conversations_table(self) -> lancedb.table.Table:
        if self._conv_table is None:
            db = self._ensure_db()
            if "conversations" in self._table_names():
                self._conv_table = db.open_table("conversations")
            else:
                empty = pa.table(
                    {f.name: pa.array([], type=f.type) for f in CONVERSATIONS_SCHEMA},
                    schema=CONVERSATIONS_SCHEMA,
                )
                self._conv_table = db.create_table("conversations", data=empty)
        return self._conv_table

    def store_observations(
        self, observations: list[Observation], vectors: list[list[float]]
    ) -> None:
        """Store observations with their embedding vectors."""
        if not observations:
            return

        table = self._ensure_observations_table()
        rows = []
        for obs, vec in zip(observations, vectors):
            rows.append({
                "id": obs.id,
                "conversation_id": obs.conversation_id,
                "category": obs.category.value,
                "content": obs.content,
                "confidence": obs.confidence,
                "tags": obs.tags,
                "timestamp": obs.timestamp.isoformat(),
                "supersedes": obs.supersedes or "",
                "vector": vec,
            })
        table.add(rows)
        logger.info("Stored %d observations", len(observations))

    def store_conversation(self, record: ConversationRecord) -> None:
        """Store conversation metadata."""
        table = self._ensure_conversations_table()
        table.add([{
            "conversation_id": record.conversation_id,
            "summary": record.summary,
            "topics": record.topics,
            "timestamp": record.timestamp.isoformat(),
            "observation_count": record.observation_count,
        }])

    def search(
        self,
        query_vector: list[float],
        limit: int = 20,
        category: Optional[str] = None,
        relevance_threshold: float = 0.3,
    ) -> list[MemoryResult]:
        """Search observations by vector similarity."""
        table = self._ensure_observations_table()

        try:
            query = table.search(query_vector).limit(limit * 2)
            if category:
                query = query.where(f"category = '{category}'")
            results = query.to_pandas()
        except Exception:
            logger.exception("Search failed")
            return []

        if results.empty:
            return []

        memories = []
        for _, row in results.iterrows():
            score = 1.0 - row.get("_distance", 1.0)
            if score < relevance_threshold:
                continue
            memories.append(MemoryResult(
                observation_id=row["id"],
                category=row["category"],
                content=row["content"],
                confidence=row["confidence"],
                tags=row["tags"] if isinstance(row["tags"], list) else [],
                timestamp=row["timestamp"],
                relevance_score=round(score, 4),
            ))
        memories.sort(key=lambda m: m.relevance_score, reverse=True)
        return memories[:limit]

    def get_observation(self, observation_id: str) -> Optional[dict]:
        """Get a single observation by ID."""
        table = self._ensure_observations_table()
        try:
            results = (
                table.search()
                .where(f"id = '{observation_id}'")
                .limit(1)
                .to_arrow()
            )
            if results.num_rows == 0:
                return None
            row = results.to_pydict()
            return {k: v[0] for k, v in row.items()}
        except Exception:
            logger.exception("Failed to get observation %s", observation_id)
            return None

    def delete_observation(self, observation_id: str) -> bool:
        """Delete an observation by ID."""
        table = self._ensure_observations_table()
        try:
            table.delete(f"id = '{observation_id}'")
            return True
        except Exception:
            logger.exception("Failed to delete observation %s", observation_id)
            return False

    def update_observation_content(
        self, observation_id: str, new_content: str, new_vector: list[float]
    ) -> bool:
        """Update an observation's content and re-embed."""
        table = self._ensure_observations_table()
        try:
            table.update(
                where=f"id = '{observation_id}'",
                values={
                    "content": new_content,
                    "vector": new_vector,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return True
        except Exception:
            logger.exception("Failed to update observation %s", observation_id)
            return False
