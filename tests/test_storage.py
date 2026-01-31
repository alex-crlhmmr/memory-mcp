"""Tests for storage layer."""

from __future__ import annotations

import pytest

from memory_mcp.models import (
    ConversationRecord,
    Observation,
    ObservationCategory,
    ProfileUpdate,
)
from memory_mcp.storage.lance_store import LanceStore
from memory_mcp.storage.profile import UserProfileStore


def _make_observation(content: str = "test obs", category: str = "coding_preference") -> Observation:
    return Observation(
        conversation_id="conv-001",
        category=ObservationCategory(category),
        content=content,
        confidence=0.9,
        tags=["test"],
    )


def _make_vector(seed: int = 0) -> list[float]:
    """Create a simple normalized vector."""
    import hashlib
    h = hashlib.sha256(str(seed).encode()).digest()
    vec = [float(b) / 255.0 for b in h]
    vec = (vec * 32)[:1024]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


class TestLanceStore:
    def test_store_and_search(self, tmp_data_dir):
        store = LanceStore()
        obs = _make_observation("User prefers Python")
        vec = _make_vector(42)
        store.store_observations([obs], [vec])

        # Search with same vector should find it
        results = store.search(vec, limit=5, relevance_threshold=0.0)
        assert len(results) >= 1
        assert results[0].content == "User prefers Python"
        assert results[0].observation_id == obs.id

    def test_store_conversation(self, tmp_data_dir):
        store = LanceStore()
        record = ConversationRecord(
            conversation_id="conv-001",
            summary="Test conversation",
            topics=["testing"],
            observation_count=1,
        )
        store.store_conversation(record)
        # No assertion needed; just verifying it doesn't raise

    def test_get_observation(self, tmp_data_dir):
        store = LanceStore()
        obs = _make_observation("Specific observation")
        vec = _make_vector(99)
        store.store_observations([obs], [vec])

        result = store.get_observation(obs.id)
        assert result is not None
        assert result["content"] == "Specific observation"

    def test_get_nonexistent_observation(self, tmp_data_dir):
        store = LanceStore()
        # Ensure table exists first
        obs = _make_observation()
        store.store_observations([obs], [_make_vector()])
        result = store.get_observation("nonexistent-id")
        assert result is None

    def test_delete_observation(self, tmp_data_dir):
        store = LanceStore()
        obs = _make_observation("To be deleted")
        vec = _make_vector(77)
        store.store_observations([obs], [vec])

        assert store.delete_observation(obs.id) is True
        assert store.get_observation(obs.id) is None

    def test_update_observation(self, tmp_data_dir):
        store = LanceStore()
        obs = _make_observation("Original content")
        vec = _make_vector(55)
        store.store_observations([obs], [vec])

        new_vec = _make_vector(56)
        assert store.update_observation_content(obs.id, "Updated content", new_vec) is True

        result = store.get_observation(obs.id)
        assert result is not None
        assert result["content"] == "Updated content"

    def test_search_with_category_filter(self, tmp_data_dir):
        store = LanceStore()
        obs1 = _make_observation("Python preference", "coding_preference")
        obs2 = _make_observation("Jetson context", "technical_context")
        vec1 = _make_vector(1)
        vec2 = _make_vector(2)
        store.store_observations([obs1, obs2], [vec1, vec2])

        results = store.search(
            vec1, limit=10, category="coding_preference", relevance_threshold=0.0
        )
        categories = {r.category for r in results}
        assert categories <= {"coding_preference"}

    def test_store_empty_observations(self, tmp_data_dir):
        store = LanceStore()
        store.store_observations([], [])
        # Should not raise


class TestUserProfileStore:
    def test_load_empty(self, tmp_data_dir):
        ps = UserProfileStore()
        assert ps.load() == {}

    def test_save_and_load(self, tmp_data_dir):
        ps = UserProfileStore()
        ps.save({"language": "Python", "os": "Linux"})
        profile = ps.load()
        assert profile["language"] == "Python"
        assert profile["os"] == "Linux"

    def test_apply_updates_high_confidence(self, tmp_data_dir):
        ps = UserProfileStore()
        updates = [
            ProfileUpdate(key="editor", value="Neovim", confidence=0.9),
            ProfileUpdate(key="low_conf", value="maybe", confidence=0.5),
        ]
        profile = ps.apply_updates(updates)
        assert profile["editor"] == "Neovim"
        assert "low_conf" not in profile  # Below 0.7 threshold

    def test_set_key(self, tmp_data_dir):
        ps = UserProfileStore()
        ps.set_key("theme", "dark")
        assert ps.load()["theme"] == "dark"

    def test_delete_key(self, tmp_data_dir):
        ps = UserProfileStore()
        ps.save({"a": "1", "b": "2"})
        assert ps.delete_key("a") is True
        assert "a" not in ps.load()
        assert ps.delete_key("nonexistent") is False
