"""Aggregated user profile stored as a JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from memory_mcp.config import get_settings
from memory_mcp.models import ProfileUpdate

logger = logging.getLogger(__name__)


class UserProfileStore:
    """Manages the aggregated user profile on disk."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._path: Path = self._settings.profile_path

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """Load the user profile from disk."""
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load user profile")
            return {}

    def save(self, profile: dict[str, Any]) -> None:
        """Save the user profile to disk."""
        self._ensure_dir()
        self._path.write_text(json.dumps(profile, indent=2, default=str))

    def apply_updates(self, updates: list[ProfileUpdate]) -> dict[str, Any]:
        """Apply profile updates and return the updated profile."""
        profile = self.load()
        for update in updates:
            if update.confidence >= 0.7:
                profile[update.key] = update.value
                logger.info("Profile updated: %s = %s", update.key, update.value)
        self.save(profile)
        return profile

    def set_key(self, key: str, value: str) -> dict[str, Any]:
        """Set a single profile key."""
        profile = self.load()
        profile[key] = value
        self.save(profile)
        return profile

    def delete_key(self, key: str) -> bool:
        """Delete a profile key. Returns True if key existed."""
        profile = self.load()
        if key in profile:
            del profile[key]
            self.save(profile)
            return True
        return False
