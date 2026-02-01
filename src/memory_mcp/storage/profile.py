"""Aggregated user profile stored as a JSON file."""

from __future__ import annotations

import asyncio
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

    # -- Async public API --

    async def load(self) -> dict[str, Any]:
        """Load the user profile from disk."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._load_sync)

    async def save(self, profile: dict[str, Any]) -> None:
        """Save the user profile to disk."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync, profile)

    async def apply_updates(self, updates: list[ProfileUpdate]) -> dict[str, Any]:
        """Apply profile updates and return the updated profile."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._apply_updates_sync, updates)

    async def set_key(self, key: str, value: str) -> dict[str, Any]:
        """Set a single profile key."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._set_key_sync, key, value)

    async def delete_key(self, key: str) -> bool:
        """Delete a profile key. Returns True if key existed."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._delete_key_sync, key)

    # -- Sync implementations --

    def _load_sync(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load user profile")
            return {}

    def _save_sync(self, profile: dict[str, Any]) -> None:
        self._ensure_dir()
        self._path.write_text(json.dumps(profile, indent=2, default=str))

    def _apply_updates_sync(self, updates: list[ProfileUpdate]) -> dict[str, Any]:
        profile = self._load_sync()
        for update in updates:
            if update.confidence >= 0.7:
                profile[update.key] = update.value
                logger.info("Profile updated: %s = %s", update.key, update.value)
        self._save_sync(profile)
        return profile

    def _set_key_sync(self, key: str, value: str) -> dict[str, Any]:
        profile = self._load_sync()
        profile[key] = value
        self._save_sync(profile)
        return profile

    def _delete_key_sync(self, key: str) -> bool:
        profile = self._load_sync()
        if key in profile:
            del profile[key]
            self._save_sync(profile)
            return True
        return False
