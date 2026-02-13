"""Load-on-demand embedding manager."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from memory_mcp.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages the embedding model with configurable device."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model: SentenceTransformer | None = None
        self._device: str | None = None
        self._lock = asyncio.Lock()

    def _resolve_device(self) -> str:
        """Resolve the device to use based on config."""
        choice = self._settings.embedding_device.lower()
        if choice == "cpu":
            return "cpu"
        if choice == "cuda":
            return "cuda"
        # "auto": use CUDA if available, else CPU
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU for embeddings")
            return "cuda"
        logger.info("CUDA not available, using CPU for embeddings")
        return "cpu"

    def _load_model(self) -> SentenceTransformer:
        """Load model to the configured device (first call downloads/caches weights)."""
        if self._model is None:
            device = self._resolve_device()
            logger.info("Loading embedding model %s to %s...", self._settings.embedding_model, device)
            cache_dir = str(self._settings.models_dir)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self._settings.embedding_model,
                cache_folder=cache_dir,
                device=device,
                truncate_dim=self._settings.embedding_dim,
            )
            self._device = device
            logger.info("Embedding model loaded to %s", device)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts on the configured device."""
        async with self._lock:
            return await asyncio.get_running_loop().run_in_executor(
                None, self._embed_sync, texts
            )

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding on the configured device."""
        model = self._load_model()
        batch_size = 32 if self._device == "cuda" else 16
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        results = await self.embed([query])
        return results[0]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed multiple document strings for storage."""
        if not documents:
            return []
        return await self.embed(documents)
