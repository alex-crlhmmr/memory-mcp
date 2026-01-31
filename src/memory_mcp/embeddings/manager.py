"""Load-on-demand Qwen3-Embedding-8B embedding manager."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from memory_mcp.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages the Qwen3-Embedding-8B model with GPU load-on-demand."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    def _load_model(self) -> SentenceTransformer:
        """Load model to CPU (first call downloads/caches weights)."""
        if self._model is None:
            logger.info("Loading embedding model %s to CPU...", self._settings.embedding_model)
            cache_dir = str(self._settings.models_dir)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self._settings.embedding_model,
                cache_folder=cache_dir,
                device="cpu",
                truncate_dim=self._settings.embedding_dim,
            )
            logger.info("Embedding model loaded to CPU")
        return self._model

    def _gpu_available(self) -> bool:
        """Check if CUDA is available and has enough free memory."""
        if not torch.cuda.is_available():
            return False
        try:
            free, _ = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
            threshold = self._settings.gpu_memory_threshold_gb
            logger.debug("GPU free memory: %.2f GB (threshold: %.1f GB)", free_gb, threshold)
            return free_gb > threshold
        except Exception:
            logger.debug("Could not query GPU memory", exc_info=True)
            return False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Qwen3 with GPU acceleration when available.

        Uses instruct prompt format for retrieval-quality embeddings.
        """
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._embed_sync, texts
            )

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding with GPU load/offload."""
        model = self._load_model()
        use_gpu = self._gpu_available()

        if use_gpu:
            logger.info("Moving embedding model to CUDA")
            model = model.to("cuda")

        try:
            embeddings = model.encode(
                texts,
                batch_size=8 if use_gpu else 2,
                show_progress_bar=False,
                normalize_embeddings=True,
                prompt_name="query",
            )
            return embeddings.tolist()
        finally:
            if use_gpu:
                logger.info("Offloading embedding model to CPU")
                model = model.to("cpu")
                torch.cuda.empty_cache()

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        results = await self.embed([query])
        return results[0]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed multiple document strings for storage."""
        if not documents:
            return []
        return await self.embed(documents)
