"""Configuration with environment variable overrides."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Memory MCP server settings."""

    # Paths
    data_dir: Path = Path("/home/acarlham/memory_mcp/data")
    models_dir: Path = Path("/home/acarlham/memory_mcp/models")

    # Embedding model
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_dim: int = 1024
    gpu_memory_threshold_gb: float = 4.0

    # Extraction
    extraction_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str = ""

    # LanceDB
    lance_db_path: Path = Path("/home/acarlham/memory_mcp/data/lancedb")

    # Profile
    profile_path: Path = Path("/home/acarlham/memory_mcp/data/user_profile.json")

    # Recall defaults
    default_recall_limit: int = 20
    relevance_threshold: float = 0.3

    model_config = {
        "env_prefix": "MEMORY_MCP_",
        "env_file": "/home/acarlham/memory_mcp/.env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
