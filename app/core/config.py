"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Flokka Ingestion Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    redis_url: str = "redis://localhost:6379/0"

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "flokka_documents"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    default_chunk_size: int = 512
    default_chunk_overlap: int = 64

    max_upload_size_mb: int = 50
    upload_dir: str = "./uploads"


settings = Settings()
