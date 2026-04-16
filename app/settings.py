from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    finsight_env: str = "development"
    log_level: str = "info"

    database_url: str = "postgresql+asyncpg://finsight:finsight@localhost:5432/finsight"

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://ollama:11434"
    embed_model: str = "nomic-embed-text"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    customer_mcp_url: str = "http://customer-mcp:7001"
    market_mcp_url: str = "http://market-mcp:7002"
    compliance_mcp_url: str = "http://compliance-mcp:7003"

    mlflow_tracking_uri: str = "http://mlflow:5000"

    otel_sdk_disabled: str = "false"
    otel_traces_exporter: str = "otlp"
    otel_metrics_exporter: str = "otlp"
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_service_name: str = "finsight-agents"

    default_prompt_version: str = "v1"

    @property
    def is_test(self) -> bool:
        return self.finsight_env == "test"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
