from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    # --- paths ---
    data_dir: Path = Path("data")

    # --- LLM (Qwen Responses API) ---
    llm_base_url: str = Field(default="", alias="LLM_BASEURL")
    llm_api_key: str = Field(default="", alias="LLM_APIKEY")
    # May hold a comma-separated list of selectable models, e.g.
    # "qwen3.6-plus,qwen3.7-max". The first entry is the default; the full list
    # is offered to the frontend as a dropdown (see `thinking_models`).
    thinking_model: str = Field(default="", alias="THINKING_MODELNAME")
    embed_model: str = Field(default="", alias="EMBED_MODELNAME")
    rerank_model: str = Field(default="", alias="RERANK_MODELNAME")

    # --- Tavily web search (used by publication-rank fallback) ---
    tavily_api_key: str = Field(default="", alias="TAVILY_KEY")

    # --- Publication rank (EasyScholar) ---
    easyscholar_secret_key: str = Field(default="", alias="EASYSCHOLAR_SECRET_KEY")
    easyscholar_api_url: str = Field(
        default="https://www.easyscholar.cc/open/getPublicationRank",
        alias="EASYSCHOLAR_API_URL",
    )

    # --- MinerU ---
    mineru_token: str = Field(default="", alias="MINERU_TOKEN")
    mineru_model_version: str = Field(default="vlm", alias="MINERU_MODEL_VERSION")
    mineru_poll_interval_seconds: int = Field(default=6, alias="MINERU_POLL_INTERVAL_SECONDS")
    mineru_parse_timeout_seconds: int = Field(default=1800, alias="MINERU_PARSE_TIMEOUT_SECONDS")
    mineru_batch_timeout_seconds: int = Field(default=7200, alias="MINERU_BATCH_TIMEOUT_SECONDS")

    # --- Paper downloader (OA Resolver / Elsevier TDM / Wiley TDM) ---
    unpaywall_email: str = Field(default="", alias="UNPAYWALL_EMAIL")
    core_api_key: str = Field(default="", alias="CORE_API_KEY")
    elsevier_api_key: str = Field(default="", alias="ELSEVIER_API_KEY")
    elsevier_inst_token: str = Field(default="", alias="ELSEVIER_INSTTOKEN")
    wiley_tdm_token: str = Field(default="", alias="WILEY_TDM_TOKEN")

    # --- Research Sphere ---
    sphere_radius: int = Field(default=1, alias="SPHERE_RADIUS")
    sphere_candidate_cap: int = Field(default=200, alias="SPHERE_CANDIDATE_CAP")
    sphere_layer1_cap: int = Field(default=40, alias="SPHERE_LAYER1_CAP")
    sphere_pdf_parse_cap: int = Field(default=0, alias="SPHERE_PDF_PARSE_CAP")

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {
        "env_file": str(Path(__file__).resolve().parents[2] / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def thinking_models(self) -> list[str]:
        """Selectable thinking models, parsed from the comma-separated env var."""
        return [m.strip() for m in self.thinking_model.split(",") if m.strip()]

    @property
    def default_thinking_model(self) -> str:
        """First configured model — used when the caller does not pick one."""
        models = self.thinking_models
        return models[0] if models else ""


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
