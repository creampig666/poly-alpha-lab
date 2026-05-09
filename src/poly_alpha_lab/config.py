"""Application configuration loaded from environment variables."""

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for read-only Gamma API research."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="POLYMARKET_",
        extra="ignore",
    )

    gamma_base_url: AnyHttpUrl = "https://gamma-api.polymarket.com"
    clob_base_url: AnyHttpUrl = "https://clob.polymarket.com"
    http_timeout_seconds: float = Field(default=15.0, gt=0)
    user_agent: str = "poly-alpha-lab/1.3.2"
    default_limit: int = Field(default=20, ge=1, le=500)
    min_liquidity: float = Field(default=1000.0, ge=0)
    min_net_edge: float = Field(default=0.03, ge=0)
    default_position_size: float = Field(default=100.0, gt=0)
    default_fair_yes_probability: float = Field(default=0.55, ge=0, le=1)
    journal_db_path: str = "data/poly_alpha_lab.sqlite"
    weather_backtest_db_path: str = "data/weather_backtest.sqlite"


settings = Settings()
