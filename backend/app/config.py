"""
Configuration management for the Stock Market ML Decision Support System.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Data storage and fetching configuration."""
    data_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("DATA_DIR", "data"))
    raw_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw"))
    processed_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    model_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("MODEL_DIR", "data/models"))
    cache_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("CACHE_DIR", "data/cache"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'market_data.db'}"))

    def __post_init__(self):
        """Create directories if they don't exist."""
        for d in [self.data_dir, self.raw_dir, self.processed_dir, self.model_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class APIKeysConfig:
    """API keys for data providers."""
    alpha_vantage: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", ""))
    fred: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))


@dataclass
class ModelConfig:
    """ML model training configuration."""
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "64")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.001")))
    epochs: int = field(default_factory=lambda: int(os.getenv("EPOCHS", "100")))
    sequence_length: int = field(default_factory=lambda: int(os.getenv("SEQUENCE_LENGTH", "60")))
    train_split: float = field(default_factory=lambda: float(os.getenv("TRAIN_SPLIT", "0.7")))
    val_split: float = field(default_factory=lambda: float(os.getenv("VAL_SPLIT", "0.15")))

    # Model-specific
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3

    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1

    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])

    # Ensemble
    ensemble_method: str = "stacking"  # stacking, voting, blending

    # Optuna
    optuna_n_trials: int = 50
    optuna_timeout: int = 3600  # seconds


@dataclass
class TradingConfig:
    """Trading and risk management configuration."""
    initial_capital: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "100000")))
    transaction_cost_pct: float = field(default_factory=lambda: float(os.getenv("TRANSACTION_COST_PCT", "0.001")))
    slippage_pct: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_PCT", "0.0005")))
    max_position_pct: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.1")))
    risk_free_rate: float = field(default_factory=lambda: float(os.getenv("RISK_FREE_RATE", "0.05")))

    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    max_drawdown_limit: float = 0.15  # 15% max portfolio drawdown
    kelly_fraction: float = 0.5  # Half-Kelly for conservative sizing
    max_correlation: float = 0.7  # Max correlation between positions


@dataclass
class ServerConfig:
    """API server configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))


@dataclass
class AppConfig:
    """Root configuration object."""
    data: DataConfig = field(default_factory=DataConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Default stock universe
    default_tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM",
        "PFE", "CSCO", "INTC", "VZ", "KO", "PEP", "MRK", "ABT", "TMO",
        "COST", "AVGO", "NKE"
    ])

    # Benchmark
    benchmark_ticker: str = "SPY"


# Global configuration instance
config = AppConfig()
