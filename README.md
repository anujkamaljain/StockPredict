# Stock Market ML Decision Support System

> **Disclaimer:** Stock markets are stochastic and partially efficient. No model guarantees profit. This system provides probabilistic decision support — not financial advice. Focus on risk-adjusted returns, not certainty.

## Architecture

A production-grade ML pipeline that:
- Fetches real-time and historical stock market data (Alpha Vantage primary + Yahoo Finance fallback + FRED for macro)
- Engineers 160+ meaningful financial features (technical, statistical, regime, time, lag, macro)
- Trains six diverse ML models on cloud GPUs (LSTM, Transformer, CNN, XGBoost, LightGBM, CatBoost) and stacks them into a calibrated ensemble
- Enforces **mandatory acceptance gates** before any model is considered "trained"
- Outputs actionable signals (BUY / SELL / HOLD + confidence score)
- Includes backtesting with realistic market simulation
- Provides an intuitive dark-theme UI for monitoring and decision-making

## End-to-End Workflow

You only have two things to do:

| Step | Read this | Outcome |
|------|-----------|---------|
| **1. Train models** | [`cloud_training_guide.md`](./cloud_training_guide.md) | Trained models in `backend/data/models/` with acceptance gates passed |
| **2. Deploy** | [`deployment_guide.md`](./deployment_guide.md) | Live app on Vercel + Cloud Run |

Both guides are step-by-step with copy-paste commands.

## Quick local development (optional, while waiting for training)

If you want to run the app locally even **before** training, the API gracefully
falls back to a statistical heuristic so the UI still works (all signals will
just be HOLD — that's expected without trained models).

### Backend

```bash
cd backend
pip install -r requirements.txt
copy ..\.env.example .env       # then edit .env with your API keys
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

### API Keys

| Service | Priority | Purpose | Get Key |
|---------|----------|---------|---------|
| Alpha Vantage | Primary | Market data, search, fundamentals | https://www.alphavantage.co/support/ |
| FRED | — | Macroeconomic indicators | https://fred.stlouisfed.org/docs/api/api_key.html |
| Yahoo Finance | Fallback | Missing data recovery, sanity checks | No key needed |

> Alpha Vantage is the **primary** data source (official API, stable). Yahoo Finance is used as a **fallback** for missing data recovery and sanity checks. The system gracefully degrades to Yahoo-only if no Alpha Vantage key is configured.

## Features

### Signal Generation
- **6 model types**: LSTM, Temporal Fusion Transformer, CNN-1D, XGBoost, LightGBM, CatBoost
- **Stacking meta-ensemble**: regularized logistic regression + cross-validated isotonic calibration
- **160+ engineered features**: trend, momentum, volatility, volume, candlestick, lag, regime, time, macro
- **Top-N feature selection** via XGBoost importance (trained on TRAIN split only)

### Robustness (built into `train_cloud.py`)
- **Per-ticker sequence construction** — no cross-ticker contamination in 60-day windows
- **Aligned tree + deep predictions** — no off-by-`seq_length` ensemble bugs
- **Reproducibility seeds** (`torch`, `numpy`, `random`, `cudnn.deterministic`)
- **Scaler fit on TRAIN only** — no test leakage
- **Purged gap** between train/val/test splits — no rolling-window leakage
- **Early stopping on validation AUC** (not just loss) — robust selection
- **Label smoothing** + gradient clipping + AdamW with weight decay
- **LR warmup + cosine annealing with warm restarts**
- **Mixed precision (AMP)** on CUDA with safe fallback
- **VRAM-safe batched inference** — works on 4 GB laptop GPUs
- **Cross-validated isotonic calibration** on ensemble output
- **Threshold optimization** for F1 / accuracy / balanced
- **Trading-style validation** (simulated hit-rate, sharpe-like, trade-rate)
- **Permutation-importance sanity check** — confirms the model uses real signal

### Mandatory Acceptance Gates
Training exits non-zero unless:
- Test AUC ≥ 0.52 (better than coin flip)
- Test accuracy ≥ 0.51
- Test Brier ≤ 0.26 (better than uniform)
- Train–Test AUC gap ≤ 0.10 (no severe overfitting)
- Each individual model has test AUC ≥ 0.50
- Ensemble AUC ≥ best individual AUC − 0.005 (ensemble adds value)
- Permutation-importance AUC drop ≥ 0.005

Every successful run produces a `training_report.json` you can inspect and version-control.

### Risk Management
- Kelly Criterion position sizing (half-Kelly for conservatism)
- Stop-loss and trailing stop logic
- Maximum drawdown limits
- Portfolio-level exposure constraints

### Backtesting
- Realistic simulation with transaction costs and slippage
- Comprehensive metrics: Sharpe, Sortino, Calmar, CAGR, Max Drawdown
- Buy-and-hold comparison
- Trade-by-trade analysis

### Markets Supported
- US stocks (NYSE, NASDAQ)
- Indian stocks (NSE via `.NS` suffix, e.g., `RELIANCE.NS`)
- Forex & macro data (via Alpha Vantage)
- Any ticker supported by Alpha Vantage / Yahoo Finance

## Model Architecture

```
Input (OHLCV + Macro) → Feature Engineering (160+ features) → Feature Selection (top N)
    ↓
├── LSTM (Bidirectional + Attention)
├── Temporal Fusion Transformer (LSTM + Multi-Head Attention + GRN)
├── Multi-Scale CNN-1D (3/7/15-day kernels + Squeeze-Excitation)
├── XGBoost
├── LightGBM
└── CatBoost
    ↓
Meta-Ensemble (Stacking + Calibrated Isotonic) → Signal (BUY/SELL/HOLD + Confidence)
    ↓
Risk Manager (Kelly Criterion + Stop-Loss) → Position Size
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data/fetch/{ticker}` | GET | Fetch OHLCV data |
| `/api/data/search?q=` | GET | Search stocks |
| `/api/data/fundamentals/{ticker}` | GET | Get fundamentals |
| `/api/signals/generate/{ticker}` | GET | Generate ML signal (auto-uses trained models if present) |
| `/api/signals/batch?tickers=` | GET | Batch signals |
| `/api/signals/model-info` | GET | Inspect which trained models are loaded |
| `/api/signals/reload-models` | POST | Hot-reload models from `data/models/` |
| `/api/portfolio/configure` | POST | Set portfolio params |
| `/api/portfolio/position-size` | GET | Kelly sizing |
| `/api/backtest/run/{ticker}` | GET | Run backtest |

## Project Layout

```
StockMarketPredectior/
├── backend/
│   ├── app/                       # FastAPI application (inference + routes)
│   │   ├── api/routes/            # Data, signals, portfolio, backtest endpoints
│   │   ├── backtest/              # Backtest engine
│   │   ├── data/                  # Providers (alpha_vantage, yahoo, fred), validation, ingestion
│   │   ├── features/              # technical, statistical, regime, engineering
│   │   ├── models/                # lstm, transformer, cnn, tree_models, ensemble
│   │   ├── risk/                  # Kelly-based risk manager
│   │   ├── signals/               # Signal generator
│   │   ├── config.py              # Env-based configuration
│   │   └── main.py                # FastAPI entrypoint
│   ├── data/
│   │   ├── cache/                 # Provider caches (gitignored)
│   │   ├── logs/                  # App logs (gitignored)
│   │   └── models/                # Trained model files (drop trained_models/ here)
│   ├── requirements.txt           # Modern stack (Colab, local)
│   ├── requirements_dgx.txt       # DGX-friendly pinned wheels
│   └── train_cloud.py             # Robust training script (Colab + DGX)
├── frontend/                      # Next.js dark-theme UI
├── cloud_training_guide.md        # How to train on Colab or DGX
├── deployment_guide.md            # How to deploy backend + frontend
└── README.md                      # This file
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [`cloud_training_guide.md`](./cloud_training_guide.md) | Train models on Colab (modern) or DGX (pinned wheels) and verify acceptance gates |
| [`deployment_guide.md`](./deployment_guide.md) | Deploy backend on GCP Cloud Run and frontend on Vercel |
