# Stock Market ML Decision Support System

> **⚠️ Disclaimer:** Stock markets are stochastic and partially efficient. No model guarantees profit. This system provides probabilistic decision support — not financial advice. Focus on risk-adjusted returns, not certainty.

## 🏗️ Architecture

A production-grade ML pipeline that:
- Fetches real-time and historical stock market data (Alpha Vantage 🥇 + Yahoo Finance 🥈 + FRED)
- Engineers 100+ meaningful financial features (technical, statistical, regime, macro)
- Trains state-of-the-art ML/DL models (LSTM, Transformer, CNN, XGBoost, LightGBM, CatBoost)
- Outputs actionable signals (BUY / SELL / HOLD + confidence score)
- Includes backtesting with realistic market simulation
- Provides an intuitive dark-theme UI for monitoring and decision-making

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- (Optional) CUDA-capable GPU

### Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Copy and configure environment
cp ../.env.example ../.env
# Edit .env with your API keys (optional but recommended)

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### API Keys

| Service | Priority | Purpose | Get Key |
|---------|----------|---------|---------|
| Alpha Vantage | 🥇 Primary | Market data, search, fundamentals | https://www.alphavantage.co/support/ |
| FRED | — | Macroeconomic indicators | https://fred.stlouisfed.org/docs/api/api_key.html |
| Yahoo Finance | 🥈 Fallback | Missing data recovery, sanity checks | No key needed |

> Alpha Vantage is the **primary** data source (official API, stable). Yahoo Finance is used as a **fallback** for missing data recovery and sanity checks. The system gracefully degrades to Yahoo-only if no Alpha Vantage key is configured.

## 📊 Features

### Signal Generation
- **6 model types**: LSTM, GRU, Temporal Fusion Transformer, CNN-1D, XGBoost, LightGBM, CatBoost
- **Meta-ensemble**: Stacking with calibrated logistic regression
- **100+ features**: Technical indicators, statistical features, regime detection, macro data
- **Walk-forward validation**: No data leakage, time-series aware CV

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
- Indian stocks (NSE via .NS suffix, e.g., RELIANCE.NS)
- Forex & macro data (via Alpha Vantage)
- Any ticker supported by Alpha Vantage / Yahoo Finance

## 🧠 Model Architecture

```
Input (OHLCV + Macro) → Feature Engineering (100+ features) → 
├── LSTM (Bidirectional + Attention)
├── Temporal Fusion Transformer (LSTM + Multi-Head Attention + GRN)
├── Multi-Scale CNN-1D (3/7/15-day kernels + Squeeze-Excitation)
├── XGBoost
├── LightGBM
└── CatBoost
    ↓
Meta-Ensemble (Calibrated Stacking) → Signal (BUY/SELL/HOLD + Confidence)
    ↓
Risk Manager (Kelly Criterion + Stop-Loss) → Position Size
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data/fetch/{ticker}` | GET | Fetch OHLCV data |
| `/api/data/search?q=` | GET | Search stocks |
| `/api/data/fundamentals/{ticker}` | GET | Get fundamentals |
| `/api/signals/generate/{ticker}` | GET | Generate ML signal |
| `/api/signals/batch?tickers=` | GET | Batch signals |
| `/api/portfolio/configure` | POST | Set portfolio params |
| `/api/portfolio/position-size` | GET | Kelly sizing |
| `/api/backtest/run/{ticker}` | GET | Run backtest |

## 🛡️ Anti-Overfitting Measures

1. **Walk-forward validation** — no future data leakage
2. **Regularization** — dropout, L2, early stopping
3. **Ensemble diversity** — 6 different model architectures
4. **Probability calibration** — isotonic regression on outputs
5. **Sharpe-aware loss** — optimizes risk-adjusted returns, not just accuracy
6. **Robust scaling** — handles outliers in financial data
