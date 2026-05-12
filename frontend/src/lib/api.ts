/**
 * API client for the StockML backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface StockSignal {
  ticker: string;
  timestamp: string;
  action: "BUY" | "SELL" | "HOLD";
  confidence: number;
  direction_prob: number;
  expected_return: number | null;
  risk_score: number;
  model_agreement: number;
  features_summary: Record<string, number>;
}

export interface PriceData {
  current: number;
  change_1d: number;
  change_5d: number;
  change_20d: number;
}

export interface SignalResponse {
  signal: StockSignal;
  price: PriceData;
  features: Record<string, number>;
  meta: {
    data_points: number;
    latest_date: string;
    risk_tolerance: string;
    signal_source?: "ml_ensemble" | "statistical_fallback";
    individual_predictions?: Record<string, number> | null;
  };
}

export interface ModelInfo {
  available: boolean;
  model_dir: string;
  n_features: number;
  seq_length: number;
  loaded_models: string[];
  thresholds: { buy: number; sell: number };
  report: {
    trained_at?: string;
    ensemble_test_auc?: number;
    ensemble_test_brier?: number;
    gates_passed?: boolean;
    tickers?: string[];
  };
}

export interface BacktestMetrics {
  total_return: number;
  cagr: number;
  annual_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  total_trades: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
}

export interface BacktestTrade {
  ticker: string;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  direction: string;
  pnl: number;
  return_pct: number;
  holding_days: number;
  exit_reason: string;
}

export interface BacktestResult {
  ticker: string;
  period: { start: string; end: string };
  metrics: BacktestMetrics;
  buy_hold_metrics: BacktestMetrics;
  equity_curve: Record<string, number>;
  trades: BacktestTrade[];
  total_trades: number;
  initial_capital: number;
  final_capital: number;
  signal_source?: "ml_ensemble" | "statistical_fallback";
}

export interface StockData {
  ticker: string;
  source: string;
  rows: number;
  start: string;
  end: string;
  data: {
    dates: string[];
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume: number[];
  };
  latest: { close: number; change: number; volume: number };
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  // For GET requests we don't need a JSON Content-Type; only inject it when
  // a body is present (POST/PUT/etc.) so plain GETs aren't tagged with a
  // meaningless content type (helps with CORS preflights too).
  const hasBody = !!options?.body;
  const headers: Record<string, string> = hasBody
    ? { "Content-Type": "application/json", ...(options?.headers as Record<string, string> ?? {}) }
    : { ...(options?.headers as Record<string, string> ?? {}) };

  const res = await fetch(`${API_BASE}${endpoint}`, { ...options, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }

  return res.json();
}

// Encode a ticker safely for URLs. Tickers like "BRK.B" or "RELIANCE.NS"
// contain dots and dashes which are URL-safe, but defensive encoding catches
// stranger inputs without breaking valid tickers.
const encT = (t: string) => encodeURIComponent(t);

export const api = {
  // Data endpoints
  fetchStock: (ticker: string, start = "2015-01-01", signal?: AbortSignal) =>
    fetchAPI<StockData>(`/api/data/fetch/${encT(ticker)}?start=${start}`, { signal }),

  searchStocks: (query: string, signal?: AbortSignal) =>
    fetchAPI<
      Array<{
        ticker: string;
        name: string;
        exchange: string;
        type: string;
        sector?: string;
        industry?: string;
      }>
    >(`/api/data/search?q=${encodeURIComponent(query)}`, { signal }),

  getFundamentals: (ticker: string, signal?: AbortSignal) =>
    fetchAPI<Record<string, unknown>>(`/api/data/fundamentals/${encT(ticker)}`, { signal }),

  // Signal endpoints
  getSignal: (ticker: string, riskTolerance = "medium", signal?: AbortSignal) =>
    fetchAPI<SignalResponse>(
      `/api/signals/generate/${encT(ticker)}?risk_tolerance=${riskTolerance}`,
      { signal },
    ),

  getBatchSignals: (tickers: string[], riskTolerance = "medium", signal?: AbortSignal) =>
    fetchAPI<{ signals: SignalResponse[]; count: number }>(
      `/api/signals/batch?tickers=${tickers.map(encT).join(",")}&risk_tolerance=${riskTolerance}`,
      { signal },
    ),

  getModelInfo: () => fetchAPI<ModelInfo>("/api/signals/model-info"),

  reloadModels: () =>
    fetchAPI<{ reloaded: boolean; available: boolean; info: ModelInfo }>(
      "/api/signals/reload-models",
      { method: "POST" },
    ),

  // Portfolio endpoints
  configurePortfolio: (cfg: Record<string, number>) =>
    fetchAPI<Record<string, unknown>>("/api/portfolio/configure", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),

  getPortfolioStatus: () =>
    fetchAPI<Record<string, unknown>>("/api/portfolio/status"),

  getPositionSize: (params: Record<string, number>) => {
    const qs = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)])
    ).toString();
    return fetchAPI<Record<string, unknown>>(`/api/portfolio/position-size?${qs}`);
  },

  // Backtest endpoints
  runBacktest: (
    ticker: string,
    params: Record<string, string | number> = {},
    signal?: AbortSignal,
  ) => {
    const qs = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)])
    ).toString();
    return fetchAPI<BacktestResult>(`/api/backtest/run/${encT(ticker)}?${qs}`, { signal });
  },
};
