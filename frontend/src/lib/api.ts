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
  meta: { data_points: number; latest_date: string; risk_tolerance: string };
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

export interface BacktestResult {
  ticker: string;
  period: { start: string; end: string };
  metrics: BacktestMetrics;
  buy_hold_metrics: BacktestMetrics;
  equity_curve: Record<string, number>;
  trades: Array<Record<string, unknown>>;
  total_trades: number;
  initial_capital: number;
  final_capital: number;
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
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export const api = {
  // Data endpoints
  fetchStock: (ticker: string, start = "2015-01-01") =>
    fetchAPI<StockData>(`/api/data/fetch/${ticker}?start=${start}`),

  searchStocks: (query: string) =>
    fetchAPI<Array<Record<string, unknown>>>(`/api/data/search?q=${query}`),

  getFundamentals: (ticker: string) =>
    fetchAPI<Record<string, unknown>>(`/api/data/fundamentals/${ticker}`),

  // Signal endpoints
  getSignal: (ticker: string, riskTolerance = "medium") =>
    fetchAPI<SignalResponse>(
      `/api/signals/generate/${ticker}?risk_tolerance=${riskTolerance}`
    ),

  getBatchSignals: (tickers: string[], riskTolerance = "medium") =>
    fetchAPI<{ signals: SignalResponse[]; count: number }>(
      `/api/signals/batch?tickers=${tickers.join(",")}&risk_tolerance=${riskTolerance}`
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
  runBacktest: (ticker: string, params: Record<string, string | number> = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)])
    ).toString();
    return fetchAPI<BacktestResult>(`/api/backtest/run/${ticker}?${qs}`);
  },
};
