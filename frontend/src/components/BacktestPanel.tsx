"use client";

import { useState, useMemo, useEffect } from "react";
import { Loader2, Play, Brain, Sigma, AlertCircle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import { api, type BacktestResult } from "@/lib/api";
import { getCurrencySymbol } from "@/lib/currency";

interface Props {
  ticker: string;
  result: BacktestResult | null;
  onRunBacktest: (result: BacktestResult | null) => void;
  capital: number;
  setCapital: (v: number) => void;
  riskTolerance: string;
}

export function BacktestPanel({
  ticker,
  result,
  onRunBacktest,
  capital,
  setCapital,
  riskTolerance,
}: Props) {
  const currency = getCurrencySymbol(ticker);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [startDate, setStartDate] = useState("2018-01-01");
  // Local string buffer so the user can clear / type freely without state
  // flickering back to a formatted value mid-edit.
  const [capitalInput, setCapitalInput] = useState(String(capital));

  // Keep the buffer in sync if the parent capital changes (e.g. from
  // Portfolio Settings tab).
  useEffect(() => {
    setCapitalInput(String(capital));
  }, [capital]);

  // Track the inputs used for the currently-displayed result so we can
  // show a "settings changed" warning + clear stale chart data.
  const [lastRun, setLastRun] = useState<{ start: string; capital: number; risk: string } | null>(null);
  const inputsChanged =
    !!result &&
    !!lastRun &&
    (lastRun.start !== startDate ||
      lastRun.capital !== capital ||
      lastRun.risk !== riskTolerance);

  // The moment any input differs from the last run, drop the stale result
  // so the chart and metrics can't mislead the user.
  useEffect(() => {
    if (inputsChanged) {
      onRunBacktest(null);
    }
  }, [inputsChanged, onRunBacktest]);

  const handleRun = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.runBacktest(ticker, {
        start: startDate,
        initial_capital: capital,
        risk_tolerance: riskTolerance,
      });
      onRunBacktest(res);
      setLastRun({ start: startDate, capital, risk: riskTolerance });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  };

  const handleCapitalChange = (raw: string) => {
    // Allow empty / partial input while typing; strip everything except digits.
    const cleaned = raw.replace(/[^0-9]/g, "");
    setCapitalInput(cleaned);
    const parsed = parseInt(cleaned, 10);
    if (!Number.isNaN(parsed) && parsed > 0) {
      setCapital(parsed);
    }
  };

  const capitalPresets = [10000, 50000, 100000, 500000, 1000000];

  const chartData = useMemo(() => {
    if (!result) return [];
    // Keep FULL date as the dataKey so Recharts identifies each point uniquely.
    // Truncating to MM-DD (the previous behavior) caused two different years'
    // entries to collide → hover would always show the earliest matching year.
    const entries = Object.entries(result.equity_curve).sort(
      ([a], [b]) => a.localeCompare(b),
    );
    // Downsample to ~250 points for chart performance while preserving recency.
    const step = Math.max(1, Math.ceil(entries.length / 250));
    return entries
      .filter((_, i) => i % step === 0 || i === entries.length - 1)
      .map(([date, value]) => ({
        date,
        strategy: Number(value),
      }));
  }, [result]);

  // Format full ISO date as "MMM 'YY" for compact x-axis ticks.
  const formatTick = (iso: string) => {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
    const m = d.toLocaleString("en", { month: "short" });
    const y = String(d.getFullYear()).slice(2);
    return `${m} '${y}`;
  };

  // Format full ISO date as "DD MMM YYYY" for tooltip header.
  const formatTooltipLabel = (label: React.ReactNode): string => {
    const iso = String(label ?? "");
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
    const day = String(d.getDate()).padStart(2, "0");
    const m = d.toLocaleString("en", { month: "short" });
    return `${day} ${m} ${d.getFullYear()}`;
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="glass-card p-6">
        <h3 className="font-bold text-lg mb-4">Backtest — {ticker}</h3>
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1.5">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              max={new Date().toISOString().slice(0, 10)}
              className="input-dark w-44"
            />
          </div>
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1.5">
              Capital
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm text-[var(--text-muted)] pointer-events-none select-none">
                {currency}
              </span>
              <input
                type="text"
                inputMode="numeric"
                value={
                  capitalInput === ""
                    ? ""
                    : Number(capitalInput).toLocaleString()
                }
                onChange={(e) => handleCapitalChange(e.target.value)}
                onBlur={() => {
                  // If left empty, restore the parent's current capital.
                  if (capitalInput === "") setCapitalInput(String(capital));
                }}
                placeholder="100000"
                className="input-dark w-44 pl-7"
                aria-label="Initial capital"
              />
            </div>
          </div>
          <button
            onClick={handleRun}
            disabled={loading || capital <= 0}
            className="btn-primary h-[42px] flex items-center gap-2"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Run Backtest
          </button>
        </div>

        {/* Capital quick-presets */}
        <div className="flex flex-wrap gap-2 mt-4">
          {capitalPresets.map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setCapital(v)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                capital === v
                  ? "border-[var(--accent-blue)] bg-[var(--accent-blue)]/10 text-white"
                  : "border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent-blue)]"
              }`}
            >
              {currency}
              {v.toLocaleString()}
            </button>
          ))}
        </div>

        {error && (
          <p className="mt-3 text-sm text-[var(--accent-red)] flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            {error}
          </p>
        )}

        {/* Inputs differ from the last run — clear hint */}
        {inputsChanged && (
          <p className="mt-3 text-sm text-[var(--accent-yellow)] flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            Inputs changed — click <b>Run Backtest</b> to refresh results.
          </p>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Signal source badge */}
          {result.signal_source && (
            (() => {
              const isML = result.signal_source === "ml_ensemble";
              const Icon = isML ? Brain : Sigma;
              const color = isML ? "var(--accent-blue)" : "var(--accent-yellow)";
              const label = isML ? "Powered by ML Ensemble" : "Statistical Fallback";
              return (
                <div
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold w-fit fade-in"
                  style={{
                    background: `${color}15`,
                    color,
                    border: `1px solid ${color}30`,
                  }}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </div>
              );
            })()
          )}

          {/* Metrics comparison */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 fade-in">
            {[
              { label: "Total Return", val: `${(result.metrics.total_return * 100).toFixed(1)}%`, bh: `${(result.buy_hold_metrics.total_return * 100).toFixed(1)}%` },
              { label: "Sharpe Ratio", val: result.metrics.sharpe_ratio.toFixed(2), bh: result.buy_hold_metrics.sharpe_ratio.toFixed(2) },
              { label: "Max Drawdown", val: `${(result.metrics.max_drawdown * 100).toFixed(1)}%`, bh: `${(result.buy_hold_metrics.max_drawdown * 100).toFixed(1)}%` },
              { label: "Win Rate", val: `${(result.metrics.win_rate * 100).toFixed(0)}%`, bh: "—" },
              { label: "CAGR", val: `${(result.metrics.cagr * 100).toFixed(1)}%`, bh: `${(result.buy_hold_metrics.cagr * 100).toFixed(1)}%` },
              { label: "Sortino Ratio", val: result.metrics.sortino_ratio.toFixed(2), bh: result.buy_hold_metrics.sortino_ratio.toFixed(2) },
              { label: "Total Trades", val: result.metrics.total_trades.toString(), bh: "1" },
              { label: "Profit Factor", val: result.metrics.profit_factor.toFixed(2), bh: "—" },
            ].map((m) => (
              <div key={m.label} className="metric-card">
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">{m.label}</p>
                <p className="text-lg font-bold">{m.val}</p>
                <p className="text-xs text-[var(--text-secondary)] mt-1">
                  B&H: {m.bh}
                </p>
              </div>
            ))}
          </div>

          {/* Equity chart */}
          <div className="glass-card p-6 fade-in">
            <h4 className="font-bold mb-4">Equity Curve</h4>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2640" />
                  <XAxis
                    dataKey="date"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#5a6580", fontSize: 11 }}
                    interval={Math.max(1, Math.floor(chartData.length / 8))}
                    tickFormatter={formatTick}
                    minTickGap={20}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#5a6580", fontSize: 11 }}
                    tickFormatter={(v) => `${currency}${(v / 1000).toFixed(0)}K`}
                    width={65}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1f2e",
                      border: "1px solid #2a3050",
                      borderRadius: "8px",
                      color: "#f0f2f8",
                      fontSize: "13px",
                    }}
                    labelFormatter={formatTooltipLabel}
                    formatter={(value) => [
                      `${currency}${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                      "Portfolio",
                    ]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="strategy"
                    name="ML Strategy"
                    stroke="#4f6bff"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Trades table */}
          {result.trades.length > 0 && (
            <div className="glass-card p-6 fade-in overflow-x-auto">
              <h4 className="font-bold mb-4">
                Recent Trades ({result.total_trades} total)
              </h4>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[var(--text-muted)] text-xs uppercase">
                    <th className="pb-3 pr-4">Entry</th>
                    <th className="pb-3 pr-4">Exit</th>
                    <th className="pb-3 pr-4">Entry Price</th>
                    <th className="pb-3 pr-4">Exit Price</th>
                    <th className="pb-3 pr-4">P&L</th>
                    <th className="pb-3 pr-4">Return</th>
                    <th className="pb-3">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.slice(0, 15).map((t, i) => (
                    <tr key={i} className="border-t border-[var(--border)]">
                      <td className="py-2.5 pr-4 font-mono text-xs">{String(t.entry_date).slice(0, 10)}</td>
                      <td className="py-2.5 pr-4 font-mono text-xs">{String(t.exit_date).slice(0, 10)}</td>
                      <td className="py-2.5 pr-4">{currency}{Number(t.entry_price).toFixed(2)}</td>
                      <td className="py-2.5 pr-4">{currency}{Number(t.exit_price).toFixed(2)}</td>
                      <td className={`py-2.5 pr-4 font-bold ${Number(t.pnl) >= 0 ? "text-[var(--accent-green)]" : "text-[var(--accent-red)]"}`}>
                        {Number(t.pnl) >= 0 ? "+" : ""}{currency}{Math.abs(Number(t.pnl)).toFixed(0)}
                      </td>
                      <td className={`py-2.5 pr-4 ${Number(t.return_pct) >= 0 ? "text-[var(--accent-green)]" : "text-[var(--accent-red)]"}`}>
                        {(Number(t.return_pct) * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-xs text-[var(--text-muted)]">{String(t.exit_reason)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
