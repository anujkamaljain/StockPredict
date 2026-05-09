"use client";

import { useState, useMemo } from "react";
import { Loader2, Play } from "lucide-react";
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
  onRunBacktest: (result: BacktestResult) => void;
  capital: number;
  riskTolerance: string;
}

export function BacktestPanel({ ticker, result, onRunBacktest, capital, riskTolerance }: Props) {
  const currency = getCurrencySymbol(ticker);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [startDate, setStartDate] = useState("2018-01-01");

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
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  };

  const chartData = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.equity_curve)
      .filter((_,i) => i % 5 === 0) // Sample every 5 days for performance
      .map(([date, value]) => ({
        date: date.slice(5, 10),
        strategy: Number(value),
      }));
  }, [result]);

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
              className="input-dark w-44"
            />
          </div>
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1.5">Capital</label>
            <input
              type="text"
              value={`${currency}${capital.toLocaleString()}`}
              readOnly
              className="input-dark w-36 opacity-60"
            />
          </div>
          <button
            onClick={handleRun}
            disabled={loading}
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
        {error && <p className="mt-3 text-sm text-[var(--accent-red)]">{error}</p>}
      </div>

      {/* Results */}
      {result && (
        <>
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
                    interval={Math.floor(chartData.length / 8)}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#5a6580", fontSize: 11 }}
                    tickFormatter={(v) => `${currency}${(v / 1000).toFixed(0)}K`}
                    width={65}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1f2e",
                      border: "1px solid #2a3050",
                      borderRadius: "8px",
                      color: "#f0f2f8",
                      fontSize: "13px",
                    }}
                    formatter={(value) => [`${currency}${Number(value).toFixed(0)}`, "Portfolio"]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="strategy"
                    name="ML Strategy"
                    stroke="#4f6bff"
                    strokeWidth={2}
                    dot={false}
                    animationDuration={1500}
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
