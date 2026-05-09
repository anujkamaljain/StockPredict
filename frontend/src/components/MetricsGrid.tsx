"use client";

import type { SignalResponse } from "@/lib/api";
import { TrendingUp, Activity, AlertTriangle, Gauge } from "lucide-react";

interface Props {
  signal: SignalResponse;
}

export function MetricsGrid({ signal }: Props) {
  const f = signal.features;

  const metrics = [
    {
      label: "RSI (14)",
      value: f.rsi_14?.toFixed(1) ?? "—",
      subtext: f.rsi_14 ? (f.rsi_14 < 30 ? "Oversold" : f.rsi_14 > 70 ? "Overbought" : "Neutral") : "",
      icon: Gauge,
      color: f.rsi_14 ? (f.rsi_14 < 30 ? "green" : f.rsi_14 > 70 ? "red" : "blue") : "blue",
    },
    {
      label: "Trend (ADX)",
      value: f.adx?.toFixed(1) ?? "—",
      subtext: f.adx ? (f.adx > 25 ? "Strong Trend" : "Weak/Ranging") : "",
      icon: TrendingUp,
      color: f.adx && f.adx > 25 ? "green" : "yellow",
    },
    {
      label: "Volatility (20d)",
      value: f.volatility_20 != null ? `${(f.volatility_20 * 100).toFixed(1)}%` : "—",
      subtext: f.volatility_20 != null ? (f.volatility_20 > 0.3 ? "High" : f.volatility_20 < 0.15 ? "Low" : "Normal") : "",
      icon: Activity,
      color: f.volatility_20 != null ? (f.volatility_20 > 0.3 ? "red" : "green") : "blue",
    },
    {
      label: "Drawdown",
      value: f.drawdown != null ? `${(f.drawdown * 100).toFixed(1)}%` : "—",
      subtext: f.drawdown != null ? (f.drawdown < -0.1 ? "Deep" : f.drawdown < -0.05 ? "Moderate" : f.drawdown === 0 ? "At Peak" : "Shallow") : "",
      icon: AlertTriangle,
      color: f.drawdown != null ? (f.drawdown < -0.1 ? "red" : f.drawdown < -0.05 ? "yellow" : "green") : "blue",
    },
  ];

  const colorMap: Record<string, string> = {
    green: "var(--accent-green)",
    red: "var(--accent-red)",
    yellow: "var(--accent-yellow)",
    blue: "var(--accent-blue)",
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 fade-in">
      {metrics.map((m) => {
        const Icon = m.icon;
        return (
          <div key={m.label} className="metric-card">
            <div className="flex items-center gap-2 mb-2">
              <Icon className="w-4 h-4" style={{ color: colorMap[m.color] }} />
              <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
                {m.label}
              </span>
            </div>
            <div className="text-xl font-bold">{m.value}</div>
            <div className="text-xs mt-1" style={{ color: colorMap[m.color] }}>
              {m.subtext}
            </div>
          </div>
        );
      })}
    </div>
  );
}
