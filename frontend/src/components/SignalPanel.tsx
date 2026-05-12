"use client";

import { TrendingUp, TrendingDown, Minus, ShieldCheck, BarChart3, Brain, Sigma } from "lucide-react";
import type { SignalResponse } from "@/lib/api";
import { getCurrencySymbol } from "@/lib/currency";

interface Props {
  signal: SignalResponse;
}

export function SignalPanel({ signal }: Props) {
  const s = signal.signal;
  const currency = getCurrencySymbol(s.ticker);
  const actionColor =
    s.action === "BUY" ? "green" : s.action === "SELL" ? "red" : "yellow";
  const ActionIcon =
    s.action === "BUY" ? TrendingUp : s.action === "SELL" ? TrendingDown : Minus;

  const isML = signal.meta.signal_source === "ml_ensemble";
  const SourceIcon = isML ? Brain : Sigma;
  const sourceLabel = isML ? "ML Ensemble" : "Statistical Fallback";
  const sourceColor = isML ? "var(--accent-blue)" : "var(--accent-yellow)";

  const individual = signal.meta.individual_predictions ?? null;

  return (
    <div className="glass-card p-6 fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-2xl font-bold">{s.ticker}</h3>
          <p className="text-sm text-[var(--text-muted)]">
            {signal.meta.latest_date}
          </p>
        </div>
        <div className={`signal-badge signal-${s.action.toLowerCase()}`}>
          <ActionIcon className="w-4 h-4" />
          {s.action}
        </div>
      </div>

      {/* Signal source badge */}
      <div
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[11px] font-semibold mb-5 w-fit"
        style={{
          background: `${sourceColor}15`,
          color: sourceColor,
          border: `1px solid ${sourceColor}30`,
        }}
        title={
          isML
            ? "Signal produced by the trained 6-model ensemble (LSTM + Transformer + CNN + XGBoost + LightGBM + CatBoost)"
            : "No trained models found in data/models/. Signal produced by the statistical heuristic — train models for sharper results."
        }
      >
        <SourceIcon className="w-3.5 h-3.5" />
        {sourceLabel}
      </div>

      {/* Price */}
      <div className="mb-5">
        <div className="text-3xl font-bold">{currency}{signal.price.current.toFixed(2)}</div>
        <div className="flex gap-3 mt-1 text-sm">
          <span className={signal.price.change_1d >= 0 ? "text-[var(--accent-green)]" : "text-[var(--accent-red)]"}>
            {signal.price.change_1d >= 0 ? "+" : ""}{signal.price.change_1d.toFixed(2)}% 1D
          </span>
          <span className={signal.price.change_5d >= 0 ? "text-[var(--accent-green)]" : "text-[var(--accent-red)]"}>
            {signal.price.change_5d >= 0 ? "+" : ""}{signal.price.change_5d.toFixed(2)}% 5D
          </span>
          <span className={signal.price.change_20d >= 0 ? "text-[var(--accent-green)]" : "text-[var(--accent-red)]"}>
            {signal.price.change_20d >= 0 ? "+" : ""}{signal.price.change_20d.toFixed(2)}% 20D
          </span>
        </div>
      </div>

      {/* Confidence gauge */}
      <div className="mb-5">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-[var(--text-secondary)]">Confidence</span>
          <span className="font-bold">{(s.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="h-3 bg-[var(--bg-secondary)] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${s.confidence * 100}%`,
              background: `linear-gradient(90deg, var(--accent-${actionColor}), var(--accent-blue))`,
            }}
          />
        </div>
      </div>

      {/* Direction Probability */}
      <div className="mb-5">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-[var(--text-secondary)]">P(Up)</span>
          <span className="font-bold">{(s.direction_prob * 100).toFixed(1)}%</span>
        </div>
        <div className="h-3 bg-[var(--bg-secondary)] rounded-full overflow-hidden relative">
          <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--text-muted)] z-10" />
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${s.direction_prob * 100}%`,
              background:
                s.direction_prob > 0.5
                  ? "linear-gradient(90deg, #1a1f2e, var(--accent-green))"
                  : "linear-gradient(90deg, var(--accent-red), #1a1f2e)",
            }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-[var(--text-muted)] mt-1">
          <span>Bearish</span>
          <span>Neutral</span>
          <span>Bullish</span>
        </div>
      </div>

      {/* Risk Score */}
      <div className="grid grid-cols-2 gap-3">
        <div className="metric-card flex items-center gap-3">
          <ShieldCheck className="w-5 h-5 text-[var(--accent-blue)]" />
          <div>
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Risk</p>
            <p className="font-bold text-sm">
              {s.risk_score < 0.3 ? "Low" : s.risk_score < 0.6 ? "Medium" : "High"}
            </p>
          </div>
        </div>
        <div className="metric-card flex items-center gap-3">
          <BarChart3 className="w-5 h-5 text-[var(--accent-purple)]" />
          <div>
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Agreement</p>
            <p className="font-bold text-sm">
              {(s.model_agreement * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      {/* Per-model predictions (only shown when ML ensemble is active) */}
      {individual && Object.keys(individual).length > 0 && (
        <div className="mt-5 pt-5 border-t border-[var(--border)]">
          <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-3">
            Per-Model P(Up)
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(individual).map(([name, p]) => {
              const up = p > 0.5;
              return (
                <div
                  key={name}
                  className="flex justify-between items-center py-1 px-2 rounded"
                  style={{
                    background: up ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)",
                  }}
                >
                  <span className="capitalize text-[var(--text-secondary)]">{name}</span>
                  <span
                    className="font-mono font-bold"
                    style={{
                      color: up ? "var(--accent-green)" : "var(--accent-red)",
                    }}
                  >
                    {(p * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
