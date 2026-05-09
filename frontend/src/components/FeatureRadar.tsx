"use client";

import type { SignalResponse } from "@/lib/api";

interface Props {
  features: Record<string, number>;
}

export function FeatureRadar({ features }: Props) {
  // Key features to display
  const displayFeatures: { key: string; label: string; range: [number, number]; invert?: boolean }[] = [
    { key: "rsi_14", label: "RSI", range: [0, 100] },
    { key: "adx", label: "ADX (Trend)", range: [0, 60] },
    { key: "volatility_20", label: "Volatility", range: [0, 0.6], invert: true },
    { key: "volume_ratio_20", label: "Volume", range: [0, 3] },
    { key: "price_zscore_60", label: "Z-Score", range: [-3, 3] },
    { key: "macd", label: "MACD", range: [-5, 5] },
  ];

  return (
    <div className="glass-card p-6 fade-in">
      <h3 className="font-bold text-sm mb-4 text-[var(--text-secondary)] uppercase tracking-wider">
        Key Indicators
      </h3>

      <div className="space-y-3">
        {displayFeatures.map((feat) => {
          const rawVal = features[feat.key];
          if (rawVal === undefined) return null;

          const [min, max] = feat.range;
          let normalized = (rawVal - min) / (max - min);
          normalized = Math.max(0, Math.min(1, normalized));
          if (feat.invert) normalized = 1 - normalized;

          const barColor =
            normalized > 0.6
              ? "var(--accent-green)"
              : normalized < 0.4
              ? "var(--accent-red)"
              : "var(--accent-yellow)";

          return (
            <div key={feat.key}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-[var(--text-secondary)]">{feat.label}</span>
                <span className="font-mono font-bold">{rawVal.toFixed(2)}</span>
              </div>
              <div className="h-2 bg-[var(--bg-secondary)] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${normalized * 100}%`,
                    background: barColor,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
