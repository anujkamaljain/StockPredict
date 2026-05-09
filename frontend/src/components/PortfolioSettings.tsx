"use client";

import { Settings, DollarSign, Shield, Clock } from "lucide-react";

interface Props {
  capital: number;
  setCapital: (v: number) => void;
  riskTolerance: string;
  setRiskTolerance: (v: string) => void;
}

export function PortfolioSettings({ capital, setCapital, riskTolerance, setRiskTolerance }: Props) {
  const riskOptions = [
    { value: "low", label: "Conservative", desc: "Lower risk, fewer signals, wider thresholds", color: "var(--accent-green)" },
    { value: "medium", label: "Balanced", desc: "Moderate risk, standard thresholds", color: "var(--accent-yellow)" },
    { value: "high", label: "Aggressive", desc: "Higher risk, more signals, tighter thresholds", color: "var(--accent-red)" },
  ];

  const capitalPresets = [10000, 50000, 100000, 500000, 1000000];

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[var(--accent-blue)] to-[var(--accent-purple)] flex items-center justify-center">
            <Settings className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-bold text-lg">Portfolio Settings</h3>
            <p className="text-xs text-[var(--text-muted)]">Configure your investment parameters</p>
          </div>
        </div>

        {/* Capital */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-3">
            <DollarSign className="w-4 h-4 text-[var(--accent-green)]" />
            <label className="text-sm font-semibold">Initial Capital</label>
          </div>
          <input
            type="number"
            value={capital}
            onChange={(e) => setCapital(Number(e.target.value) || 0)}
            className="input-dark mb-3 text-lg font-bold"
          />
          <div className="flex flex-wrap gap-2">
            {capitalPresets.map((v) => (
              <button
                key={v}
                onClick={() => setCapital(v)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                  capital === v
                    ? "border-[var(--accent-blue)] bg-[var(--accent-blue)]/10 text-white"
                    : "border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent-blue)]"
                }`}
              >
                ${v.toLocaleString()}
              </button>
            ))}
          </div>
        </div>

        {/* Risk Tolerance */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-4 h-4 text-[var(--accent-blue)]" />
            <label className="text-sm font-semibold">Risk Tolerance</label>
          </div>
          <div className="space-y-2">
            {riskOptions.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setRiskTolerance(opt.value)}
                className={`w-full text-left p-4 rounded-xl border transition-all ${
                  riskTolerance === opt.value
                    ? "border-[var(--accent-blue)] bg-[var(--accent-blue)]/5"
                    : "border-[var(--border)] hover:border-[var(--accent-blue)]/50"
                }`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: opt.color }}
                  />
                  <div>
                    <p className="font-semibold text-sm">{opt.label}</p>
                    <p className="text-xs text-[var(--text-muted)]">{opt.desc}</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Investment Horizon */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-4 h-4 text-[var(--accent-purple)]" />
            <label className="text-sm font-semibold">Signal Frequency</label>
          </div>
          <div className="flex gap-2">
            <button className="flex-1 py-3 rounded-xl border border-[var(--accent-blue)] bg-[var(--accent-blue)]/5 text-sm font-semibold">
              Daily
            </button>
            <button className="flex-1 py-3 rounded-xl border border-[var(--border)] text-sm text-[var(--text-muted)] opacity-50 cursor-not-allowed">
              Weekly (Coming Soon)
            </button>
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="glass-card p-5 text-sm text-[var(--text-secondary)] leading-relaxed">
        <p className="font-semibold text-white mb-2">How Settings Affect Signals</p>
        <ul className="space-y-1.5 text-xs">
          <li>• <strong>Capital</strong> affects position sizing via Kelly Criterion</li>
          <li>• <strong>Risk tolerance</strong> adjusts buy/sell thresholds and confidence minimums</li>
          <li>• Conservative: Buy only when P(up) {">"} 65%, Sell when P(up) {"<"} 35%</li>
          <li>• Balanced: Buy when P(up) {">"} 58%, Sell when P(up) {"<"} 42%</li>
          <li>• Aggressive: Buy when P(up) {">"} 53%, Sell when P(up) {"<"} 47%</li>
        </ul>
      </div>
    </div>
  );
}
