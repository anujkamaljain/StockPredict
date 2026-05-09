"use client";

import { useState } from "react";
import { Header } from "@/components/Header";
import { StockSearch } from "@/components/StockSearch";
import { SignalPanel } from "@/components/SignalPanel";
import { PriceChart } from "@/components/PriceChart";
import { BacktestPanel } from "@/components/BacktestPanel";
import { PortfolioSettings } from "@/components/PortfolioSettings";
import { FeatureRadar } from "@/components/FeatureRadar";
import { MetricsGrid } from "@/components/MetricsGrid";
import type { SignalResponse, BacktestResult, StockData } from "@/lib/api";

export default function Home() {
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [signal, setSignal] = useState<SignalResponse | null>(null);
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [riskTolerance, setRiskTolerance] = useState("medium");
  const [capital, setCapital] = useState(100000);
  const [activeTab, setActiveTab] = useState<"signals" | "backtest" | "settings">("signals");

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 max-w-[1440px] mx-auto w-full px-4 sm:px-6 lg:px-8 pb-12">
        {/* Hero / Search */}
        <section className="pt-8 pb-6 fade-in">
          <div className="text-center mb-8">
            <h1 className="text-4xl sm:text-5xl font-bold mb-3">
              <span className="gradient-text">AI-Powered</span> Market Intelligence
            </h1>
            <p className="text-[var(--text-secondary)] text-lg max-w-2xl mx-auto">
              Ensemble ML models analyzing 100+ features. Probabilistic signals, not predictions.
              Decision support backed by statistical rigor.
            </p>
          </div>

          <StockSearch
            onSelect={(ticker, data, sig) => {
              setSelectedTicker(ticker);
              setStockData(data);
              setSignal(sig);
              setBacktestResult(null);
            }}
            riskTolerance={riskTolerance}
          />
        </section>

        {/* Tab Navigation */}
        {selectedTicker && (
          <div className="flex gap-1 mb-6 bg-[var(--bg-secondary)] p-1 rounded-xl w-fit mx-auto fade-in">
            {(["signals", "backtest", "settings"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-2.5 rounded-lg text-sm font-semibold capitalize transition-all ${
                  activeTab === tab
                    ? "bg-[var(--accent-blue)] text-white shadow-lg shadow-blue-500/20"
                    : "text-[var(--text-secondary)] hover:text-white hover:bg-[var(--bg-card)]"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        )}

        {/* Content */}
        {selectedTicker && activeTab === "signals" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 slide-up">
            {/* Signal Panel - Left */}
            <div className="lg:col-span-1 space-y-6">
              {signal && <SignalPanel signal={signal} />}
              {signal && <FeatureRadar features={signal.features} />}
            </div>

            {/* Chart + Metrics - Right */}
            <div className="lg:col-span-2 space-y-6">
              {stockData && <PriceChart data={stockData} />}
              {signal && <MetricsGrid signal={signal} />}
            </div>
          </div>
        )}

        {selectedTicker && activeTab === "backtest" && (
          <div className="slide-up">
            <BacktestPanel
              ticker={selectedTicker}
              result={backtestResult}
              onRunBacktest={setBacktestResult}
              capital={capital}
              riskTolerance={riskTolerance}
            />
          </div>
        )}

        {activeTab === "settings" && (
          <div className="max-w-2xl mx-auto slide-up">
            <PortfolioSettings
              capital={capital}
              setCapital={setCapital}
              riskTolerance={riskTolerance}
              setRiskTolerance={setRiskTolerance}
            />
          </div>
        )}

        {/* Disclaimer */}
        <footer className="mt-16 text-center text-xs text-[var(--text-muted)] max-w-3xl mx-auto leading-relaxed">
          <p className="mb-2 font-semibold text-[var(--text-secondary)]">⚠️ Important Disclaimer</p>
          <p>
            Stock markets are stochastic and partially efficient. No model guarantees profit.
            This system provides probabilistic decision support — not financial advice.
            All signals originate from trained ML models focusing on risk-adjusted returns.
            Past performance does not guarantee future results. Always consult a qualified
            financial advisor before making investment decisions.
          </p>
        </footer>
      </main>
    </div>
  );
}
