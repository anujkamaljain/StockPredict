"use client";

import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { StockData } from "@/lib/api";
import { getCurrencySymbol } from "@/lib/currency";

interface Props {
  data: StockData;
}

export function PriceChart({ data }: Props) {
  const currency = getCurrencySymbol(data.ticker);
  const chartData = useMemo(() => {
    const last250 = Math.max(0, data.data.dates.length - 250);
    return data.data.dates.slice(last250).map((date, i) => {
      // Format: "Jan '25" for axis, full date for tooltip
      const d = new Date(date);
      const month = d.toLocaleString("en", { month: "short" });
      const year = String(d.getFullYear()).slice(2); // "25", "26"
      return {
        date: `${month} '${year}`,
        fullDate: date,
        close: data.data.close[last250 + i],
        volume: data.data.volume[last250 + i],
      };
    });
  }, [data]);

  const minPrice = Math.min(...chartData.map((d) => d.close)) * 0.98;
  const maxPrice = Math.max(...chartData.map((d) => d.close)) * 1.02;
  const isUp = chartData.length > 1 && chartData[chartData.length - 1].close >= chartData[0].close;

  return (
    <div className="glass-card p-6 fade-in">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold text-lg">{data.ticker} — Price History</h3>
        <span className="text-xs text-[var(--text-muted)]">
          Last {chartData.length} trading days
        </span>
      </div>

      <div className="h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <defs>
              <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isUp ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                <stop offset="100%" stopColor={isUp ? "#10b981" : "#ef4444"} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2640" />
            <XAxis
              dataKey="fullDate"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#5a6580", fontSize: 11 }}
              interval={Math.floor(chartData.length / 6)}
              tickFormatter={(d) => {
                const dt = new Date(d);
                const m = dt.toLocaleString("en", { month: "short" });
                const y = String(dt.getFullYear()).slice(2);
                return `${m} '${y}`;
              }}
            />
            <YAxis
              domain={[minPrice, maxPrice]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#5a6580", fontSize: 11 }}
              tickFormatter={(v) => `${currency}${v.toFixed(0)}`}
              width={60}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1f2e",
                border: "1px solid #2a3050",
                borderRadius: "8px",
                color: "#f0f2f8",
                fontSize: "13px",
              }}
              formatter={(value) => [`${currency}${Number(value).toFixed(2)}`, "Close"]}
            />
            <Area
              type="monotone"
              dataKey="close"
              stroke={isUp ? "#10b981" : "#ef4444"}
              strokeWidth={2}
              fill="url(#priceGrad)"
              animationDuration={1500}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
