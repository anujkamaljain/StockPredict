import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "StockML — AI-Powered Market Decision Support",
  description: "Production-grade ML system for stock market analysis with ensemble models, risk management, and backtesting. Powered by LSTM, Transformers, and XGBoost.",
  keywords: ["stock market", "machine learning", "trading signals", "backtesting", "portfolio management"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
