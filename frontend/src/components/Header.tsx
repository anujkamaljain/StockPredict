"use client";

import { Activity } from "lucide-react";

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--bg-primary)]/80 backdrop-blur-xl">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[var(--accent-blue)] to-[var(--accent-cyan)] flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold tracking-tight">StockML</h2>
            <p className="text-[10px] text-[var(--text-muted)] -mt-0.5 tracking-wider uppercase">
              Decision Support
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <span className="w-2 h-2 rounded-full bg-[var(--accent-green)] animate-pulse" />
            System Online
          </div>
          <a
            href="/docs"
            target="_blank"
            className="text-xs text-[var(--text-secondary)] hover:text-white transition-colors px-3 py-1.5 rounded-lg hover:bg-[var(--bg-card)]"
          >
            API Docs
          </a>
        </div>
      </div>
    </header>
  );
}
