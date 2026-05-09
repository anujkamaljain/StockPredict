"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Search, Loader2, TrendingUp, X } from "lucide-react";
import { api, type SignalResponse, type StockData } from "@/lib/api";

interface SearchResult {
  ticker: string;
  name: string;
  exchange: string;
  type: string;
}

interface Props {
  onSelect: (ticker: string, data: StockData, signal: SignalResponse) => void;
  riskTolerance: string;
}

export function StockSearch({ onSelect, riskTolerance }: Props) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [error, setError] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);

  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced search for suggestions
  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.length < 1) {
      setSuggestions([]);
      setShowDropdown(false);
      return;
    }

    setSearchLoading(true);
    try {
      const results = await api.searchStocks(q);
      const mapped: SearchResult[] = (results as SearchResult[]).map((r) => ({
        ticker: r.ticker,
        name: r.name || "",
        exchange: r.exchange || "",
        type: r.type || "EQUITY",
      }));
      setSuggestions(mapped);
      setShowDropdown(mapped.length > 0);
      setHighlightIndex(-1);
    } catch {
      setSuggestions([]);
      setShowDropdown(false);
    } finally {
      setSearchLoading(false);
    }
  }, []);

  const handleInputChange = (value: string) => {
    const upper = value.toUpperCase();
    setQuery(upper);
    setError("");

    // Debounce search
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchSuggestions(upper);
    }, 300);
  };

  // Analyze a selected stock
  const analyzeStock = useCallback(
    async (ticker: string) => {
      setShowDropdown(false);
      setQuery(ticker);
      setLoading(true);
      setError("");

      try {
        const [data, signal] = await Promise.all([
          api.fetchStock(ticker),
          api.getSignal(ticker, riskTolerance),
        ]);
        onSelect(ticker, data, signal);
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Failed to fetch data";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [riskTolerance, onSelect]
  );

  const handleSearch = useCallback(() => {
    const ticker = query.trim().toUpperCase();
    if (!ticker) return;
    analyzeStock(ticker);
  }, [query, analyzeStock]);

  // Keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showDropdown || suggestions.length === 0) {
      if (e.key === "Enter") handleSearch();
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIndex((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIndex((prev) =>
        prev > 0 ? prev - 1 : suggestions.length - 1
      );
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (highlightIndex >= 0 && highlightIndex < suggestions.length) {
        analyzeStock(suggestions[highlightIndex].ticker);
      } else {
        handleSearch();
      }
    } else if (e.key === "Escape") {
      setShowDropdown(false);
    }
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  // Popular tickers for quick access
  const popular = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "JPM"];

  const getTypeColor = (type: string) => {
    switch (type) {
      case "ETF":
        return "var(--accent-cyan)";
      case "INDEX":
        return "var(--accent-yellow)";
      case "MUTUALFUND":
        return "var(--accent-purple)";
      default:
        return "var(--accent-blue)";
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="relative flex gap-2">
        {/* Search Input with Icon */}
        <div className="relative flex-1">
          <Search
            className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-[var(--text-muted)] pointer-events-none"
            style={{ zIndex: 1 }}
          />

          <input
            ref={inputRef}
            id="stock-search-input"
            type="text"
            placeholder="Search stocks... (e.g., AAPL, HDFC, Reliance)"
            value={query}
            onChange={(e) => handleInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => {
              if (suggestions.length > 0) setShowDropdown(true);
            }}
            className="input-dark h-12 text-base"
            style={{ paddingLeft: "44px", paddingRight: query ? "36px" : "16px" }}
            autoComplete="off"
            spellCheck={false}
          />

          {/* Clear button */}
          {query && (
            <button
              onClick={() => {
                setQuery("");
                setSuggestions([]);
                setShowDropdown(false);
                inputRef.current?.focus();
              }}
              className="absolute right-3 top-1/2 -translate-y-1/2 w-6 h-6 flex items-center justify-center rounded-full hover:bg-[var(--bg-card)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}

          {/* Autocomplete Dropdown */}
          {showDropdown && (
            <div
              ref={dropdownRef}
              className="absolute top-full left-0 right-0 mt-2 overflow-hidden z-50"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border)",
                borderRadius: "12px",
                boxShadow: "0 12px 40px rgba(0, 0, 0, 0.5)",
                maxHeight: "360px",
                overflowY: "auto",
              }}
            >
              {searchLoading ? (
                <div className="flex items-center gap-3 px-4 py-3 text-sm text-[var(--text-muted)]">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Searching...
                </div>
              ) : (
                suggestions.map((item, idx) => (
                  <button
                    key={item.ticker}
                    onClick={() => analyzeStock(item.ticker)}
                    onMouseEnter={() => setHighlightIndex(idx)}
                    className="w-full text-left px-4 py-3 flex items-center gap-3 transition-colors"
                    style={{
                      background:
                        highlightIndex === idx
                          ? "var(--bg-card-hover)"
                          : "transparent",
                      borderBottom:
                        idx < suggestions.length - 1
                          ? "1px solid rgba(42, 48, 80, 0.5)"
                          : "none",
                    }}
                  >
                    {/* Ticker badge */}
                    <div
                      className="flex-shrink-0 flex items-center justify-center rounded-lg font-bold text-xs"
                      style={{
                        width: "44px",
                        height: "36px",
                        background: `${getTypeColor(item.type)}15`,
                        color: getTypeColor(item.type),
                        border: `1px solid ${getTypeColor(item.type)}30`,
                      }}
                    >
                      {item.ticker.length > 5
                        ? item.ticker.slice(0, 4) + "…"
                        : item.ticker}
                    </div>

                    {/* Stock details */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-[var(--text-primary)] truncate">
                          {item.ticker}
                        </span>
                        <span
                          className="text-[10px] font-medium px-1.5 py-0.5 rounded"
                          style={{
                            background: `${getTypeColor(item.type)}15`,
                            color: getTypeColor(item.type),
                          }}
                        >
                          {item.type}
                        </span>
                      </div>
                      <div className="text-xs text-[var(--text-muted)] truncate mt-0.5">
                        {item.name}
                        {item.exchange && (
                          <span className="ml-1 opacity-60">
                            · {item.exchange}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Arrow indicator */}
                    <TrendingUp
                      className="flex-shrink-0 w-4 h-4"
                      style={{
                        color:
                          highlightIndex === idx
                            ? "var(--accent-blue)"
                            : "var(--text-muted)",
                        opacity: highlightIndex === idx ? 1 : 0.3,
                      }}
                    />
                  </button>
                ))
              )}

              {!searchLoading && suggestions.length === 0 && query.length >= 1 && (
                <div className="px-4 py-3 text-sm text-[var(--text-muted)] text-center">
                  No stocks found for &quot;{query}&quot;
                </div>
              )}
            </div>
          )}
        </div>

        {/* Analyze button */}
        <button
          id="search-button"
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="btn-primary h-12 px-8 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="hidden sm:inline">Analyzing...</span>
            </>
          ) : (
            <>
              <Search className="w-4 h-4" />
              <span>Analyze</span>
            </>
          )}
        </button>
      </div>

      {/* Quick picks */}
      <div className="flex flex-wrap gap-2 mt-4 justify-center">
        {popular.map((t) => (
          <button
            key={t}
            onClick={() => analyzeStock(t)}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--text-secondary)] bg-[var(--bg-card)] border border-[var(--border)] hover:border-[var(--accent-blue)] hover:text-white transition-all disabled:opacity-50"
          >
            {t}
          </button>
        ))}
      </div>

      {error && (
        <div className="mt-4 text-center text-sm text-[var(--accent-red)] bg-red-500/10 rounded-lg p-3 border border-red-500/20">
          {error}
        </div>
      )}
    </div>
  );
}
