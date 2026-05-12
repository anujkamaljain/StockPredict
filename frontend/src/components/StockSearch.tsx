"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Search, Loader2, TrendingUp, X, Sparkles } from "lucide-react";
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
  const [hasSearched, setHasSearched] = useState(false);

  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Debounced search for suggestions
  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.length < 1) {
      setSuggestions([]);
      setShowDropdown(false);
      setHasSearched(false);
      return;
    }

    // Cancel any in-flight request so stale responses can't overwrite
    // newer ones (real race condition fix — the abort signal is now
    // actually threaded into the fetch call).
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setSearchLoading(true);
    setShowDropdown(true);
    try {
      const results = await api.searchStocks(q, controller.signal);
      if (controller.signal.aborted) return;
      const mapped: SearchResult[] = results.map((r) => ({
        ticker: r.ticker,
        name: r.name || "",
        exchange: r.exchange || "",
        type: r.type || "EQUITY",
      }));
      setSuggestions(mapped);
      setShowDropdown(true);
      setHasSearched(true);
      setHighlightIndex(-1);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setSuggestions([]);
      setHasSearched(true);
    } finally {
      if (!controller.signal.aborted) {
        setSearchLoading(false);
      }
    }
  }, []);

  const handleInputChange = (value: string) => {
    const upper = value.toUpperCase();
    setQuery(upper);
    setError("");

    if (!upper.trim()) {
      setSuggestions([]);
      setShowDropdown(false);
      setHasSearched(false);
      return;
    }

    // Show dropdown immediately with loading state
    setShowDropdown(true);
    setSearchLoading(true);

    // Debounce the actual API call
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchSuggestions(upper);
    }, 250);
  };

  // Analyze a selected stock
  const analyzeStock = useCallback(
    async (ticker: string) => {
      setShowDropdown(false);
      setQuery(ticker);
      setLoading(true);
      setError("");
      setSuggestions([]);

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
      setHighlightIndex((prev) => {
        const next = prev < suggestions.length - 1 ? prev + 1 : 0;
        scrollToItem(next);
        return next;
      });
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIndex((prev) => {
        const next = prev > 0 ? prev - 1 : suggestions.length - 1;
        scrollToItem(next);
        return next;
      });
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (highlightIndex >= 0 && highlightIndex < suggestions.length) {
        analyzeStock(suggestions[highlightIndex].ticker);
      } else {
        handleSearch();
      }
    } else if (e.key === "Escape") {
      setShowDropdown(false);
      inputRef.current?.blur();
    } else if (e.key === "Tab") {
      setShowDropdown(false);
    }
  };

  const scrollToItem = (index: number) => {
    const dropdown = dropdownRef.current;
    if (!dropdown) return;
    const items = dropdown.querySelectorAll("[data-search-item]");
    if (items[index]) {
      items[index].scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  // Cleanup pending debounce + in-flight search on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
    };
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

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "ETF":
        return "📊";
      case "INDEX":
        return "📈";
      case "MUTUALFUND":
        return "🏦";
      default:
        return "🏢";
    }
  };

  // Highlight matching text in name/ticker
  const highlightMatch = (text: string, q: string) => {
    if (!q || !text) return text;
    const idx = text.toUpperCase().indexOf(q.toUpperCase());
    if (idx === -1) return text;
    return (
      <>
        {text.slice(0, idx)}
        <span style={{ color: "var(--accent-blue)", fontWeight: 700 }}>
          {text.slice(idx, idx + q.length)}
        </span>
        {text.slice(idx + q.length)}
      </>
    );
  };

  const isDropdownVisible = showDropdown && query.trim().length > 0;

  return (
    <div className="max-w-2xl mx-auto" ref={containerRef}>
      <div className="relative flex gap-2">
        {/* Search Input with Icon */}
        <div className="relative flex-1" style={{ zIndex: 50 }}>
          <Search
            className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 pointer-events-none transition-colors duration-200"
            style={{
              color: isDropdownVisible
                ? "var(--accent-blue)"
                : "var(--text-muted)",
              zIndex: 1,
            }}
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
              if (query.trim().length > 0) {
                if (suggestions.length > 0) {
                  setShowDropdown(true);
                } else if (!hasSearched) {
                  fetchSuggestions(query);
                }
              }
            }}
            className="input-dark h-12 text-base"
            style={{
              paddingLeft: "44px",
              paddingRight: query ? "36px" : "16px",
              borderColor: isDropdownVisible
                ? "var(--accent-blue)"
                : undefined,
              boxShadow: isDropdownVisible
                ? "0 0 0 3px rgba(79, 107, 255, 0.15), 0 8px 32px rgba(0, 0, 0, 0.2)"
                : undefined,
              borderBottomLeftRadius: isDropdownVisible ? "0" : undefined,
              borderBottomRightRadius: isDropdownVisible ? "0" : undefined,
            }}
            autoComplete="off"
            spellCheck={false}
            role="combobox"
            aria-expanded={isDropdownVisible}
            aria-haspopup="listbox"
            aria-autocomplete="list"
            aria-controls="search-dropdown"
          />

          {/* Clear button / loading spinner in input */}
          {query && (
            <button
              onClick={() => {
                setQuery("");
                setSuggestions([]);
                setShowDropdown(false);
                setHasSearched(false);
                inputRef.current?.focus();
              }}
              className="absolute right-3 top-1/2 -translate-y-1/2 w-6 h-6 flex items-center justify-center rounded-full hover:bg-[var(--bg-card)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all duration-200"
              aria-label="Clear search"
            >
              {searchLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" style={{ color: "var(--accent-blue)" }} />
              ) : (
                <X className="w-4 h-4" />
              )}
            </button>
          )}

          {/* Autocomplete Dropdown */}
          {isDropdownVisible && (
            <div
              ref={dropdownRef}
              id="search-dropdown"
              role="listbox"
              className="absolute top-full left-0 right-0 overflow-hidden"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--accent-blue)",
                borderTop: "1px solid var(--border)",
                borderRadius: "0 0 12px 12px",
                boxShadow: "0 16px 48px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(79, 107, 255, 0.1)",
                maxHeight: "380px",
                overflowY: "auto",
                animation: "dropdownSlide 0.15s ease-out",
              }}
            >
              {searchLoading ? (
                /* Shimmer loading skeletons */
                <div style={{ padding: "4px 0" }}>
                  {[1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 px-4 py-3"
                      style={{
                        opacity: 1 - i * 0.15,
                      }}
                    >
                      <div
                        className="shimmer"
                        style={{
                          width: "44px",
                          height: "36px",
                          borderRadius: "8px",
                          flexShrink: 0,
                        }}
                      />
                      <div className="flex-1 space-y-2">
                        <div
                          className="shimmer"
                          style={{
                            height: "14px",
                            borderRadius: "4px",
                            width: `${60 + i * 10}%`,
                          }}
                        />
                        <div
                          className="shimmer"
                          style={{
                            height: "10px",
                            borderRadius: "4px",
                            width: `${40 + i * 8}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : suggestions.length > 0 ? (
                /* Actual results */
                <div style={{ padding: "4px 0" }}>
                  <div
                    className="px-4 py-2 flex items-center gap-2"
                    style={{
                      borderBottom: "1px solid rgba(42, 48, 80, 0.5)",
                    }}
                  >
                    <Sparkles className="w-3 h-3" style={{ color: "var(--accent-blue)" }} />
                    <span
                      style={{
                        fontSize: "10px",
                        fontWeight: 600,
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                        color: "var(--text-muted)",
                      }}
                    >
                      {suggestions.length} result{suggestions.length !== 1 ? "s" : ""} found
                    </span>
                  </div>
                  {suggestions.map((item, idx) => (
                    <button
                      key={item.ticker}
                      data-search-item
                      role="option"
                      aria-selected={highlightIndex === idx}
                      onClick={() => analyzeStock(item.ticker)}
                      onMouseEnter={() => setHighlightIndex(idx)}
                      className="w-full text-left px-4 py-3 flex items-center gap-3 transition-all duration-150"
                      style={{
                        background:
                          highlightIndex === idx
                            ? "linear-gradient(90deg, rgba(79, 107, 255, 0.12), rgba(79, 107, 255, 0.04))"
                            : "transparent",
                        borderLeft:
                          highlightIndex === idx
                            ? "3px solid var(--accent-blue)"
                            : "3px solid transparent",
                        borderBottom:
                          idx < suggestions.length - 1
                            ? "1px solid rgba(42, 48, 80, 0.3)"
                            : "none",
                        animation: `fadeSlideIn 0.2s ease-out ${idx * 0.03}s both`,
                      }}
                    >
                      {/* Ticker badge with icon */}
                      <div
                        className="flex-shrink-0 flex items-center justify-center rounded-lg font-bold text-xs"
                        style={{
                          width: "48px",
                          height: "38px",
                          background: `${getTypeColor(item.type)}12`,
                          color: getTypeColor(item.type),
                          border: `1px solid ${getTypeColor(item.type)}25`,
                          transition: "all 0.2s ease",
                          transform:
                            highlightIndex === idx
                              ? "scale(1.05)"
                              : "scale(1)",
                        }}
                      >
                        <span style={{ fontSize: "10px", marginRight: "2px" }}>
                          {getTypeIcon(item.type)}
                        </span>
                      </div>

                      {/* Stock details */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span
                            className="text-sm font-bold truncate"
                            style={{
                              color:
                                highlightIndex === idx
                                  ? "var(--text-primary)"
                                  : "var(--text-primary)",
                            }}
                          >
                            {highlightMatch(item.ticker, query)}
                          </span>
                          <span
                            className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
                            style={{
                              background: `${getTypeColor(item.type)}18`,
                              color: getTypeColor(item.type),
                              letterSpacing: "0.03em",
                            }}
                          >
                            {item.type}
                          </span>
                        </div>
                        <div
                          className="text-xs truncate mt-0.5"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {highlightMatch(item.name, query)}
                          {item.exchange && (
                            <span style={{ opacity: 0.5, marginLeft: "4px" }}>
                              · {item.exchange}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Action indicator */}
                      <div
                        className="flex-shrink-0 flex items-center justify-center rounded-full transition-all duration-200"
                        style={{
                          width: "28px",
                          height: "28px",
                          background:
                            highlightIndex === idx
                              ? "rgba(79, 107, 255, 0.15)"
                              : "transparent",
                        }}
                      >
                        <TrendingUp
                          className="w-3.5 h-3.5 transition-all duration-200"
                          style={{
                            color:
                              highlightIndex === idx
                                ? "var(--accent-blue)"
                                : "var(--text-muted)",
                            opacity: highlightIndex === idx ? 1 : 0.25,
                            transform:
                              highlightIndex === idx
                                ? "translateX(1px)"
                                : "none",
                          }}
                        />
                      </div>
                    </button>
                  ))}

                  {/* Keyboard hint */}
                  <div
                    className="px-4 py-2 flex items-center gap-3 justify-center"
                    style={{
                      borderTop: "1px solid rgba(42, 48, 80, 0.5)",
                      background: "rgba(10, 14, 23, 0.3)",
                    }}
                  >
                    <span className="flex items-center gap-1 text-[10px]" style={{ color: "var(--text-muted)" }}>
                      <kbd style={{
                        background: "var(--bg-card)",
                        border: "1px solid var(--border)",
                        borderRadius: "3px",
                        padding: "1px 4px",
                        fontSize: "9px",
                        fontFamily: "monospace",
                      }}>↑↓</kbd>
                      navigate
                    </span>
                    <span className="flex items-center gap-1 text-[10px]" style={{ color: "var(--text-muted)" }}>
                      <kbd style={{
                        background: "var(--bg-card)",
                        border: "1px solid var(--border)",
                        borderRadius: "3px",
                        padding: "1px 4px",
                        fontSize: "9px",
                        fontFamily: "monospace",
                      }}>↵</kbd>
                      select
                    </span>
                    <span className="flex items-center gap-1 text-[10px]" style={{ color: "var(--text-muted)" }}>
                      <kbd style={{
                        background: "var(--bg-card)",
                        border: "1px solid var(--border)",
                        borderRadius: "3px",
                        padding: "1px 4px",
                        fontSize: "9px",
                        fontFamily: "monospace",
                      }}>esc</kbd>
                      close
                    </span>
                  </div>
                </div>
              ) : hasSearched ? (
                /* No results found */
                <div className="px-4 py-6 text-center">
                  <div
                    style={{
                      fontSize: "28px",
                      marginBottom: "8px",
                      opacity: 0.6,
                    }}
                  >
                    🔍
                  </div>
                  <div
                    className="text-sm font-medium"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    No stocks found for &quot;{query}&quot;
                  </div>
                  <div
                    className="text-xs mt-1"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Try searching by ticker symbol or company name
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>

        {/* Analyze button */}
        <button
          id="search-button"
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="btn-primary h-12 px-8 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            whiteSpace: "nowrap",
            position: "relative",
            zIndex: 51,
          }}
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
            style={{
              cursor: loading ? "not-allowed" : "pointer",
            }}
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
