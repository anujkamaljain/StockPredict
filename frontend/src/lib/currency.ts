/**
 * Get the correct currency symbol based on ticker suffix.
 *
 * .NS / .BO → ₹ (Indian Rupee)
 * .L        → £ (British Pound)
 * .TO / .V  → C$ (Canadian Dollar)
 * .AX       → A$ (Australian Dollar)
 * .HK       → HK$ (Hong Kong Dollar)
 * .T        → ¥ (Japanese Yen)
 * default   → $ (US Dollar)
 */
export function getCurrencySymbol(ticker: string): string {
  const t = ticker.toUpperCase();
  if (t.endsWith(".NS") || t.endsWith(".BO")) return "₹";
  if (t.endsWith(".L")) return "£";
  if (t.endsWith(".TO") || t.endsWith(".V")) return "C$";
  if (t.endsWith(".AX")) return "A$";
  if (t.endsWith(".HK")) return "HK$";
  if (t.endsWith(".T")) return "¥";
  if (t.endsWith(".DE") || t.endsWith(".PA") || t.endsWith(".AS")) return "€";
  return "$";
}
