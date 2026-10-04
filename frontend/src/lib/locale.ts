// Locale constants shared by the server layout and the client provider.
// Kept out of i18n.tsx, which is a client module: a server component cannot
// call functions exported from one.

export type Locale = "en" | "zh";

export const DEFAULT_LOCALE: Locale = "zh";

/** Cookie holding the reader's language, so the server renders it directly. */
export const LOCALE_COOKIE = "scholar-locale";

export function parseLocale(value: string | undefined | null): Locale | null {
  return value === "zh" || value === "en" ? value : null;
}
