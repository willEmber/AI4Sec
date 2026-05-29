const OWNER_TOKEN_KEY = "scholar_owner_token";

/**
 * Stable per-browser token used to scope which runs this client can see.
 * Generated once and persisted in localStorage. Returns "" during SSR or when
 * storage is unavailable (e.g. privacy mode), in which case the client simply
 * falls back to seeing only ownerless legacy runs.
 */
export function getOwnerToken(): string {
  if (typeof window === "undefined") return "";
  try {
    let token = window.localStorage.getItem(OWNER_TOKEN_KEY);
    if (!token) {
      token =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
      window.localStorage.setItem(OWNER_TOKEN_KEY, token);
    }
    return token;
  } catch {
    return "";
  }
}
