/**
 * Accounts client.
 *
 * The browser never sees a login credential: the session lives in an HttpOnly
 * cookie the backend sets on the OAuth callback (or on the admin password
 * login). This module only asks who the cookie belongs to, starts a login, and
 * ends one.
 */

import {
  AGENT_TOKEN_HEADER,
  apiErrorFrom,
  forgetLegacyAgentToken,
  getAgentToken,
  onPrincipalMayChange,
} from "./agent";

export interface AuthProvider {
  id: string;
  label: string;
}

export interface AuthQuota {
  runs: number;
  runs_limit: number;
  tokens: number;
  tokens_limit: number;
}

export interface AuthMe {
  mode: "multi_user" | "single_user";
  allow_anonymous: boolean;
  providers: AuthProvider[];
  /** The configured administrator can log in with a password. */
  admin_login: boolean;
  authenticated: boolean;
  principal_id: string | null;
  kind: "anonymous" | "user" | null;
  user: {
    display_name: string;
    email: string;
    avatar_url: string;
    role: string;
  } | null;
  quota: AuthQuota | null;
}

const ME_CACHE_TTL_MS = 5000;
let cachedMe: { value: AuthMe; at: number } | null = null;
let inflightMe: Promise<AuthMe | null> | null = null;

/** Drop the cached `/api/auth/me` snapshot after login, logout, or principal creation. */
export function invalidateMe(): void {
  cachedMe = null;
  inflightMe = null;
}

onPrincipalMayChange(invalidateMe);

async function fetchMeOnce(): Promise<AuthMe> {
  const headers = new Headers();
  // A pre-accounts browser still holds its credential in storage; this call
  // has the server move it into the cookie, after which it is forgotten.
  const legacy = getAgentToken();
  if (legacy) headers.set(AGENT_TOKEN_HEADER, legacy);
  const res = await fetch("/api/auth/me", { headers, credentials: "same-origin" });
  if (!res.ok) throw await apiErrorFrom(res);
  const me = (await res.json()) as AuthMe;
  if (legacy && me.principal_id) forgetLegacyAgentToken();
  return me;
}

/**
 * Fetch caller identity, coalescing concurrent mounts (`ChatShell`, `AuthMenu`,
 * `Home`) into a single request and retrying once across brief dev-server reloads.
 * Resolves to `null` when the backend is unreachable rather than throwing.
 */
export async function getMe(opts: { force?: boolean } = {}): Promise<AuthMe | null> {
  if (opts.force) invalidateMe();
  if (cachedMe && Date.now() - cachedMe.at < ME_CACHE_TTL_MS) {
    return cachedMe.value;
  }
  if (inflightMe) return inflightMe;

  let task: Promise<AuthMe | null> | null = null;
  task = (async (): Promise<AuthMe | null> => {
    try {
      const me = await fetchMeOnce();
      cachedMe = { value: me, at: Date.now() };
      return me;
    } catch {
      // One brief backoff retry covers uvicorn `--reload` restarts without
      // leaving components empty; if the server is genuinely down, return null.
      await new Promise((r) => setTimeout(r, 350));
      try {
        const me = await fetchMeOnce();
        cachedMe = { value: me, at: Date.now() };
        return me;
      } catch {
        return null;
      }
    } finally {
      if (inflightMe === task) inflightMe = null;
    }
  })();

  inflightMe = task;
  return task;
}

/** Send the browser to the provider; it comes back to the current page. */
export async function startLogin(provider: string): Promise<void> {
  const next = window.location.pathname + window.location.search;
  const res = await fetch(`/api/auth/login/${provider}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({ next }),
  });
  if (!res.ok) throw await apiErrorFrom(res);
  const { authorize_url } = (await res.json()) as { authorize_url: string };
  window.location.href = authorize_url;
}

/** Log in as the configured administrator; the cookie is set on success. */
export async function adminLogin(username: string, password: string): Promise<void> {
  const res = await fetch("/api/auth/admin/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw await apiErrorFrom(res);
  invalidateMe();
}

/** How much deleting the caller's data would remove. */
export async function getMyData(): Promise<{ sessions: number; runs: number; projects: number }> {
  const res = await fetch("/api/auth/me/data", { credentials: "same-origin" });
  if (!res.ok) throw await apiErrorFrom(res);
  return res.json();
}

/** Delete the caller's conversations, reports, projects, memories and account. */
export async function deleteMyData(): Promise<void> {
  const res = await fetch("/api/auth/me", { method: "DELETE", credentials: "same-origin" });
  if (!res.ok) throw await apiErrorFrom(res);
  invalidateMe();
}

export async function logout(): Promise<void> {
  const res = await fetch("/api/auth/logout", {
    method: "POST",
    credentials: "same-origin",
  });
  if (!res.ok) throw await apiErrorFrom(res);
  invalidateMe();
}

/** Read and strip `?auth_error=` the callback leaves on a failed login. */
export function takeAuthError(): string {
  if (typeof window === "undefined") return "";
  const url = new URL(window.location.href);
  const code = url.searchParams.get("auth_error") || "";
  if (code) {
    url.searchParams.delete("auth_error");
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
  }
  return code;
}
