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

export async function getMe(): Promise<AuthMe> {
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
}

export async function logout(): Promise<void> {
  const res = await fetch("/api/auth/logout", {
    method: "POST",
    credentials: "same-origin",
  });
  if (!res.ok) throw await apiErrorFrom(res);
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
