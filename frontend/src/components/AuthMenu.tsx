"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import { AgentApiError } from "@/lib/agent";
import {
  adminLogin,
  deleteMyData,
  getMe,
  getMyData,
  logout,
  startLogin,
  takeAuthError,
  type AuthMe,
} from "@/lib/auth";
import { useTranslation } from "@/lib/i18n";

const KNOWN_ERRORS = new Set([
  "denied",
  "account_disabled",
  "account_inactive",
  "invalid_credentials",
]);

/**
 * Login / account control for the navbar, or (`placement="above"`) for the
 * foot of the conversation sidebar, where the panel has to open upwards.
 *
 * Hidden entirely in single_user mode, where nobody logs in. Signed out it
 * lists the configured providers and, when an administrator is configured,
 * a username/password form; signed in it shows the avatar, today's usage and
 * logout.
 */
export function AuthMenu({ placement = "below" }: { placement?: "below" | "above" } = {}) {
  const { t } = useTranslation();
  const [me, setMe] = useState<AuthMe | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(() => {
    getMe().then(setMe).catch(() => setMe(null));
  }, []);

  useEffect(() => {
    refresh();
    const code = takeAuthError();
    if (code) {
      setError(
        KNOWN_ERRORS.has(code) ? t(`auth.error.${code}`) : t("auth.error.generic", { code }),
      );
      setOpen(true);
    }
    // `t` changes with the locale; the error is read once, on arrival.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh]);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  if (!me || me.mode === "single_user") return null;

  const login = async (provider: string) => {
    setBusy(true);
    setError("");
    try {
      await startLogin(provider);
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
      setBusy(false);
    }
  };

  const signInAsAdmin = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await adminLogin(username, password);
      // The page was rendered for the anonymous visitor; reload it as the account.
      window.location.reload();
    } catch (err) {
      const code = err instanceof AgentApiError ? err.code : "";
      setError(
        KNOWN_ERRORS.has(code)
          ? t(`auth.error.${code}`)
          : t("auth.error.generic", {
              code: code || (err instanceof AgentApiError ? String(err.status) : "network"),
            }),
      );
      setBusy(false);
    }
  };

  const eraseMyData = async () => {
    setBusy(true);
    setError("");
    try {
      const owned = await getMyData();
      const sure = window.confirm(
        t("auth.deleteDataConfirm", { sessions: owned.sessions, runs: owned.runs }),
      );
      if (!sure) {
        setBusy(false);
        return;
      }
      await deleteMyData();
      // Nothing the page shows exists any more.
      window.location.href = "/";
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
      setBusy(false);
    }
  };
  const eraseButton = (
    <button
      onClick={eraseMyData}
      disabled={busy}
      className="mt-2 w-full rounded-lg px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-destructive disabled:opacity-50"
    >
      {t("auth.deleteData")}
    </button>
  );

  const signOut = async () => {
    setBusy(true);
    try {
      await logout();
    } finally {
      // A new visitor now: whatever the page shows belongs to the old one.
      window.location.href = "/";
    }
  };

  const quota = me.quota;
  const quotaLine = quota
    ? quota.runs_limit
      ? t("auth.quotaToday", { runs: quota.runs, limit: quota.runs_limit })
      : t("auth.quotaTodayUnlimited", { runs: quota.runs })
    : "";
  const name = me.user?.display_name || me.user?.email || "";

  return (
    <div ref={rootRef} className="relative">
      {me.authenticated ? (
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 rounded-full border border-border py-1 pl-1 pr-3 text-sm text-muted-foreground transition-colors hover:border-foreground/25 hover:text-foreground"
          title={name}
        >
          {me.user?.avatar_url ? (
            // eslint-disable-next-line @next/next/no-img-element -- provider avatars come from arbitrary hosts
            <img
              src={me.user.avatar_url}
              alt=""
              referrerPolicy="no-referrer"
              className="h-6 w-6 rounded-full object-cover"
            />
          ) : (
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-medium">
              {(name || "?").slice(0, 1).toUpperCase()}
            </span>
          )}
          <span className="hidden max-w-[10rem] truncate sm:inline">{name}</span>
        </button>
      ) : (
        <button
          onClick={() => setOpen((v) => !v)}
          className="rounded-full border border-border px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:border-foreground/25 hover:text-foreground"
        >
          {t("auth.login")}
        </button>
      )}

      {open && (
        <div
          className={`absolute z-50 w-64 rounded-xl border border-border bg-background p-3 shadow-lg ${
            placement === "above" ? "bottom-full left-0 mb-2" : "right-0 top-full mt-2"
          }`}
        >
          {error && <p className="mb-2 text-xs text-red-600 dark:text-red-400">{error}</p>}

          {me.authenticated ? (
            <>
              <p className="truncate text-sm font-medium text-foreground">{name}</p>
              {me.user?.email && name !== me.user.email && (
                <p className="truncate text-xs text-muted-foreground">{me.user.email}</p>
              )}
              {quotaLine && <p className="mt-2 text-xs text-muted-foreground">{quotaLine}</p>}
              <button
                onClick={signOut}
                disabled={busy}
                className="mt-3 w-full rounded-lg border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
              >
                {t("auth.logout")}
              </button>
              {me.user?.role !== "admin" && eraseButton}
            </>
          ) : me.providers.length || me.admin_login ? (
            <>
              {me.providers.length > 0 && (
                <div className="flex flex-col gap-2">
                  {me.providers.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => login(p.id)}
                      disabled={busy}
                      className="w-full rounded-lg border border-border px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-muted disabled:opacity-50"
                    >
                      {t("auth.continueWith", { provider: p.label })}
                    </button>
                  ))}
                </div>
              )}
              {me.providers.length > 0 && me.admin_login && (
                <div className="my-3 flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="h-px flex-1 bg-border" />
                  {t("auth.or")}
                  <span className="h-px flex-1 bg-border" />
                </div>
              )}
              {me.admin_login && (
                <form onSubmit={signInAsAdmin} className="flex flex-col gap-2">
                  <p className="text-xs font-medium text-muted-foreground">{t("auth.adminLogin")}</p>
                  <input
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder={t("auth.username")}
                    autoComplete="username"
                    required
                    className="w-full rounded-lg border border-border bg-background px-3 py-1.5 text-sm text-foreground outline-none focus:border-foreground/40"
                  />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={t("auth.password")}
                    autoComplete="current-password"
                    required
                    className="w-full rounded-lg border border-border bg-background px-3 py-1.5 text-sm text-foreground outline-none focus:border-foreground/40"
                  />
                  <button
                    type="submit"
                    disabled={busy || !username || !password}
                    className="w-full rounded-lg bg-foreground px-3 py-1.5 text-sm font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-50"
                  >
                    {t("auth.login")}
                  </button>
                </form>
              )}
              {me.kind === "anonymous" && (
                <p className="mt-3 text-xs text-muted-foreground">{t("auth.anonymousHint")}</p>
              )}
              {quotaLine && <p className="mt-1 text-xs text-muted-foreground">{quotaLine}</p>}
              {me.kind === "anonymous" && eraseButton}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">{t("auth.noProviders")}</p>
          )}
        </div>
      )}
    </div>
  );
}
