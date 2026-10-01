"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getMe, logout, startLogin, takeAuthError, type AuthMe } from "@/lib/auth";
import { useTranslation } from "@/lib/i18n";

const KNOWN_ERRORS = new Set(["denied", "account_disabled", "account_inactive"]);

/**
 * Login / account control for the navbar.
 *
 * Hidden entirely in single_user mode, where nobody logs in. Signed out it
 * lists the configured providers; signed in it shows the avatar, today's
 * usage and logout.
 */
export function AuthMenu() {
  const { t } = useTranslation();
  const [me, setMe] = useState<AuthMe | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
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
        <div className="absolute right-0 top-full z-50 mt-2 w-64 rounded-xl border border-border bg-background p-3 shadow-lg">
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
            </>
          ) : me.providers.length ? (
            <>
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
              {me.kind === "anonymous" && (
                <p className="mt-3 text-xs text-muted-foreground">{t("auth.anonymousHint")}</p>
              )}
              {quotaLine && <p className="mt-1 text-xs text-muted-foreground">{quotaLine}</p>}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">{t("auth.noProviders")}</p>
          )}
        </div>
      )}
    </div>
  );
}
