"use client";

import { useCallback, useEffect, useState } from "react";
import { describeApiError } from "@/lib/agent";
import { deleteMyData, getMe, getMyData, logout, startLogin, type AuthMe } from "@/lib/auth";
import { useTranslation } from "@/lib/i18n";
import MemoryManager from "@/components/account/MemoryManager";
import ProjectList from "@/components/project/ProjectList";

type Tab = "overview" | "memory" | "projects";
const TABS: Tab[] = ["overview", "memory", "projects"];

function tabFromUrl(): Tab {
  const tab = new URLSearchParams(window.location.search).get("tab");
  return TABS.includes(tab as Tab) ? (tab as Tab) : "overview";
}

/**
 * The reader's own page: who they are here, what the agent remembers about
 * them, and their research projects.
 *
 * It exists for every visitor, not only a logged-in one — an anonymous
 * visitor and the single local user have memories and projects too — so it is
 * reached from the account control wherever that is shown, and never asks for
 * a login before showing what is already theirs.
 */
export default function AccountPage() {
  const { t } = useTranslation();
  const [tab, setTab] = useState<Tab>("overview");
  const [me, setMe] = useState<AuthMe | null>(null);

  // Read after mount (the server render has no query string to agree with),
  // and kept in the URL so a tab can be linked to and survives a reload.
  useEffect(() => {
    setTab(tabFromUrl());
    getMe().then(setMe).catch(() => setMe(null));
  }, []);

  const choose = useCallback((next: Tab) => {
    setTab(next);
    window.history.replaceState(null, "", next === "overview" ? "/account" : `/account?tab=${next}`);
  }, []);

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <h1 className="font-display text-3xl font-bold text-foreground">{t("account.title")}</h1>

      <div role="tablist" className="mb-8 mt-6 flex gap-1 border-b border-border">
        {TABS.map((key) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={tab === key}
            onClick={() => choose(key)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm transition-colors ${
              tab === key
                ? "border-foreground font-medium text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {t(`account.tab.${key}`)}
          </button>
        ))}
      </div>

      {tab === "overview" && <Overview me={me} onOpen={choose} />}
      {tab === "memory" && <MemoryManager />}
      {tab === "projects" && (
        <>
          <p className="mb-5 text-sm leading-relaxed text-muted-foreground">
            {t("project.list.subtitle")}
          </p>
          <ProjectList />
        </>
      )}
    </main>
  );
}

const card = "rounded-2xl border border-border bg-card p-5";
const heading = "mb-3 text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground";
const outlineButton =
  "rounded-lg border border-border px-3.5 py-1.5 text-sm text-foreground transition-colors hover:bg-muted disabled:opacity-50";

function Overview({ me, onOpen }: { me: AuthMe | null; onOpen: (tab: Tab) => void }) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  if (!me) return null;

  const local = me.mode === "single_user";
  const name = local
    ? t("account.identity.local")
    : me.user?.display_name || me.user?.email || t("account.identity.anonymous");
  const note = local
    ? t("account.identity.local_note")
    : me.authenticated
      ? me.user?.email && me.user.email !== name
        ? me.user.email
        : ""
      : t("account.identity.anonymous_note");
  const admin = me.user?.role === "admin";

  const guard = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (err) {
      setError(describeApiError(err, t));
      setBusy(false);
    }
  };

  const signOut = () =>
    guard(async () => {
      await logout();
      // A new visitor now: whatever the page shows belongs to the old one.
      window.location.href = "/";
    });

  const erase = () =>
    guard(async () => {
      const owned = await getMyData();
      if (!window.confirm(t("auth.deleteDataConfirm", { sessions: owned.sessions, runs: owned.runs }))) {
        setBusy(false);
        return;
      }
      await deleteMyData();
      window.location.href = "/";
    });

  const quota = me.quota;

  return (
    <div className="space-y-6">
      <section className={`${card} flex items-center gap-4`}>
        {me.user?.avatar_url ? (
          // eslint-disable-next-line @next/next/no-img-element -- provider avatars come from arbitrary hosts
          <img
            src={me.user.avatar_url}
            alt=""
            referrerPolicy="no-referrer"
            className="h-14 w-14 shrink-0 rounded-full object-cover"
          />
        ) : (
          <span className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full bg-muted text-xl font-medium text-muted-foreground">
            {name.slice(0, 1).toUpperCase()}
          </span>
        )}
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-2 truncate text-lg font-medium text-foreground">
            <span className="truncate">{name}</span>
            {admin && (
              <span className="shrink-0 rounded bg-muted px-1.5 py-[1px] text-[0.65rem] font-normal text-muted-foreground">
                {t("account.identity.admin")}
              </span>
            )}
          </p>
          {note && <p className="mt-0.5 text-sm leading-relaxed text-muted-foreground">{note}</p>}
        </div>
        {me.authenticated && (
          <button type="button" disabled={busy} onClick={() => void signOut()} className={outlineButton}>
            {t("auth.logout")}
          </button>
        )}
      </section>

      {!local && !me.authenticated && me.providers.length > 0 && (
        <section className={card}>
          <h2 className={heading}>{t("auth.login")}</h2>
          <div className="flex flex-wrap gap-2">
            {me.providers.map((p) => (
              <button
                key={p.id}
                type="button"
                disabled={busy}
                onClick={() => void guard(() => startLogin(p.id))}
                className={outlineButton}
              >
                {t("auth.continueWith", { provider: p.label })}
              </button>
            ))}
          </div>
        </section>
      )}

      {quota && (
        <section className={card}>
          <h2 className={heading}>{t("account.usage.heading")}</h2>
          <div className="grid gap-5 sm:grid-cols-2">
            <Usage label={t("account.usage.runs")} used={quota.runs} limit={quota.runs_limit} />
            <Usage label={t("account.usage.tokens")} used={quota.tokens} limit={quota.tokens_limit} />
          </div>
          <p className="mt-3 text-xs text-muted-foreground">{t("account.usage.reset")}</p>
        </section>
      )}

      <section className="grid gap-3 sm:grid-cols-2">
        {(["memory", "projects"] as const).map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => onOpen(key)}
            className={`${card} lift text-left`}
          >
            <span className="block text-sm font-medium text-foreground">{t(`account.tab.${key}`)}</span>
            <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">
              {t(`account.card.${key}`)}
            </span>
          </button>
        ))}
      </section>

      {!local && !admin && (
        <section className={card}>
          <h2 className={heading}>{t("account.data.heading")}</h2>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="min-w-0 flex-1 text-sm leading-relaxed text-muted-foreground">
              {t("account.data.note")}
            </p>
            <button
              type="button"
              disabled={busy}
              onClick={() => void erase()}
              className="rounded-lg border border-destructive/30 px-3.5 py-1.5 text-sm text-destructive transition-colors hover:bg-destructive/5 disabled:opacity-50"
            >
              {t("auth.deleteData")}
            </button>
          </div>
        </section>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

function Usage({ label, used, limit }: { label: string; used: number; limit: number }) {
  const { t } = useTranslation();
  const share = limit > 0 ? Math.min(used / limit, 1) : 0;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className="text-sm tabular-nums text-foreground">
          {used.toLocaleString()}
          <span className="text-muted-foreground">
            {" / "}
            {limit > 0 ? limit.toLocaleString() : t("account.usage.unlimited")}
          </span>
        </span>
      </div>
      {limit > 0 && (
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
          <div
            className={`h-full rounded-full ${share >= 1 ? "bg-destructive" : "bg-foreground/60"}`}
            style={{ width: `${share * 100}%` }}
          />
        </div>
      )}
    </div>
  );
}
