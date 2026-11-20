"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { describeApiError } from "@/lib/agent";
import { deleteMyData, getMe, getMyData, logout, startLogin, type AuthMe } from "@/lib/auth";
import { useTranslation } from "@/lib/i18n";
import MemoryManager from "@/components/account/MemoryManager";

type Tab = "overview" | "memory";
const TABS: Tab[] = ["overview", "memory"];

/**
 * Account, quota & long-term memory center.
 *
 * Research projects live directly in the workspace sidebar (`/projects`); legacy
 * `/account?tab=projects` links redirect there cleanly.
 */
export default function AccountPage() {
  const { t } = useTranslation();
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("overview");
  const [me, setMe] = useState<AuthMe | null>(null);

  useEffect(() => {
    const raw = new URLSearchParams(window.location.search).get("tab");
    if (raw === "projects") {
      router.replace("/projects");
      return;
    }
    if (raw === "memory") setTab("memory");
    getMe().then(setMe).catch(() => setMe(null));
  }, [router]);

  const choose = useCallback((next: Tab) => {
    setTab(next);
    window.history.replaceState(null, "", next === "overview" ? "/account" : `/account?tab=${next}`);
  }, []);

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="font-display text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">
        {t("account.title")}
      </h1>

      <div role="tablist" className="mb-8 mt-5 flex gap-1 border-b border-border">
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
    </div>
  );
}

const card = "rounded-2xl border border-border bg-card p-5 soft-shadow";
const heading = "mb-3 text-xs font-semibold text-muted-foreground";
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
    <div className="space-y-5">
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

      <button
        type="button"
        onClick={() => onOpen("memory")}
        className={`${card} lift block w-full text-left`}
      >
        <span className="block text-sm font-medium text-foreground">{t("account.tab.memory")}</span>
        <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">
          {t("account.card.memory")}
        </span>
      </button>

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
