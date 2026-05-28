"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { listRecentRuns } from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import { IconArrowRight, IconCheck } from "@/components/icons";
import type { RecentRunResponse } from "@/lib/types";

interface Props {
  /** Auto-refresh interval in ms while there are active runs. 0 disables. */
  refreshMs?: number;
  /** How many rows to show in the "history" list (active runs always all shown). */
  historyLimit?: number;
}

function relativeTime(iso: string, locale: string): string {
  if (!iso) return "";
  // SQLite datetime('now') returns "YYYY-MM-DD HH:MM:SS" in UTC — treat as UTC.
  const ts = iso.includes("T") ? Date.parse(iso) : Date.parse(iso.replace(" ", "T") + "Z");
  if (Number.isNaN(ts)) return iso;
  const diff = Math.max(0, Date.now() - ts);
  const sec = Math.floor(diff / 1000);
  const min = Math.floor(sec / 60);
  const hr = Math.floor(min / 60);
  const day = Math.floor(hr / 24);
  if (locale === "zh") {
    if (sec < 60) return `${sec} 秒前`;
    if (min < 60) return `${min} 分钟前`;
    if (hr < 24) return `${hr} 小时前`;
    return `${day} 天前`;
  }
  if (sec < 60) return `${sec}s ago`;
  if (min < 60) return `${min}m ago`;
  if (hr < 24) return `${hr}h ago`;
  return `${day}d ago`;
}

export default function RecentRuns({ refreshMs = 5000, historyLimit = 8 }: Props) {
  const { t, locale } = useTranslation();
  const [runs, setRuns] = useState<RecentRunResponse[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      listRecentRuns(20, false)
        .then((data) => {
          if (!cancelled) setRuns(data);
        })
        .catch(() => {
          if (!cancelled) setRuns([]);
        });
    };
    load();
    if (refreshMs <= 0) return;
    const id = setInterval(load, refreshMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [refreshMs]);

  if (runs === null || runs.length === 0) return null;

  const active = runs.filter((r) => r.status === "running" || r.status === "pending");
  const history = runs.filter((r) => r.status !== "running" && r.status !== "pending").slice(0, historyLimit);

  return (
    <section className="mb-8 space-y-3">
      {active.length > 0 && (
        <div className="rounded-2xl border border-primary/40 bg-primary/5 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-primary">
            <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />
            {t("recent.active_header", { count: active.length })}
          </div>
          <ul className="space-y-1.5">
            {active.map((r) => (
              <ActiveRow key={r.run_id} run={r} t={t} locale={locale} />
            ))}
          </ul>
        </div>
      )}

      {history.length > 0 && (
        <details className="rounded-2xl border border-border bg-card/60 p-4">
          <summary className="cursor-pointer select-none text-sm font-semibold text-foreground/80">
            {t("recent.history_header", { count: history.length })}
          </summary>
          <ul className="mt-3 space-y-1.5">
            {history.map((r) => (
              <HistoryRow key={r.run_id} run={r} t={t} locale={locale} />
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function ActiveRow({
  run,
  t,
  locale,
}: {
  run: RecentRunResponse;
  t: (k: string, vars?: Record<string, string | number>) => string;
  locale: string;
}) {
  const stepLabel = run.current_step ? t(`step.${run.current_step}`) || run.current_step : t("recent.starting");
  const modeLabel = t(`upload.mode.${run.mode}.label`) || run.mode;
  return (
    <li>
      <Link
        href={`/paper/${run.paper_id}/run/${run.run_id}`}
        className="group flex items-center gap-3 rounded-xl px-3 py-2 transition-colors hover:bg-primary/10"
      >
        <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">
            {run.paper_title || run.paper_id}
          </span>
          <span className="block truncate text-xs text-muted-foreground">
            {modeLabel} · {stepLabel} · {relativeTime(run.started_at, locale)}
          </span>
        </span>
        <IconArrowRight className="shrink-0 text-base text-muted-foreground transition-colors group-hover:text-primary" />
      </Link>
    </li>
  );
}

function HistoryRow({
  run,
  t,
  locale,
}: {
  run: RecentRunResponse;
  t: (k: string, vars?: Record<string, string | number>) => string;
  locale: string;
}) {
  const modeLabel = t(`upload.mode.${run.mode}.label`) || run.mode;
  const ok = run.status === "done";
  return (
    <li>
      <Link
        href={`/paper/${run.paper_id}/run/${run.run_id}`}
        className="group flex items-center gap-3 rounded-xl px-3 py-2 transition-colors hover:bg-muted"
      >
        <span
          className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] ${
            ok ? "bg-success/15 text-success" : "bg-destructive/15 text-destructive"
          }`}
        >
          {ok ? <IconCheck /> : "!"}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm">
            {run.paper_title || run.paper_id}
          </span>
          <span className="block truncate text-xs text-muted-foreground">
            {modeLabel} · {relativeTime(run.finished_at || run.started_at, locale)}
          </span>
        </span>
        <IconArrowRight className="shrink-0 text-base text-muted-foreground transition-colors group-hover:text-foreground/70" />
      </Link>
    </li>
  );
}
