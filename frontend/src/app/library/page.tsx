"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { describeApiError, listOwnerPapers } from "@/lib/agent";
import type { SessionPaper } from "@/lib/agent";
import {
  askLibrary,
  deleteRun,
  getLibraryMarkdown,
  getMarkdownExportUrl,
  getRunOutput,
  listLibraryDocuments,
  listRecentRuns,
  searchLibrary,
} from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import { mergeLibraryItems } from "@/lib/libraryHub";
import type { LibraryHubItem } from "@/lib/libraryHub";
import { parseSnapData } from "@/lib/snap";
import type {
  LibraryAskResponse,
  LibraryDocument,
  LibrarySearchRecord,
  RecentRunResponse,
  SearchMethod,
} from "@/lib/types";
import { useChatShell } from "@/components/agent/ChatShell";
import { AVAILABILITY_CLASS } from "@/components/agent/PaperList";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import SnapMatrix, { type MatrixEntry } from "@/components/snap/SnapMatrix";
import SplitPane from "@/components/SplitPane";
import {
  IconArrowRight,
  IconChat,
  IconCheck,
  IconDownload,
  IconLens,
  IconPlus,
  IconSearch,
  IconSnap,
  IconSparkles,
  IconSphere,
  IconTable,
  IconTrash,
  IconX,
} from "@/components/icons";

type HubView = "papers" | "compare" | "corpus";
type PaperFilter = "all" | "reports" | "readable";

const MODE_META = {
  snap: { Icon: IconSnap, labelKey: "chat.mode.snap.label" },
  lens: { Icon: IconLens, labelKey: "chat.mode.lens.label" },
  sphere: { Icon: IconSphere, labelKey: "chat.mode.sphere.label" },
} as const;

function relativeTime(iso: string, locale: string): string {
  if (!iso) return "";
  const ts = iso.includes("T") ? Date.parse(iso) : Date.parse(iso.replace(" ", "T") + "Z");
  if (Number.isNaN(ts)) return iso.slice(0, 10);
  const diff = Math.max(0, Date.now() - ts);
  const min = Math.floor(diff / 60000);
  const hr = Math.floor(min / 60);
  const day = Math.floor(hr / 24);
  if (locale === "zh") {
    if (min < 1) return "刚刚";
    if (min < 60) return `${min} 分钟前`;
    if (hr < 24) return `${hr} 小时前`;
    return `${day} 天前`;
  }
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  if (hr < 24) return `${hr}h ago`;
  return `${day}d ago`;
}

export default function LibraryPage() {
  return (
    <Suspense fallback={<div className="h-full" />}>
      <LibraryHub />
    </Suspense>
  );
}

function LibraryHub() {
  const { t, locale } = useTranslation();
  const searchParams = useSearchParams();
  const initialViewParam = searchParams.get("view");
  const initialDocId = searchParams.get("doc") || "";
  const { libraryEnabled, refreshRuns } = useChatShell();

  const showCorpusTab = libraryEnabled || Boolean(initialDocId);

  const [view, setView] = useState<HubView>(() => {
    if (initialDocId) return "corpus";
    if (initialViewParam === "compare") return "compare";
    return "papers";
  });

  useEffect(() => {
    if (initialDocId) setView("corpus");
    else if (initialViewParam === "compare") setView("compare");
  }, [initialDocId, initialViewParam]);

  const [papers, setPapers] = useState<SessionPaper[]>([]);
  const [runs, setRuns] = useState<RecentRunResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<PaperFilter>("all");
  const [search, setSearch] = useState("");
  const [selectedPaperIds, setSelectedPaperIds] = useState<Set<string>>(new Set());
  const [preselectedSnapRunIds, setPreselectedSnapRunIds] = useState<string[] | null>(null);

  const loadHub = useCallback(async () => {
    const [papersRes, runsRes] = await Promise.all([
      listOwnerPapers(200).catch(() => ({ papers: [] as SessionPaper[] })),
      listRecentRuns(100, false, 0).catch(() => [] as RecentRunResponse[]),
    ]);
    setPapers(papersRes.papers);
    setRuns(runsRes);
    setLoading(false);
  }, []);

  useEffect(() => {
    void loadHub();
  }, [loadHub]);

  const activeRuns = useMemo(
    () => runs.filter((r) => r.status === "pending" || r.status === "running"),
    [runs],
  );

  // Poll while background mode runs are still executing.
  useEffect(() => {
    if (activeRuns.length === 0) return;
    const id = setInterval(() => {
      listRecentRuns(100, false, 0)
        .then((next) => {
          setRuns(next);
          refreshRuns();
        })
        .catch(() => {});
    }, 5000);
    return () => clearInterval(id);
  }, [activeRuns.length, refreshRuns]);

  const items = useMemo(() => mergeLibraryItems(papers, runs), [papers, runs]);

  const doneSnapRuns = useMemo(
    () => runs.filter((r) => r.mode === "snap" && r.status === "done"),
    [runs],
  );

  const filteredItems = useMemo(() => {
    const q = search.trim().toLowerCase();
    return items.filter((item) => {
      const doneRuns = item.runs.filter((r) => r.status === "done");
      if (filter === "reports" && doneRuns.length === 0) return false;
      if (filter === "readable" && item.availability !== "parsed" && !item.paperId) return false;
      if (!q) return true;
      return (
        item.title.toLowerCase().includes(q) ||
        item.venue.toLowerCase().includes(q) ||
        (item.yearKnown && String(item.year).includes(q))
      );
    });
  }, [items, filter, search]);

  const toggleSelect = useCallback((paperId: string) => {
    if (!paperId) return;
    setSelectedPaperIds((prev) => {
      const next = new Set(prev);
      if (next.has(paperId)) next.delete(paperId);
      else next.add(paperId);
      return next;
    });
  }, []);

  const handleDeleteRun = useCallback(
    (runId: string) => {
      if (!window.confirm(t("run.delete_confirm"))) return;
      setRuns((prev) => prev.filter((r) => r.run_id !== runId));
      deleteRun(runId)
        .then(() => refreshRuns())
        .catch(() => {});
    },
    [refreshRuns, t],
  );

  const selectedSnapRunIds = useMemo(() => {
    const ids: string[] = [];
    for (const item of items) {
      if (!selectedPaperIds.has(item.paperId)) continue;
      const snap = item.runs.find((r) => r.mode === "snap" && r.status === "done");
      if (snap) ids.push(snap.run_id);
    }
    return ids;
  }, [items, selectedPaperIds]);

  const openSelectedInMatrix = useCallback(() => {
    setPreselectedSnapRunIds(selectedSnapRunIds);
    setView("compare");
  }, [selectedSnapRunIds]);

  const tabs: { key: HubView; label: string; badge?: number }[] = [
    { key: "papers", label: t("hub.tab.papers"), badge: items.length || undefined },
    { key: "compare", label: t("hub.tab.compare"), badge: doneSnapRuns.length || undefined },
  ];
  if (showCorpusTab) {
    tabs.push({ key: "corpus", label: t("hub.tab.corpus") });
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="shrink-0 border-b border-border bg-background px-5 py-4 sm:px-8">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <h1 className="font-display text-2xl font-semibold tracking-tight text-foreground">
              {t("hub.title")}
            </h1>
            <p className="mt-0.5 text-xs text-muted-foreground">{t("hub.subtitle")}</p>
          </div>

          <div className="flex items-center gap-2">
            <div role="tablist" className="flex items-center gap-0.5 rounded-xl border border-border bg-muted/50 p-1">
              {tabs.map((tabItem) => {
                const active = view === tabItem.key;
                return (
                  <button
                    key={tabItem.key}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => setView(tabItem.key)}
                    className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                      active
                        ? "bg-card text-foreground soft-shadow"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <span>{tabItem.label}</span>
                    {tabItem.badge !== undefined && (
                      <span className="rounded-full bg-muted px-1.5 py-[1px] text-[10.5px] tabular-nums text-muted-foreground">
                        {tabItem.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            <Link
              href="/chat"
              className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-3.5 py-2 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
            >
              <IconPlus className="text-[13px]" />
              <span className="hidden sm:inline">{t("hub.new_chat")}</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Body */}
      {view === "corpus" ? (
        <div className="min-h-0 flex-1 overflow-hidden">
          <ExternalCorpusView initialDocId={initialDocId} />
        </div>
      ) : view === "compare" ? (
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-8">
          <div className="mx-auto max-w-5xl">
            <CompareMatrixView
              snapRuns={doneSnapRuns}
              preselectedRunIds={preselectedSnapRunIds}
            />
          </div>
        </div>
      ) : (
        <div className="relative min-h-0 flex-1 overflow-y-auto px-5 pb-24 pt-6 sm:px-8">
          <div className="mx-auto max-w-5xl space-y-5">
            {/* Active background runs */}
            {activeRuns.length > 0 && (
              <section className="rounded-2xl border border-primary/35 bg-accent/40 p-4">
                <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-accent-foreground">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />
                  {t("recent.active_header", { count: activeRuns.length })}
                </div>
                <ul className="space-y-1">
                  {activeRuns.map((run) => {
                    const step = run.current_step
                      ? t(`step.${run.current_step}`) || run.current_step
                      : t("recent.starting");
                    const modeLabel = t(`upload.mode.${run.mode}.label`) || run.mode;
                    return (
                      <li
                        key={run.run_id}
                        className="flex items-center rounded-xl bg-card/80 px-3 py-2 transition-colors hover:bg-card"
                      >
                        {/* A report is written, and stopped, in its conversation. */}
                        <Link
                          href={
                            run.agent_session_id
                              ? `/chat/${run.agent_session_id}`
                              : `/paper/${run.paper_id}/run/${run.run_id}`
                          }
                          className="flex min-w-0 flex-1 items-center gap-3"
                        >
                          <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-medium text-foreground">
                              {run.paper_title || run.paper_id}
                            </span>
                            <span className="block truncate text-xs text-muted-foreground">
                              {modeLabel} · {step} · {relativeTime(run.started_at, locale)}
                            </span>
                          </span>
                          <IconArrowRight className="shrink-0 text-sm text-muted-foreground" />
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </section>
            )}

            {/* Filter & search controls */}
            {items.length > 0 && (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="relative min-w-[15rem] flex-1 sm:max-w-sm">
                  <IconSearch className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[14px] text-muted-foreground" />
                  <input
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder={t("hub.search_placeholder")}
                    className="w-full rounded-xl border border-border bg-card py-2 pl-9 pr-8 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/50 focus:outline-none"
                  />
                  {search && (
                    <button
                      type="button"
                      onClick={() => setSearch("")}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      <IconX className="text-[13px]" />
                    </button>
                  )}
                </div>

                <div className="flex items-center gap-1 rounded-xl border border-border bg-card p-1 text-xs">
                  {(
                    [
                      ["all", t("hub.filter.all")],
                      ["reports", t("hub.filter.with_reports")],
                      ["readable", t("hub.filter.readable")],
                    ] as const
                  ).map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setFilter(key)}
                      className={`rounded-lg px-2.5 py-1 transition-colors ${
                        filter === key
                          ? "bg-muted font-medium text-foreground"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* List */}
            {loading ? (
              <div className="flex items-center gap-2.5 py-12 text-sm text-muted-foreground">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                {t("chat.cite.loading")}
              </div>
            ) : items.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-border bg-card/60 px-6 py-14 text-center">
                <p className="mx-auto max-w-md text-sm leading-relaxed text-muted-foreground">
                  {t("hub.empty")}
                </p>
                <Link
                  href="/chat"
                  className="mt-5 inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
                >
                  <IconSparkles className="text-[15px]" />
                  {t("home.cta")}
                </Link>
              </div>
            ) : filteredItems.length === 0 ? (
              <p className="py-12 text-center text-sm text-muted-foreground">{t("hub.no_match")}</p>
            ) : (
              <div className="space-y-2.5">
                {filteredItems.map((item) => (
                  <PaperHubRow
                    key={item.key}
                    item={item}
                    locale={locale}
                    selected={Boolean(item.paperId && selectedPaperIds.has(item.paperId))}
                    onToggleSelect={() => toggleSelect(item.paperId)}
                    onDeleteRun={handleDeleteRun}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Floating multi-select bar */}
          {selectedPaperIds.size > 0 && (
            <div className="animate-fade-in fixed bottom-5 left-1/2 z-30 flex -translate-x-1/2 items-center gap-2 rounded-2xl border border-border bg-card px-4 py-2.5 shadow-xl">
              <span className="text-xs font-medium text-foreground">
                {t("hub.selected_count", { count: selectedPaperIds.size })}
              </span>
              <div className="mx-1 h-4 w-px bg-border" />
              <Link
                href={`/chat?papers=${[...selectedPaperIds].join(",")}`}
                className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
              >
                <IconChat className="text-[13px]" />
                {t("hub.compare_chat")}
              </Link>
              {selectedSnapRunIds.length > 0 && (
                <button
                  type="button"
                  onClick={openSelectedInMatrix}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted"
                >
                  <IconTable className="text-[13px]" />
                  {t("hub.compare_matrix")}
                </button>
              )}
              <button
                type="button"
                onClick={() => setSelectedPaperIds(new Set())}
                className="rounded-lg px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                {t("hub.clear_selection")}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PaperHubRow({
  item,
  locale,
  selected,
  onToggleSelect,
  onDeleteRun,
}: {
  item: LibraryHubItem;
  locale: string;
  selected: boolean;
  onToggleSelect: () => void;
  onDeleteRun: (runId: string) => void;
}) {
  const { t } = useTranslation();
  const doneRuns = item.runs.filter((r) => r.status === "done");
  const chatHref = item.sessionId
    ? `/chat/${item.sessionId}`
    : item.paperId
      ? `/chat?paper=${item.paperId}`
      : "/chat";

  return (
    <div
      className={`rounded-2xl border bg-card p-4 transition-colors ${
        selected ? "border-primary/50 bg-accent/20" : "border-border hover:border-foreground/20"
      }`}
    >
      <div className="flex items-start gap-3">
        {item.paperId ? (
          <button
            type="button"
            role="checkbox"
            aria-checked={selected}
            onClick={onToggleSelect}
            className={`mt-0.5 flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded border transition-colors ${
              selected
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-background hover:border-foreground/40"
            }`}
          >
            {selected && <IconCheck className="text-[11px]" />}
          </button>
        ) : (
          <span className="mt-0.5 h-4.5 w-4.5 shrink-0" />
        )}

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <h2 className="min-w-0 flex-1 text-sm font-medium leading-snug text-foreground">
              {item.title || item.paperId.slice(0, 16) || t("chat.papers.untitled")}
            </h2>
            {item.paperId && (
              <Link
                href={chatHref}
                className="inline-flex shrink-0 items-center gap-1 rounded-lg border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <IconChat className="text-[12px]" />
                {item.sessionId ? t("hub.continue_chat") : t("hub.open_chat")}
              </Link>
            )}
          </div>

          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className={`rounded px-1.5 py-[1px] text-[11px] ${AVAILABILITY_CLASS[item.availability]}`}>
              {t(`chat.availability.${item.availability}`)}
            </span>
            {item.yearKnown && <span>{item.year}</span>}
            {item.venue && <span className="max-w-[18rem] truncate">{item.venue}</span>}
            {item.updatedAt && <span>{relativeTime(item.updatedAt, locale)}</span>}
          </div>

          {doneRuns.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              {doneRuns.map((run) => {
                const meta = MODE_META[run.mode as keyof typeof MODE_META];
                const RunIcon = meta?.Icon ?? IconSparkles;
                const label = meta ? t(meta.labelKey) : run.mode;
                return (
                  <span
                    key={run.run_id}
                    className="group/chip inline-flex items-center rounded-lg border border-border bg-muted/50 text-xs transition-colors hover:border-primary/35 hover:bg-accent/50"
                  >
                    <Link
                      href={`/paper/${run.paper_id}/run/${run.run_id}`}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 font-medium text-foreground"
                    >
                      <RunIcon className="text-[12px] text-primary" />
                      <span>{label}</span>
                    </Link>
                    <a
                      href={getMarkdownExportUrl(run.run_id)}
                      title={t("run.export_md")}
                      className="px-1 py-1 text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <IconDownload className="text-[11px]" />
                    </a>
                    <button
                      type="button"
                      onClick={() => onDeleteRun(run.run_id)}
                      title={t("run.delete")}
                      aria-label={t("run.delete")}
                      className="pr-1.5 pl-0.5 py-1 text-muted-foreground transition-colors hover:text-destructive"
                    >
                      <IconTrash className="text-[11px]" />
                    </button>
                  </span>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CompareMatrixView({
  snapRuns,
  preselectedRunIds,
}: {
  snapRuns: RecentRunResponse[];
  preselectedRunIds: string[] | null;
}) {
  const { t } = useTranslation();
  const [entries, setEntries] = useState<Map<string, MatrixEntry>>(new Map());
  const [unavailable, setUnavailable] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      const loaded = new Map<string, MatrixEntry>();
      const missing = new Set<string>();
      await Promise.all(
        snapRuns.map(async (run) => {
          const output = await getRunOutput(run.run_id).catch(() => null);
          const data = output ? parseSnapData(output.json_data) : null;
          if (data) {
            loaded.set(run.run_id, { runId: run.run_id, paperId: run.paper_id, data });
          } else {
            missing.add(run.run_id);
          }
        }),
      );
      if (cancelled) return;
      setEntries(loaded);
      setUnavailable(missing);
      const initial =
        preselectedRunIds && preselectedRunIds.length > 0
          ? preselectedRunIds.filter((id) => loaded.has(id))
          : [...loaded.keys()].slice(0, 5);
      setSelected(new Set(initial));
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [snapRuns, preselectedRunIds]);

  const toggle = useCallback((runId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  }, []);

  const visible = useMemo(
    () => [...selected].map((id) => entries.get(id)).filter((e): e is MatrixEntry => !!e),
    [selected, entries],
  );

  const visiblePaperIds = useMemo(
    () => [...new Set(visible.map((e) => e.paperId).filter(Boolean))],
    [visible],
  );

  if (loading) {
    return (
      <div className="flex items-center gap-3 py-10 text-sm text-muted-foreground">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        {t("matrix.loading")}
      </div>
    );
  }

  if (entries.size === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card/60 p-10 text-center">
        <p className="text-sm text-muted-foreground">{t("matrix.empty")}</p>
        <Link
          href="/chat?mode=snap"
          className="mt-4 inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
        >
          <IconSnap className="text-[14px]" />
          {t("chat.mode.snap.label")}
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs font-medium text-muted-foreground">
          {t("matrix.pick", { n: String(selected.size), total: String(entries.size) })}
        </p>
        {visiblePaperIds.length >= 2 && (
          <Link
            href={`/chat?papers=${visiblePaperIds.join(",")}`}
            className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted"
          >
            <IconChat className="text-[13px] text-primary" />
            {t("hub.compare_chat")} ({visiblePaperIds.length})
          </Link>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {snapRuns.map((run) => {
          const entry = entries.get(run.run_id);
          const isMissing = unavailable.has(run.run_id);
          const isOn = selected.has(run.run_id);
          return (
            <button
              key={run.run_id}
              type="button"
              disabled={isMissing}
              onClick={() => toggle(run.run_id)}
              aria-pressed={isOn}
              title={isMissing ? t("matrix.unavailable") : run.paper_title}
              className={`max-w-[20rem] truncate rounded-lg border px-2.5 py-1.5 text-left text-xs transition-colors ${
                isMissing
                  ? "cursor-not-allowed border-dashed border-border text-muted-foreground/60"
                  : isOn
                    ? "border-primary/45 bg-accent text-accent-foreground"
                    : "border-border bg-card text-foreground/80 hover:border-primary/30 hover:bg-muted"
              }`}
            >
              {entry?.data.title || run.paper_title || run.paper_id}
            </button>
          );
        })}
      </div>

      {unavailable.size > 0 && (
        <p className="text-xs italic text-muted-foreground">
          {t("matrix.unavailable_n", { n: String(unavailable.size) })}
        </p>
      )}

      {visible.length > 0 ? (
        <SnapMatrix entries={visible} />
      ) : (
        <p className="rounded-xl border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
          {t("matrix.select_prompt")}
        </p>
      )}
    </div>
  );
}

const METHODS: SearchMethod[] = ["full_text_search", "semantic_search", "hybrid_search"];

function fmtScore(score: number | null | undefined): string | null {
  return typeof score === "number" && score > 0 ? score.toFixed(3) : null;
}

/**
 * External Dify knowledge-base browser & RAG Q&A, mounted only when enabled
 * (or deep-linked via `?doc=`) so unconfigured deployments never see a 503.
 */
function ExternalCorpusView({ initialDocId }: { initialDocId: string }) {
  const { t, locale } = useTranslation();
  const [tab, setTab] = useState<"search" | "ask">("search");
  const [method, setMethod] = useState<SearchMethod>("full_text_search");
  const [error, setError] = useState<string>("");

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<LibrarySearchRecord[]>([]);
  const [searched, setSearched] = useState(false);
  const [searching, setSearching] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<LibraryAskResponse | null>(null);
  const [asking, setAsking] = useState(false);

  const [docs, setDocs] = useState<LibraryDocument[]>([]);
  const [docsPage, setDocsPage] = useState(0);
  const [docsHasMore, setDocsHasMore] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);

  const [selectedName, setSelectedName] = useState("");
  const [docContent, setDocContent] = useState("");
  const [docLoading, setDocLoading] = useState(false);

  const loadDocs = useCallback(
    async (page: number) => {
      setDocsLoading(true);
      try {
        const res = await listLibraryDocuments({ page, limit: 20 });
        setDocs((prev) => (page === 1 ? res.data : [...prev, ...res.data]));
        setDocsHasMore(Boolean(res.has_more));
        setDocsPage(page);
        setError("");
      } catch (err) {
        setError(describeApiError(err, t));
      } finally {
        setDocsLoading(false);
      }
    },
    [t],
  );

  const openDoc = useCallback(
    async (documentId: string, name: string) => {
      if (!documentId) return;
      setDocLoading(true);
      setSelectedName(name);
      try {
        const res = await getLibraryMarkdown(documentId);
        setDocContent(res.content || "");
        if (!name && res.document_name) setSelectedName(res.document_name);
        setError("");
      } catch (err) {
        setDocContent("");
        setError(describeApiError(err, t));
      } finally {
        setDocLoading(false);
      }
    },
    [t],
  );

  useEffect(() => {
    void loadDocs(1);
  }, [loadDocs]);

  useEffect(() => {
    if (initialDocId) void openDoc(initialDocId, "");
  }, [initialDocId, openDoc]);

  const runSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) {
      setError(t("library.empty_query"));
      return;
    }
    setSearching(true);
    setError("");
    try {
      const res = await searchLibrary({ query: q, search_method: method, top_k: 20 });
      setResults(res.records || []);
      setSearched(true);
    } catch (err) {
      setError(describeApiError(err, t));
    } finally {
      setSearching(false);
    }
  }, [query, method, t]);

  const runAsk = useCallback(async () => {
    const q = question.trim();
    if (!q) {
      setError(t("library.empty_question"));
      return;
    }
    setAsking(true);
    setError("");
    try {
      const res = await askLibrary({
        question: q,
        search_method: method,
        language: locale,
        top_k: 10,
      });
      setAnswer(res);
    } catch (err) {
      setError(describeApiError(err, t));
    } finally {
      setAsking(false);
    }
  }, [question, method, locale, t]);

  const onLibraryCitationClick = useCallback(
    (idx: number) => {
      const src = answer?.sources.find((s) => s.idx === idx);
      if (src) void openDoc(src.document_id, src.document_name);
    },
    [answer, openDoc],
  );

  return (
    <SplitPane
      defaultLeftWidth={48}
      left={
        <div className="flex h-full flex-col">
          <div className="shrink-0 space-y-2.5 border-b border-border px-5 py-3">
            <div className="flex items-center justify-between gap-2">
              <div className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5 text-xs">
                {(["search", "ask"] as const).map((key) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setTab(key)}
                    className={`rounded-md px-2.5 py-1 transition-colors ${
                      tab === key ? "bg-card font-medium text-foreground soft-shadow" : "text-muted-foreground"
                    }`}
                  >
                    {t(`library.tab.${key}`)}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-muted-foreground">{t("library.method_label")}</span>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value as SearchMethod)}
                  className="rounded-lg border border-border bg-card px-2 py-1 text-xs"
                >
                  {METHODS.map((m) => (
                    <option key={m} value={m}>
                      {t(`library.method.${m}`)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {tab === "search" ? (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void runSearch();
                }}
                className="flex items-center gap-2"
              >
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={t("library.search_placeholder")}
                  className="min-w-0 flex-1 rounded-lg border border-border bg-card px-3 py-1.5 text-sm outline-none focus:border-primary/50"
                />
                <button
                  type="submit"
                  disabled={searching}
                  className="shrink-0 rounded-lg bg-primary px-3.5 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-50"
                >
                  {searching ? t("library.searching") : t("library.run_search")}
                </button>
              </form>
            ) : (
              <div className="space-y-2">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                      e.preventDefault();
                      void runAsk();
                    }
                  }}
                  placeholder={t("library.ask_placeholder")}
                  rows={2}
                  className="w-full resize-y rounded-lg border border-border bg-card px-3 py-2 text-sm outline-none focus:border-primary/50"
                />
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => void runAsk()}
                    disabled={asking}
                    className="rounded-lg bg-primary px-3.5 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-50"
                  >
                    {asking ? t("library.thinking") : t("library.run_ask")}
                  </button>
                </div>
              </div>
            )}

            {error && <p className="text-xs text-destructive">{error}</p>}
          </div>

          <div className="relative min-h-0 flex-1 overflow-y-auto px-5 py-3">
            {tab === "search" ? (
              searching ? (
                <p className="py-10 text-center text-sm text-muted-foreground">{t("library.searching")}</p>
              ) : searched ? (
                results.length === 0 ? (
                  <p className="py-10 text-center text-sm text-muted-foreground">{t("library.no_results")}</p>
                ) : (
                  <div className="space-y-2">
                    <p className="px-1 text-xs text-muted-foreground">
                      {t("library.results_count", { count: results.length })}
                    </p>
                    {results.map((r, i) => (
                      <CorpusDocCard
                        key={`${r.segment_id}-${i}`}
                        name={r.document_name}
                        score={fmtScore(r.score)}
                        snippet={r.content}
                        onClick={() => void openDoc(r.document_id, r.document_name)}
                      />
                    ))}
                  </div>
                )
              ) : (
                <div className="space-y-2">
                  <p className="px-1 text-xs font-medium text-muted-foreground">{t("library.documents")}</p>
                  {docs.map((d) => (
                    <CorpusDocCard
                      key={d.id}
                      name={d.name}
                      score={d.word_count ? `${d.word_count.toLocaleString()} w` : null}
                      onClick={() => void openDoc(d.id, d.name)}
                    />
                  ))}
                  {docsHasMore && (
                    <button
                      type="button"
                      onClick={() => void loadDocs(docsPage + 1)}
                      disabled={docsLoading}
                      className="w-full rounded-lg border border-border py-2 text-xs text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50"
                    >
                      {docsLoading ? t("library.searching") : t("library.load_more")}
                    </button>
                  )}
                </div>
              )
            ) : asking ? (
              <p className="py-10 text-center text-sm text-muted-foreground">{t("library.thinking")}</p>
            ) : answer ? (
              <div className="space-y-5">
                <MarkdownRenderer
                  content={answer.markdown}
                  onLibraryCitationClick={onLibraryCitationClick}
                />
                {answer.sources.length > 0 && (
                  <div className="border-t border-border pt-3">
                    <p className="mb-2 text-xs font-medium text-muted-foreground">{t("library.sources")}</p>
                    <div className="space-y-1.5">
                      {answer.sources.map((s) => (
                        <button
                          key={s.idx}
                          type="button"
                          onClick={() => void openDoc(s.document_id, s.document_name)}
                          className="flex w-full items-start gap-2 rounded-lg border border-border bg-card px-3 py-2 text-left text-sm transition-colors hover:border-primary/40"
                        >
                          <span className="mt-0.5 shrink-0 font-mono text-xs text-primary">L{s.idx}</span>
                          <span className="min-w-0 flex-1 truncate">{s.document_name}</span>
                          {fmtScore(s.score) && (
                            <span className="shrink-0 text-xs text-muted-foreground">{fmtScore(s.score)}</span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="py-10 text-center text-sm text-muted-foreground">{t("library.select_hint")}</p>
            )}
          </div>
        </div>
      }
      right={
        <div className="h-full overflow-y-auto">
          {docLoading ? (
            <p className="py-12 text-center text-sm text-muted-foreground">{t("library.doc_loading")}</p>
          ) : docContent ? (
            <div className="px-6 py-6">
              {selectedName && (
                <p className="mb-4 break-words border-b border-border pb-3 text-sm font-medium text-muted-foreground">
                  {selectedName}
                </p>
              )}
              <div className="mx-auto max-w-3xl">
                <MarkdownRenderer content={docContent} />
              </div>
            </div>
          ) : (
            <p className="flex h-full items-center justify-center px-6 text-center text-sm text-muted-foreground">
              {t("library.select_hint")}
            </p>
          )}
        </div>
      }
    />
  );
}

function CorpusDocCard({
  name,
  score,
  snippet,
  onClick,
}: {
  name: string;
  score: string | null;
  snippet?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="block w-full rounded-xl border border-border bg-card px-3.5 py-2.5 text-left transition-colors hover:border-primary/40"
    >
      <div className="flex items-start gap-2">
        <span className="min-w-0 flex-1 break-words text-sm font-medium">{name}</span>
        {score && (
          <span className="shrink-0 rounded-md bg-accent px-1.5 py-0.5 font-mono text-xs text-primary">
            {score}
          </span>
        )}
      </div>
      {snippet && (
        <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
          {snippet.trim().slice(0, 240)}
        </p>
      )}
    </button>
  );
}
