"use client";

import { useEffect, useState } from "react";
import type { ComponentType } from "react";
import { useTranslation } from "@/lib/i18n";
import { elapsedMs, formatDuration } from "@/lib/agentEvents";
import type { ToolActivity, ToolPreviewItem } from "@/lib/agentEvents";
import {
  IconBook,
  IconBookmark,
  IconChevronRight,
  IconDownload,
  IconGlobe,
  IconHistory,
  IconLens,
  IconListChecks,
  IconSearch,
  IconSnap,
  IconSphere,
  IconTool,
} from "@/components/icons";

type Translate = (key: string, vars?: Record<string, string | number>) => string;
type Icon = ComponentType<{ className?: string }>;

/** What kind of work a tool is, for its icon. Unknown tools fall back to a wrench. */
const TOOL_ICON: Record<string, Icon> = {
  get_paper_outline: IconBook,
  search_paper_content: IconBook,
  read_paper_section: IconBook,
  search_papers: IconSearch,
  resolve_paper: IconSearch,
  get_related_papers: IconSearch,
  get_paper_metadata: IconSearch,
  search_paper_snippets: IconSearch,
  get_peer_reviews: IconSearch,
  query_publication_rank: IconSearch,
  web_search: IconGlobe,
  read_web_page: IconGlobe,
  download_paper: IconDownload,
  ensure_paper_parsed: IconDownload,
  run_insight_snap: IconSnap,
  run_logic_lens: IconLens,
  run_research_sphere: IconSphere,
  save_memory: IconBookmark,
  list_memories: IconBookmark,
  forget_memory: IconBookmark,
  recall_conversations: IconHistory,
  open_project_paper: IconHistory,
  write_todos: IconListChecks,
};

/**
 * A tool's label in the reader's language. A tool the UI has no words for yet
 * shows its own name made readable, never the raw translation key.
 */
export function toolLabel(tool: string, t: Translate): string {
  const key = `chat.tool.${tool}`;
  const label = t(key);
  if (label !== key) return label;
  const words = tool.replace(/[_-]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : tool;
}

/**
 * What the agent did, one row per tool call.
 *
 * Shows the shape of the work, not its payload — a section read can be tens of
 * kilobytes, and the point here is that the reader can see the agent went and
 * looked, rather than answered from memory. That means what was asked for, as
 * soon as the call starts, and the first few things that came back.
 *
 * `previews` is whether those results start unfolded: yes for the work in
 * front of the reader, no for the steps a long turn has already moved past,
 * which would otherwise push the answer off the screen.
 */
export default function ToolActivityList({
  tools,
  previews = "open",
}: {
  tools: ToolActivity[];
  previews?: "open" | "closed";
}) {
  if (tools.length === 0) return null;
  return (
    <ol className="space-y-0.5">
      {tools.map((tool, index) => (
        <ToolRow key={`${tool.callId}-${index}`} tool={tool} previews={previews} />
      ))}
    </ol>
  );
}

/** Seconds since this mounted — how long a running call has been visible. */
function useSecondsShown(active: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!active) return;
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return seconds;
}

function ToolRow({ tool, previews }: { tool: ToolActivity; previews: "open" | "closed" }) {
  const { t } = useTranslation();
  const Icon = TOOL_ICON[tool.tool] ?? IconTool;
  const subject = subjectOf(tool);
  const counts = countsOf(tool, t);
  const filters = filtersOf(tool, t);
  const running = tool.status === "running";
  const waited = useSecondsShown(running);
  // The reader's own choice, once made, outlives the default changing as the
  // turn moves on.
  const [chosen, setChosen] = useState<boolean | null>(null);
  const preview = tool.preview ?? [];
  const previewOpen = chosen ?? previews === "open";
  const duration = running
    ? waited >= 3 ? formatDuration(waited * 1000) : ""
    : formatDuration(elapsedMs(tool.startedAt, tool.endedAt));
  const reused = Boolean(tool.summary?.reused || tool.summary?.cached);
  const failed = tool.status === "error" || tool.status === "unavailable";

  return (
    <li className="group flex items-start gap-2.5 rounded-lg px-2 py-1.5 text-xs transition-colors hover:bg-muted/60">
      <StatusIcon status={tool.status} Icon={Icon} />
      <div className="min-w-0 flex-1 leading-5">
        <div className="flex min-w-0 items-baseline gap-1.5">
          <span
            className={`shrink-0 font-medium ${
              tool.status === "running" ? "text-foreground" : "text-foreground/85"
            }`}
          >
            {toolLabel(tool.tool, t)}
          </span>
          {subject && (
            <span className="min-w-0 truncate text-muted-foreground" title={subject}>
              {subject}
            </span>
          )}
          <span className="ml-auto flex shrink-0 items-center gap-1.5 pl-2 text-[0.7rem] text-muted-foreground">
            {reused && (
              <span className="rounded bg-muted px-1 py-[1px]">{t("chat.tool.reused")}</span>
            )}
            {tool.status === "partial" && (
              <span className="rounded bg-accent px-1 py-[1px] text-accent-foreground">
                {t("chat.tool.status.partial")}
              </span>
            )}
            {failed && (
              <span
                className={`rounded px-1 py-[1px] ${
                  tool.status === "error" ? "bg-destructive/10 text-destructive" : "bg-muted"
                }`}
              >
                {t(`chat.tool.status.${tool.status}`)}
              </span>
            )}
            {duration && <span className="tabular-nums">{duration}</span>}
          </span>
        </div>

        {filters && <div className="truncate text-xs text-muted-foreground/90">{filters}</div>}

        {preview.length > 0 ? (
          <button
            type="button"
            onClick={() => setChosen(!previewOpen)}
            aria-expanded={previewOpen}
            className="-ml-1 flex max-w-full items-center gap-1 rounded px-1 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <IconChevronRight
              className={`shrink-0 text-[10px] transition-transform ${previewOpen ? "rotate-90" : ""}`}
            />
            <span className="min-w-0 truncate">{counts || t("chat.tool.preview")}</span>
          </button>
        ) : (
          counts && <div className="text-xs text-muted-foreground">{counts}</div>
        )}
        {preview.length > 0 && previewOpen && <PreviewList items={preview} />}

        {tool.error && (
          <div className="mt-0.5 line-clamp-2 text-xs text-destructive/90" title={tool.error.message}>
            {tool.error.message || tool.error.code}
          </div>
        )}
        {tool.note && !tool.steps?.length && !tool.error && (
          <div className="mt-0.5 line-clamp-2 text-xs text-muted-foreground/90" title={tool.note}>
            {tool.note}
          </div>
        )}
        {tool.steps && tool.steps.length > 0 && (
          <StepList steps={tool.steps} running={tool.status === "running"} />
        )}
      </div>
    </li>
  );
}

/** The first few results of a call: enough to see what the agent is working from. */
function PreviewList({ items }: { items: ToolPreviewItem[] }) {
  const { t } = useTranslation();
  return (
    <ul className="animate-fade-in mb-1 mt-1 space-y-1.5 border-l border-border/80 pl-2.5">
      {items.map((item, index) => (
        <li key={index} className="min-w-0 text-xs leading-[1.45]">
          <div className="flex min-w-0 items-baseline gap-1.5">
            {item.url ? (
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="min-w-0 truncate text-foreground/85 underline-offset-2 hover:underline"
                title={item.title || item.url}
              >
                {item.title || item.url}
              </a>
            ) : (
              item.title && (
                <span className="min-w-0 truncate text-foreground/85" title={item.title}>
                  {item.title}
                </span>
              )
            )}
            {(item.meta || item.page) && (
              <span className="shrink-0 text-[0.7rem] text-muted-foreground">
                {[item.meta, item.page ? t("chat.cite.page", { page: item.page }) : ""]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            )}
          </div>
          {item.text && <p className="line-clamp-2 text-muted-foreground">{item.text}</p>}
        </li>
      ))}
    </ul>
  );
}

function StatusIcon({ status, Icon }: { status: ToolActivity["status"]; Icon: Icon }) {
  const tone =
    status === "running"
      ? "border-primary/40 bg-accent text-primary"
      : status === "error"
        ? "border-destructive/30 bg-destructive/10 text-destructive"
        : status === "unavailable"
          ? "border-border bg-muted text-muted-foreground"
          : status === "partial"
            ? "border-border bg-accent text-accent-foreground"
            : "border-border bg-card text-muted-foreground";
  return (
    <span
      className={`relative mt-[1px] flex h-[1.15rem] w-[1.15rem] shrink-0 items-center justify-center rounded-md border ${tone}`}
    >
      <Icon className="text-[11px]" />
      {status === "running" && (
        <span className="absolute -inset-[3px] animate-spin rounded-[0.5rem] border-[1.5px] border-transparent border-t-primary" />
      )}
      {status === "ok" && (
        <span className="absolute -bottom-[3px] -right-[3px] h-[7px] w-[7px] rounded-full border border-card bg-success" />
      )}
    </span>
  );
}

/**
 * The steps a long tool went through — a mode report's pipeline stages, or a
 * parse. Labels are the pipeline's own, so the same stage reads the same here
 * and on the report page.
 */
function StepList({
  steps,
  running,
}: {
  steps: NonNullable<ToolActivity["steps"]>;
  running: boolean;
}) {
  const { t } = useTranslation();
  const done = steps.filter((s) => s.status === "done" || s.status === "skipped").length;
  return (
    <div className="mt-1.5 rounded-md border border-border/70 bg-background/60 px-2 py-1.5">
      <div className="mb-1 flex items-center gap-2">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary/70 transition-all duration-500"
            style={{ width: `${Math.round((done / steps.length) * 100)}%` }}
          />
        </div>
        <span className="text-[0.7rem] tabular-nums text-muted-foreground">
          {done}/{steps.length}
        </span>
      </div>
      <ul className="space-y-0.5">
        {steps.map((step, i) => {
          const finished = step.status === "done" || step.status === "skipped";
          const stopped = ["failed", "error", "waiting", "cancelled"].includes(step.status);
          const active = !finished && !stopped && running && i === steps.length - 1;
          const label = t(`step.${step.step}`);
          const phaseKey = step.status === "waiting" ? "chat.parse.waiting" : `chat.parse.phase.${step.phase}`;
          const phaseLabel = !finished && step.phase ? t(phaseKey) : "";
          const elapsed = step.elapsed_s == null ? "" : formatDuration(step.elapsed_s * 1000);
          return (
            <li
              key={step.step}
              className={`flex items-center gap-1.5 text-xs ${
                finished ? "text-muted-foreground" : "text-foreground"
              }`}
            >
              {finished ? (
                <span className="w-2.5 text-center text-success">✓</span>
              ) : active ? (
                <span className="inline-block h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
              ) : step.status === "failed" || step.status === "error" ? (
                <span className="w-2.5 text-center text-destructive">!</span>
              ) : (
                <span className="w-2.5 text-center opacity-50">·</span>
              )}
              <span className={step.status === "skipped" ? "line-through opacity-70" : ""}>
                {label === `step.${step.step}` ? step.step : label}
                {phaseLabel && phaseLabel !== phaseKey && (
                  <span className="ml-1.5 text-muted-foreground">{phaseLabel}</span>
                )}
              </span>
              <span className="ml-auto shrink-0 tabular-nums text-muted-foreground">
                {Boolean(step.total_pages) && t("chat.parse.pages", {
                  done: step.extracted_pages ?? 0, total: step.total_pages ?? 0,
                })}
                {elapsed && <span className="ml-1.5">{elapsed}</span>}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

const text = (v: unknown) => (typeof v === "string" || typeof v === "number" ? String(v).trim() : "");

/**
 * What the call was about. The result says it best (a section's real title,
 * the venue's canonical name); until there is one, the call's own arguments do.
 */
function subjectOf(tool: ToolActivity): string {
  const s = tool.summary || {};
  const i = tool.input || {};
  const quoted =
    text(s.section) || text(s.query) || text(s.question) || text(s.topic) ||
    text(i.query) || text(i.question) || text(i.section_title) || text(i.focus) || text(i.content);
  if (quoted) return `“${quoted}”`;
  const title = text(s.title) || text(s.venue) || text(i.title) || text(i.venue) || text(i.paper);
  if (title) return title;
  const domain = text(s.domain) || hostOf(text(s.url)) || hostOf(text(i.url));
  if (domain) return domain;
  return text(i.doi) || text(i.arxiv_id);
}

/** The constraints a search was given, so a narrow result reads as narrow on purpose. */
function filtersOf(tool: ToolActivity, t: Translate): string {
  const i = tool.input || {};
  const list = (v: unknown) =>
    Array.isArray(v) ? v.map(text).filter(Boolean).slice(0, 4).join(", ") : "";
  const bits: string[] = [];
  const from = text(i.year_from) || text(i.start_date);
  const to = text(i.year_to) || text(i.end_date);
  if (from || to) bits.push(`${from}–${to}`);
  for (const key of ["venues", "fields_of_study", "platforms", "include_domains"]) {
    const joined = list(i[key]);
    if (joined) bits.push(joined);
  }
  if (typeof i.min_citations === "number") {
    bits.push(t("chat.tool.min_citations", { count: i.min_citations }));
  }
  if (i.open_access_only === true) bits.push(t("chat.tool.open_access"));
  const sort = text(i.sort);
  if (sort && sort !== "relevance") {
    const key = `chat.tool.sort.${sort}`;
    const label = t(key);
    bits.push(label === key ? sort : label);
  }
  if (Array.isArray(i.paper_ids) && i.paper_ids.length > 1) {
    bits.push(t("chat.tool.n_papers", { count: i.paper_ids.length }));
  }
  return bits.join(" · ");
}

function hostOf(url: string): string {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, "") : "";
  } catch {
    return "";
  }
}

const COUNT_LABELS: Array<[string, string]> = [
  ["sections_count", "chat.tool.n_sections"],
  ["hits_count", "chat.tool.n_hits"],
  ["blocks_count", "chat.tool.n_blocks"],
  ["chunks_count", "chat.tool.n_chunks"],
  ["results_count", "chat.tool.n_results"],
  ["papers_count", "chat.tool.n_papers"],
  ["rankings_count", "chat.tool.n_rankings"],
  ["memories_count", "chat.tool.n_memories"],
  ["turns_count", "chat.tool.n_turns"],
  ["unknown_year_count", "chat.tool.n_unknown_year"],
  ["unknown_date_count", "chat.tool.n_unknown_date"],
];

function countsOf(tool: ToolActivity, t: Translate): string {
  const s = tool.summary || {};
  const bits: string[] = [];
  if (s.relation) {
    const key = `chat.relation.${String(s.relation)}`;
    const label = t(key);
    bits.push(label === key ? String(s.relation) : label);
  }
  for (const [key, label] of COUNT_LABELS) {
    const value = s[key];
    if (typeof value === "number" && value > 0) bits.push(t(label, { count: value }));
  }
  if (s.provider) bits.push(String(s.provider));
  if (tool.evidenceCount > 0) bits.push(t("chat.tool.n_evidence", { count: tool.evidenceCount }));
  return bits.join(" · ");
}
