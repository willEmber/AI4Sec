"use client";

import { useTranslation } from "@/lib/i18n";
import type { ToolActivity } from "@/hooks/useAgentStream";

/**
 * What the agent is doing right now.
 *
 * Shows the shape of the work, not its payload — a section read can be tens of
 * kilobytes, and the point here is that the reader can see the agent went and
 * looked, rather than answered from memory.
 */
export default function ToolActivityList({ tools }: { tools: ToolActivity[] }) {
  const { t } = useTranslation();
  if (tools.length === 0) return null;

  return (
    <div className="space-y-1.5 rounded-xl border border-border bg-muted/60 px-3 py-2.5">
      {tools.map((tool, index) => (
        <div
          key={`${tool.callId}-${index}`}
          className="flex items-start gap-2 text-xs text-muted-foreground"
        >
          <StatusMark status={tool.status} />
          <span className="min-w-0 leading-5">
            <span className="font-medium text-foreground">
              {t(`chat.tool.${tool.tool}`)}
            </span>
            {detail(tool, t) && <span className="ml-1.5">{detail(tool, t)}</span>}
            {tool.error && (
              <span className="ml-1.5 text-accent-foreground">{tool.error.code}</span>
            )}
            {tool.note && <span className="ml-1.5 opacity-75">{tool.note}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

function StatusMark({ status }: { status: ToolActivity["status"] }) {
  if (status === "running") {
    return (
      <span className="mt-1 inline-block h-3 w-3 shrink-0 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
    );
  }
  if (status === "ok") {
    return <span className="mt-0.5 shrink-0 text-success">✓</span>;
  }
  if (status === "partial") {
    return <span className="mt-0.5 shrink-0 text-accent-foreground">◐</span>;
  }
  return <span className="mt-0.5 shrink-0 text-destructive">!</span>;
}

type Translate = (key: string, vars?: Record<string, string | number>) => string;

const COUNT_LABELS: Array<[string, string]> = [
  ["sections_count", "chat.tool.n_sections"],
  ["hits_count", "chat.tool.n_hits"],
  ["blocks_count", "chat.tool.n_blocks"],
  ["results_count", "chat.tool.n_results"],
  ["papers_count", "chat.tool.n_papers"],
  ["rankings_count", "chat.tool.n_rankings"],
  ["unknown_year_count", "chat.tool.n_unknown_year"],
];

function detail(tool: ToolActivity, t: Translate): string {
  const summary = tool.summary || {};
  const bits: string[] = [];

  // What the call was about, in whichever field the tool reports it.
  const subject =
    summary.section ?? summary.query ?? summary.venue ?? summary.title ?? summary.question;
  if (subject) bits.push(`“${String(subject)}”`);
  if (summary.relation) bits.push(t(`chat.relation.${String(summary.relation)}`));

  for (const [key, label] of COUNT_LABELS) {
    const value = summary[key];
    if (typeof value === "number" && value > 0) bits.push(t(label, { count: value }));
  }

  if (tool.evidenceCount > 0) {
    bits.push(t("chat.tool.n_evidence", { count: tool.evidenceCount }));
  }
  return bits.join(" · ");
}
