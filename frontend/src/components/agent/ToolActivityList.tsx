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
          <span className="min-w-0 flex-1 leading-5">
            <span className="font-medium text-foreground">
              {t(`chat.tool.${tool.tool}`)}
            </span>
            {detail(tool, t) && <span className="ml-1.5">{detail(tool, t)}</span>}
            {tool.error && (
              <span className="ml-1.5 text-accent-foreground">{tool.error.code}</span>
            )}
            {tool.note && !tool.steps?.length && (
              <span className="ml-1.5 opacity-75">{tool.note}</span>
            )}
            {tool.steps && tool.steps.length > 0 && (
              <StepList steps={tool.steps} running={tool.status === "running"} />
            )}
          </span>
        </div>
      ))}
    </div>
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
  return (
    <span className="mt-1 block space-y-0.5">
      {steps.map((step, i) => {
        const done = step.status === "done" || step.status === "skipped";
        const active = !done && running && i === steps.length - 1;
        return (
          <span
            key={step.step}
            className={`flex items-center gap-1.5 text-[0.7rem] ${
              done ? "text-muted-foreground" : "text-foreground"
            }`}
          >
            {done ? (
              <span className="text-success">✓</span>
            ) : active ? (
              <span className="inline-block h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
            ) : (
              <span className="opacity-50">·</span>
            )}
            <span>{t(`step.${step.step}`) || step.step}</span>
          </span>
        );
      })}
    </span>
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
  ["turns_count", "chat.tool.n_turns"],
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
