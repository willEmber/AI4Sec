"use client";

import { useCallback, useState } from "react";
import { getRunActivity } from "@/lib/agent";
import type { AgentRun } from "@/lib/agent";
import { elapsedMs, foldToolActivity, formatDuration, sealActivity } from "@/lib/agentEvents";
import type { ToolActivity } from "@/lib/agentEvents";
import { useTranslation } from "@/lib/i18n";
import ToolActivityList from "@/components/agent/ToolActivityList";
import { IconChevronRight, IconTool } from "@/components/icons";

interface Props {
  runId: string;
  /** The run row, for its step count and duration before anything is fetched. */
  run?: AgentRun;
  /**
   * The activity, when this page already has it — the turn that just finished
   * live. Saves a request and shows the counts at once.
   */
  initialTools?: ToolActivity[];
}

/**
 * What an earlier turn did, folded into one line until asked for.
 *
 * Tool activity is streamed, so reopening a session used to show answers with
 * no record of the work behind them — which is exactly the part that makes an
 * answer checkable. The events are durable, so the record exists; this fetches
 * it when the reader opens it rather than firing one request per turn on load,
 * because a long session has a lot of turns and most are not being questioned.
 */
export default function PastTurnActivity({ runId, run, initialTools }: Props) {
  const { t } = useTranslation();
  const [tools, setTools] = useState<ToolActivity[] | null>(initialTools ?? null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const toggle = useCallback(async () => {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (tools !== null) return;
    setLoading(true);
    try {
      const activity = await getRunActivity(runId);
      // Sealed: a call still marked running in a replayed log never reported
      // back, and an endless spinner would misrepresent that as work in flight.
      setTools(sealActivity(foldToolActivity(activity.events)));
    } catch {
      setTools([]);
    } finally {
      setLoading(false);
    }
  }, [open, tools, runId]);

  const reportedCalls = typeof run?.usage?.tool_calls === "number" ? run.usage.tool_calls : null;
  // A turn that answered from what it already had did no visible work; a
  // "0 steps" box under it would only be noise.
  if (reportedCalls === 0 && !tools?.length) return null;

  const steps = tools?.length ?? reportedCalls;
  const evidence = tools?.reduce((n, tool) => n + tool.evidenceCount, 0) ?? 0;
  const failures = tools?.filter((tool) => tool.status === "error").length ?? 0;
  const duration = formatDuration(elapsedMs(run?.started_at, run?.finished_at));

  const bits = [
    steps != null ? t("chat.process.steps", { count: steps }) : t("chat.history.show"),
    evidence > 0 ? t("chat.process.evidence", { count: evidence }) : "",
    failures > 0 ? t("chat.process.failures", { count: failures }) : "",
    duration ? t("chat.process.took", { duration }) : "",
  ].filter(Boolean);

  return (
    <div className="rounded-xl border border-border/80 bg-card/50">
      <button
        type="button"
        onClick={() => void toggle()}
        aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
      >
        <IconTool className="shrink-0 text-[13px]" />
        <span className="min-w-0 flex-1 truncate">
          <span className="font-medium text-foreground/80">{t("chat.process.title")}</span>
          <span className="mx-1.5 opacity-40">·</span>
          {bits.join(" · ")}
        </span>
        <IconChevronRight
          className={`shrink-0 text-[13px] transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>
      {open && (
        <div className="animate-fade-in border-t border-border/70 px-1.5 py-1.5">
          {loading && (
            <p className="px-2 py-1 text-[0.7rem] text-muted-foreground">{t("chat.history.loading")}</p>
          )}
          {!loading && tools !== null && tools.length === 0 && (
            <p className="px-2 py-1 text-[0.7rem] text-muted-foreground">{t("chat.history.empty")}</p>
          )}
          {!loading && tools !== null && tools.length > 0 && <ToolActivityList tools={tools} />}
        </div>
      )}
    </div>
  );
}
