"use client";

import { useCallback, useState } from "react";
import { getRunActivity } from "@/lib/agent";
import { foldToolActivity, sealActivity } from "@/lib/agentEvents";
import type { ToolActivity } from "@/lib/agentEvents";
import { useTranslation } from "@/lib/i18n";
import ToolActivityList from "@/components/agent/ToolActivityList";

/**
 * What an earlier turn read, restored on demand.
 *
 * Tool activity is streamed, so reopening a session used to show answers with
 * no record of the work behind them — which is exactly the part that makes an
 * answer checkable. The events are durable, so the record exists; this fetches
 * it when the reader asks for it rather than firing one request per turn on
 * load, because a long session has a lot of turns and most are not being
 * questioned.
 */
export default function PastTurnActivity({ runId }: { runId: string }) {
  const { t } = useTranslation();
  const [tools, setTools] = useState<ToolActivity[] | null>(null);
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

  return (
    <div className="mt-1.5">
      <button
        type="button"
        onClick={() => void toggle()}
        className="text-[0.7rem] text-muted-foreground transition-colors hover:text-foreground"
      >
        {open ? t("chat.history.hide") : t("chat.history.show")}
      </button>
      {open && (
        <div className="mt-1.5">
          {loading && (
            <p className="text-[0.7rem] text-muted-foreground">
              {t("chat.history.loading")}
            </p>
          )}
          {!loading && tools !== null && tools.length === 0 && (
            <p className="text-[0.7rem] text-muted-foreground">
              {t("chat.history.empty")}
            </p>
          )}
          {!loading && tools !== null && tools.length > 0 && (
            <ToolActivityList tools={tools} />
          )}
        </div>
      )}
    </div>
  );
}
