"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getEventStreamUrl } from "@/lib/agent";
import type { AgentEvent, EventType } from "@/lib/agent";

const TERMINAL: EventType[] = ["run.completed", "run.failed", "run.cancelled"];

export interface ToolActivity {
  callId: string;
  tool: string;
  status: "running" | "ok" | "partial" | "unavailable" | "error";
  evidenceCount: number;
  note?: string;
  error?: { code: string; message: string };
  summary?: Record<string, unknown>;
}

interface UseAgentStreamReturn {
  /** Answer text assembled from message.delta events. */
  answer: string;
  tools: ToolActivity[];
  isStreaming: boolean;
  error: string | null;
  finishedRunId: string | null;
  /** Evidence ids the tools produced this turn, in order. */
  evidenceIds: string[];
  /**
   * Increments when the agent attached a paper to the session mid-turn. The
   * agent can now go and fetch a paper the reader never uploaded, so the paper
   * list has to refresh during the turn rather than after it.
   */
  papersChanged: number;
  start: (runId: string) => void;
  stop: () => void;
}

/**
 * Subscribes to a run's event stream.
 *
 * Resumes rather than restarts: the last `seq` seen is kept, so a dropped
 * connection reconnects with `after=<seq>` and the backend replays the gap from
 * its durable log. Events are also de-duplicated by `seq`, because a replay
 * legitimately overlaps whatever arrived live just before the drop.
 */
export function useAgentStream(): UseAgentStreamReturn {
  const [answer, setAnswer] = useState("");
  const [tools, setTools] = useState<ToolActivity[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [finishedRunId, setFinishedRunId] = useState<string | null>(null);
  const [evidenceIds, setEvidenceIds] = useState<string[]>([]);
  const [papersChanged, setPapersChanged] = useState(0);

  const sourceRef = useRef<EventSource | null>(null);
  const seqRef = useRef(0);
  const runIdRef = useRef<string>("");
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const teardown = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
    if (retryRef.current) {
      clearTimeout(retryRef.current);
      retryRef.current = null;
    }
  }, []);

  const handle = useCallback((event: AgentEvent) => {
    if (event.seq <= seqRef.current) return; // replayed; already applied
    seqRef.current = event.seq;
    const payload = event.payload || {};

    switch (event.type) {
      case "message.delta":
        setAnswer((prev) => prev + String(payload.text || ""));
        break;

      case "tool.started":
        setTools((prev) => [
          ...prev,
          {
            callId: String(payload.call_id || `${payload.tool}-${event.seq}`),
            tool: String(payload.tool || ""),
            status: "running",
            evidenceCount: 0,
          },
        ]);
        break;

      case "tool.completed":
      case "tool.failed": {
        const toolName = String(payload.tool || "");
        setTools((prev) => {
          // Match the most recent still-running call of this tool: the backend
          // does not correlate a result back to its call id, and a tool can be
          // called more than once in a turn.
          const index = prev.findLastIndex(
            (t) => t.tool === toolName && t.status === "running",
          );
          const entry: ToolActivity = {
            callId: index >= 0 ? prev[index].callId : `${toolName}-${event.seq}`,
            tool: toolName,
            status: (payload.status as ToolActivity["status"]) || "ok",
            evidenceCount: Number(payload.evidence_count || 0),
            note: payload.note ? String(payload.note) : undefined,
            error: payload.error as ToolActivity["error"],
            summary: payload.summary as Record<string, unknown> | undefined,
          };
          if (index < 0) return [...prev, entry];
          const next = [...prev];
          next[index] = entry;
          return next;
        });
        break;
      }

      case "paper.added":
        setPapersChanged((n) => n + 1);
        break;

      case "run.completed":
        setEvidenceIds((payload.citations as string[]) || []);
        setIsStreaming(false);
        setFinishedRunId(event.run_id);
        break;

      case "run.failed":
        setError(String(payload.error || "The run failed."));
        setIsStreaming(false);
        setFinishedRunId(event.run_id);
        break;

      case "run.cancelled":
        setIsStreaming(false);
        setFinishedRunId(event.run_id);
        break;

      default:
        break;
    }
  }, []);

  const connect = useCallback(
    (runId: string) => {
      teardown();
      const source = new EventSource(getEventStreamUrl(runId, seqRef.current));
      sourceRef.current = source;

      const onEvent = (e: MessageEvent) => {
        try {
          handle(JSON.parse(e.data) as AgentEvent);
        } catch {
          // A malformed frame is not worth tearing the stream down for.
        }
      };
      // Named events, so `onmessage` alone would never fire.
      const types: EventType[] = [
        "run.started",
        "message.delta",
        "tool.started",
        "tool.completed",
        "tool.failed",
        "paper.added",
        ...TERMINAL,
      ];
      types.forEach((t) => source.addEventListener(t, onEvent as EventListener));

      source.onerror = () => {
        source.close();
        sourceRef.current = null;
        // The backend closes the stream once the run ends, which surfaces here
        // as an error too; only reconnect while the turn is still open.
        setIsStreaming((streaming) => {
          if (streaming && runIdRef.current === runId) {
            retryRef.current = setTimeout(() => connect(runId), 1500);
          }
          return streaming;
        });
      };
    },
    [handle, teardown],
  );

  const start = useCallback(
    (runId: string) => {
      teardown();
      seqRef.current = 0;
      runIdRef.current = runId;
      setAnswer("");
      setTools([]);
      setEvidenceIds([]);
      setError(null);
      setFinishedRunId(null);
      setIsStreaming(true);
      connect(runId);
    },
    [connect, teardown],
  );

  const stop = useCallback(() => {
    setIsStreaming(false);
    teardown();
  }, [teardown]);

  useEffect(() => teardown, [teardown]);

  return {
    answer,
    tools,
    isStreaming,
    error,
    finishedRunId,
    evidenceIds,
    papersChanged,
    start,
    stop,
  };
}
