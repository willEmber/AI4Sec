"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getEventStreamUrl } from "@/lib/agent";
import type { AgentEvent, EventType } from "@/lib/agent";
import { applyToolEvent } from "@/lib/agentEvents";
import type { ToolActivity } from "@/lib/agentEvents";

const TERMINAL: EventType[] = ["run.completed", "run.failed", "run.cancelled"];

export type { ToolActivity };

interface UseAgentStreamReturn {
  /** Answer text assembled from message.delta events. */
  answer: string;
  tools: ToolActivity[];
  isStreaming: boolean;
  error: string | null;
  /** Stable code behind `error`, e.g. `interrupted` when a restart ended the turn. */
  errorCode: string;
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
  const [errorCode, setErrorCode] = useState("");
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
      case "tool.completed":
      case "tool.failed":
        // Same reducer the restored view uses, so a turn looks the same while
        // it runs and after it is reopened.
        setTools((prev) => applyToolEvent(prev, event));
        break;

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
        setErrorCode(String(payload.code || ""));
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
      setErrorCode("");
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
    errorCode,
    finishedRunId,
    evidenceIds,
    papersChanged,
    start,
    stop,
  };
}
