"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getRunStreamUrl } from "@/lib/api";
import type { SSEEvent } from "@/lib/types";

interface UseRunStreamReturn {
  events: SSEEvent[];
  isConnected: boolean;
  isDone: boolean;
  error: string | null;
  connect: (runId: string) => void;
}

export function useRunStream(): UseRunStreamReturn {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const connectTimeRef = useRef<number>(0);

  const connect = useCallback((runId: string) => {
    // Close existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setEvents([]);
    setIsDone(false);
    setError(null);
    setIsConnected(true);
    connectTimeRef.current = performance.now();

    const url = getRunStreamUrl(runId);
    console.log(`[SSE] Connecting to ${url}`);
    const source = new EventSource(url);
    eventSourceRef.current = source;

    source.onopen = () => {
      console.log(`[SSE] Connected (${(performance.now() - connectTimeRef.current).toFixed(0)}ms)`);
    };

    source.onmessage = (e) => {
      const elapsed = ((performance.now() - connectTimeRef.current) / 1000).toFixed(1);
      try {
        const parsed: SSEEvent = JSON.parse(e.data);
        console.log(`[SSE +${elapsed}s] ${parsed.event}`, parsed.data);
        setEvents((prev) => [...prev, parsed]);

        if (parsed.event === "done" || parsed.event === "end") {
          console.log(`[SSE] Stream completed at +${elapsed}s`);
          setIsDone(true);
          setIsConnected(false);
          source.close();
        } else if (parsed.event === "error") {
          console.error(`[SSE] Error at +${elapsed}s:`, parsed.data?.error);
          setError(String(parsed.data?.error || "Unknown error"));
          setIsConnected(false);
          source.close();
        } else if (parsed.event === "timeout") {
          console.warn(`[SSE] Timeout at +${elapsed}s — stream disconnected, polling will continue`);
          setIsConnected(false);
          source.close();
        }
      } catch {
        console.warn(`[SSE +${elapsed}s] Failed to parse:`, e.data);
      }
    };

    source.onerror = (e) => {
      const elapsed = ((performance.now() - connectTimeRef.current) / 1000).toFixed(1);
      console.error(`[SSE +${elapsed}s] Connection error`, e);
      setIsConnected(false);
      source.close();
    };
  }, []);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  return { events, isConnected, isDone, error, connect };
}
