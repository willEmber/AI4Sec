"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getRun, getRunOutput, getPaperPdfUrl, getPaper } from "@/lib/api";
import { useRunStream } from "@/hooks/useRunStream";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import PdfViewer from "@/components/PdfViewer";
import SplitPane from "@/components/SplitPane";
import type { RunResponse, PaperResponse, SSEEvent } from "@/lib/types";

const STEP_LABELS: Record<string, string> = {
  ingest_pdf: "Verifying PDF",
  mineru_parse: "Parsing with MinerU",
  build_paper_ir: "Building document structure",
  run_snap: "Generating Insight Snap",
  run_lens: "Generating Logic Lens",
  run_sphere: "Generating Research Sphere",
  persist_output: "Saving results",
  // Sphere pipeline steps
  sphere_init_from_pdf: "Extracting references from PDF",
  extract_core_metadata: "Extracting core metadata",
  resolve_canonical_ids: "Resolving paper identifiers",
  expand_graph_candidates: "Expanding citation graph",
  dedup_and_score: "Scoring and selecting candidates",
  layer0_summarize_metadata: "Summarizing metadata",
  layer1_abstract_snap: "Analyzing abstracts",
  download_and_mineru_parse: "Downloading and parsing papers",
  synthesize_landscape: "Synthesizing research landscape",
  render_output: "Rendering final output",
};

export default function RunPage() {
  const params = useParams();
  const paperId = params.paperId as string;
  const runId = params.runId as string;

  const [run, setRun] = useState<RunResponse | null>(null);
  const [paper, setPaper] = useState<PaperResponse | null>(null);
  const [markdown, setMarkdown] = useState<string>("");
  const [targetPage, setTargetPage] = useState<number | undefined>(undefined);
  const [pageLoadTime] = useState(() => performance.now());

  const { events, isConnected, isDone, error, connect } = useRunStream();

  // Load paper info
  useEffect(() => {
    const t0 = performance.now();
    getPaper(paperId).then((p) => {
      console.log(`[RunPage] getPaper: ${(performance.now() - t0).toFixed(0)}ms`);
      setPaper(p);
    }).catch(() => {});
  }, [paperId]);

  // Immediate check: if run is already completed (e.g. page refresh)
  useEffect(() => {
    const t0 = performance.now();
    getRun(runId).then((r) => {
      console.log(`[RunPage] Initial getRun: ${(performance.now() - t0).toFixed(0)}ms status=${r.status}`);
      setRun(r);
      if (r.status === "done") {
        getRunOutput(runId).then((o) => {
          console.log(`[RunPage] Initial getRunOutput: ${o.markdown.length} chars`);
          setMarkdown(o.markdown);
        }).catch(() => {});
      }
    }).catch(() => {});
  }, [runId]);

  // Connect to SSE stream (direct to backend, bypasses Next.js proxy)
  useEffect(() => {
    console.log(`[RunPage] Mounting, connecting SSE for run=${runId} (page loaded ${(performance.now() - pageLoadTime).toFixed(0)}ms ago)`);
    connect(runId);
  }, [runId, connect, pageLoadTime]);

  // When SSE reports done, fetch output immediately
  useEffect(() => {
    if (!isDone) return;
    const t0 = performance.now();
    getRun(runId).then((r) => {
      console.log(`[RunPage] SSE done → getRun: ${(performance.now() - t0).toFixed(0)}ms status=${r.status}`);
      setRun(r);
    });
    getRunOutput(runId).then((o) => {
      console.log(`[RunPage] SSE done → getRunOutput: ${(performance.now() - t0).toFixed(0)}ms markdown=${o.markdown.length} chars`);
      console.log(`[RunPage] Total time from page load to output: ${((performance.now() - pageLoadTime) / 1000).toFixed(1)}s`);
      setMarkdown(o.markdown);
    }).catch(() => {});
  }, [isDone, runId, pageLoadTime]);

  // Always-on backup polling — runs regardless of SSE state until we have results
  useEffect(() => {
    if (markdown) return; // Already got results, stop polling
    if (run?.status === "failed") return; // Run failed, stop polling

    const interval = setInterval(() => {
      getRun(runId).then((r) => {
        setRun(r);
        if (r.status === "done") {
          console.log(`[RunPage] Polling detected completion, fetching output...`);
          getRunOutput(runId).then((o) => {
            console.log(`[RunPage] Poll → getRunOutput: markdown=${o.markdown.length} chars`);
            setMarkdown(o.markdown);
          }).catch(() => {});
        }
      }).catch(() => {});
    }, 3000);

    return () => clearInterval(interval);
  }, [runId, markdown, run?.status]);

  const handleCitationClick = useCallback((page: number) => {
    setTargetPage(page);
  }, []);

  // Progress steps from events — deduplicate by step name, keeping latest status
  const progressSteps = (() => {
    const all = events
      .filter((e: SSEEvent) => e.event === "progress")
      .map((e: SSEEvent) => e.data as { step: string; status: string });
    const map = new Map<string, { step: string; status: string }>();
    for (const s of all) {
      map.set(s.step, s);
    }
    return Array.from(map.values());
  })();

  const currentStep = progressSteps.length > 0
    ? progressSteps[progressSteps.length - 1]
    : null;

  const isComplete = run?.status === "done" || (isDone && markdown);
  const isFailed = (run?.status === "failed" || !!error) && !isComplete;
  const isRunning = (run?.status === "running" || isConnected) && !isComplete && !isFailed;

  return (
    <div className="h-[calc(100vh-49px)] flex flex-col">
      {/* Header bar */}
      <div className="border-b border-[var(--border)] px-4 py-2 flex items-center gap-4 shrink-0">
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate">
            {paper?.title || paperId}
          </p>
          <p className="text-xs text-[var(--muted-foreground)]">
            Mode: {run?.mode || "..."} | Run: {runId}
          </p>
        </div>
        <div className="shrink-0">
          {isRunning && (
            <span className="inline-flex items-center gap-2 text-sm text-[var(--primary)]">
              <span className="w-2 h-2 bg-[var(--primary)] rounded-full animate-pulse" />
              {currentStep ? STEP_LABELS[currentStep.step] || currentStep.step : "Running..."}
            </span>
          )}
          {isComplete && (
            <span className="text-sm text-green-600">Complete</span>
          )}
          {isFailed && (
            <span className="text-sm text-[var(--destructive)]">
              Failed: {error || run?.error_msg || "Unknown error"}
            </span>
          )}
        </div>
      </div>

      {/* Main content */}
      {isRunning && !markdown ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-10 h-10 border-4 border-[var(--primary)] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-[var(--muted-foreground)]">
              {currentStep ? STEP_LABELS[currentStep.step] || currentStep.step : "Starting analysis..."}
            </p>
            {/* Progress steps */}
            <div className="mt-6 text-left max-w-xs mx-auto">
              {progressSteps.map((step, i) => (
                <div key={i} className="flex items-center gap-2 text-sm py-1">
                  {step.status === "done" ? (
                    <span className="text-green-600">✓</span>
                  ) : (
                    <span className="w-3.5 h-3.5 border-2 border-[var(--primary)] border-t-transparent rounded-full animate-spin inline-block" />
                  )}
                  <span>{STEP_LABELS[step.step] || step.step}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : isFailed && !markdown ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <p className="text-[var(--destructive)] font-medium mb-2">Analysis Failed</p>
            <p className="text-sm text-[var(--muted-foreground)]">
              {error || run?.error_msg || "An unknown error occurred"}
            </p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-hidden">
          <SplitPane
            left={
              <div className="p-6 max-w-none">
                <MarkdownRenderer content={markdown} onCitationClick={handleCitationClick} />
              </div>
            }
            right={
              <PdfViewer
                url={getPaperPdfUrl(paperId)}
                targetPage={targetPage}
              />
            }
          />
        </div>
      )}
    </div>
  );
}
