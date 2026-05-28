"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getRun, getRunOutput, getPaperPdfUrl, getPaper } from "@/lib/api";
import { useRunStream } from "@/hooks/useRunStream";
import { useTranslation } from "@/lib/i18n";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import PdfViewer from "@/components/PdfViewer";
import SplitPane from "@/components/SplitPane";
import RankBadges from "@/components/RankBadges";
import { IconDownload, IconCheck } from "@/components/icons";
import type { RunResponse, PaperResponse, SSEEvent, ProgressEntry } from "@/lib/types";

export default function RunPage() {
  const params = useParams();
  const paperId = params.paperId as string;
  const runId = params.runId as string;
  const { t } = useTranslation();

  const [run, setRun] = useState<RunResponse | null>(null);
  const [paper, setPaper] = useState<PaperResponse | null>(null);
  const [markdown, setMarkdown] = useState<string>("");
  const [targetPage, setTargetPage] = useState<number | undefined>(undefined);
  const [pageLoadTime] = useState(() => performance.now());

  const { events, isConnected, isDone, error, connect } = useRunStream();

  const stepLabel = useCallback(
    (step: string) => t(`step.${step}`) || step,
    [t],
  );

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
      console.log(`[RunPage] SSE done -> getRun: ${(performance.now() - t0).toFixed(0)}ms status=${r.status}`);
      setRun(r);
    });
    getRunOutput(runId).then((o) => {
      console.log(`[RunPage] SSE done -> getRunOutput: ${(performance.now() - t0).toFixed(0)}ms markdown=${o.markdown.length} chars`);
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
            console.log(`[RunPage] Poll -> getRunOutput: markdown=${o.markdown.length} chars`);
            setMarkdown(o.markdown);
          }).catch(() => {});
        }
      }).catch(() => {});
      // Re-fetch paper to pick up venue/rank data from enrich_metadata
      if (!paper?.venue || (!paper?.sci_rank && !paper?.ccf_rank)) {
        getPaper(paperId).then((p) => setPaper(p)).catch(() => {});
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [runId, paperId, markdown, run?.status, paper?.venue]);

  const handleCitationClick = useCallback((page: number) => {
    setTargetPage(page);
  }, []);

  const handleExportMarkdown = useCallback(() => {
    if (!markdown) return;
    const title = paper?.title || paperId;
    const mode = run?.mode || "analysis";
    const filename = `${title.replace(/[^a-zA-Z0-9一-鿿]+/g, "_").slice(0, 60)}_${mode}.md`;
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, [markdown, paper?.title, paperId, run?.mode]);

  // Progress steps merged from two sources:
  //   1. persisted `run.progress_json` (full history, even if SSE wasn't connected)
  //   2. live SSE events received this session
  // Deduplicate by step name, keeping the latest status per step.
  const progressSteps = (() => {
    const persisted: ProgressEntry[] = (() => {
      if (!run?.progress_json) return [];
      try {
        const parsed = JSON.parse(run.progress_json);
        return Array.isArray(parsed) ? (parsed as ProgressEntry[]) : [];
      } catch {
        return [];
      }
    })();

    const live = events
      .filter((e: SSEEvent) => e.event === "progress")
      .map((e: SSEEvent) => e.data as ProgressEntry);

    const map = new Map<string, ProgressEntry>();
    for (const s of [...persisted, ...live]) {
      if (s && typeof s.step === "string") map.set(s.step, s);
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
    <div className="flex h-[calc(100vh-3.5rem)] flex-col">
      {/* Header bar */}
      <div className="flex shrink-0 items-center gap-4 border-b border-border bg-card/60 px-5 py-2.5">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">
            {paper?.title || paperId}
          </p>
          <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
            <span className="truncate">
              {t("run.mode")}: <span className="font-medium text-foreground/80">{run?.mode || "..."}</span>
              <span className="mx-1.5 opacity-40">·</span>
              {runId}
            </span>
            {paper && <RankBadges venue={paper.venue} year={paper.year} sciRank={paper.sci_rank} ccfRank={paper.ccf_rank} />}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          {markdown && (
            <button
              onClick={handleExportMarkdown}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
              title={t("run.export_md")}
            >
              <IconDownload className="text-[15px]" />
              {t("run.export_md")}
            </button>
          )}
          {isRunning && (
            <span className="inline-flex items-center gap-2 text-sm font-medium text-primary">
              <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />
              {currentStep ? stepLabel(currentStep.step) : t("run.status.running")}
            </span>
          )}
          {isComplete && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-success/10 px-2.5 py-1 text-sm font-medium text-success">
              <IconCheck className="text-[15px]" />
              {t("run.status.complete")}
            </span>
          )}
          {isFailed && (
            <span className="text-sm text-destructive">
              {error || run?.error_msg || t("run.status.unknown")}
            </span>
          )}
        </div>
      </div>

      {/* Smart Q&A question + detected intent banner */}
      {run?.user_question && (
        <div className="shrink-0 border-b border-border bg-accent/40 px-5 py-2.5">
          <p className="text-xs">
            <span className="text-muted-foreground">{t("run.your_question")}</span>{" "}
            <span className="font-medium">{run.user_question}</span>
          </p>
          {run.detected_intent && (
            <p className="mt-0.5 text-xs">
              <span className="text-muted-foreground">{t("run.detected_intent")}</span>{" "}
              <span className="font-medium text-primary">{t(`intent.${run.detected_intent}`)}</span>
            </p>
          )}
        </div>
      )}

      {/* Main content — split pane is always mounted so PDF loads immediately,
          while the left side shows progress / failure / markdown as state evolves. */}
      <div className="flex-1 overflow-hidden">
        <SplitPane
          left={
            markdown ? (
              <div className="px-6 py-8 sm:px-10">
                <div className="mx-auto max-w-3xl">
                  <MarkdownRenderer content={markdown} onCitationClick={handleCitationClick} />
                </div>
              </div>
            ) : isFailed ? (
              <div className="flex h-full items-center justify-center px-6">
                <div className="max-w-md rounded-2xl border border-destructive/25 bg-destructive/5 p-8 text-center">
                  <p className="mb-2 font-semibold text-destructive">{t("run.status.failed_label")}</p>
                  <p className="text-sm text-muted-foreground">
                    {error || run?.error_msg || t("run.status.unknown")}
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex h-full items-start justify-center overflow-auto px-6 py-10">
                <div className="w-full max-w-sm text-center">
                  <div className="mx-auto mb-5 h-10 w-10 animate-spin rounded-full border-[3px] border-primary border-t-transparent" />
                  <p className="font-medium text-foreground">
                    {currentStep ? stepLabel(currentStep.step) : t("run.starting")}
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {t("run.pdf_ready_hint")}
                  </p>
                  {progressSteps.length > 0 && (
                    <div className="mx-auto mt-7 max-w-xs space-y-0.5 text-left">
                      {progressSteps.map((step, i) => {
                        const done = step.status === "done" || step.status === "skipped";
                        return (
                          <div
                            key={i}
                            className={`flex items-center gap-2.5 rounded-lg px-3 py-1.5 text-sm ${
                              done ? "text-muted-foreground" : "bg-accent/50 font-medium"
                            }`}
                          >
                            {done ? (
                              <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-success/20 text-[11px] text-success">
                                <IconCheck />
                              </span>
                            ) : (
                              <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                            )}
                            <span>{stepLabel(step.step)}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )
          }
          right={
            <PdfViewer
              url={getPaperPdfUrl(paperId)}
              targetPage={targetPage}
            />
          }
        />
      </div>
    </div>
  );
}
