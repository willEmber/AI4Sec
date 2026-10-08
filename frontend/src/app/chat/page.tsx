"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { createSession, describeApiError, listProjects, postMessage } from "@/lib/agent";
import type { AgentMode, AgentProject } from "@/lib/agent";
import { listModels, listRecentRuns, uploadPaper } from "@/lib/api";
import type { RecentRunResponse } from "@/lib/types";
import { useTranslation } from "@/lib/i18n";
import { useChatShell } from "@/components/agent/ChatShell";
import Composer from "@/components/agent/Composer";
import { IconBook, IconMenu } from "@/components/icons";

const MODES: AgentMode[] = ["auto", "snap", "lens", "sphere"];
const RECENT_PAPERS = 6;

/**
 * The entry to the platform: a message box.
 *
 * Typing a question and sending it is what starts a conversation — the session
 * is created at that moment, not when the page opens, so looking at this page
 * leaves nothing behind. Attaching a PDF or picking a recent paper creates the
 * session straight away instead: a conversation with a paper in it is not
 * empty, and its parse starts while the reader is still typing.
 *
 * `?paper=<id>` (from a report page) skips the choice; `?mode=` preselects a
 * mode and `?project=` files the new session under that research project.
 *
 * Reading the query string opts the page out of prerendering, so it needs a
 * Suspense boundary of its own.
 */
export default function ChatEntryPage() {
  return (
    <Suspense fallback={<div className="h-full" />}>
      <ChatEntry />
    </Suspense>
  );
}

function ChatEntry() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const presetPaperId = searchParams.get("paper") || "";
  const presetMode = (searchParams.get("mode") || "auto") as AgentMode;
  const presetProjectId = searchParams.get("project") || "";
  const { t, locale } = useTranslation();
  const { refreshSessions, openNav, setCurrentProject } = useChatShell();

  const [projects, setProjects] = useState<AgentProject[]>([]);
  const [runs, setRuns] = useState<RecentRunResponse[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [llmModel, setLlmModel] = useState("");
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<AgentMode>(MODES.includes(presetMode) ? presetMode : "auto");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  // A session whose first message was refused (quota, network): the retry
  // posts into it instead of creating a second one.
  const createdRef = useRef("");

  useEffect(() => {
    listProjects()
      .then((data) => setProjects(data.projects))
      .catch(() => {});
    listRecentRuns(40, false)
      .then(setRuns)
      .catch(() => {});
    listModels()
      .then((res) => {
        // Conversations offer only the models verified against the agent's
        // tools, which can be fewer than the report modes list.
        const offered = res.agent_models ?? res.models;
        setModels(offered);
        setLlmModel(res.agent_default || offered[0] || "");
      })
      .catch(() => {});
  }, []);

  const presetProject = projects.find((p) => p.project_id === presetProjectId) ?? null;
  useEffect(() => {
    setCurrentProject(presetProject);
  }, [presetProject, setCurrentProject]);

  // One row per paper: the same PDF is usually analysed several times, and the
  // agent reads the paper, not the run.
  const papers = useMemo(() => {
    const seen = new Map<string, RecentRunResponse>();
    for (const run of runs) {
      if (run.paper_id && !seen.has(run.paper_id)) seen.set(run.paper_id, run);
    }
    return [...seen.values()].slice(0, RECENT_PAPERS);
  }, [runs]);

  const create = useCallback(
    (paperIds: string[], title: string) =>
      createSession({
        title,
        language: locale === "zh" ? "zh" : "en",
        llm_model: llmModel,
        paper_ids: paperIds,
        project_id: presetProjectId,
      }),
    [locale, llmModel, presetProjectId],
  );

  /** Open a conversation on a paper; the mode chosen here travels with it. */
  const openOnPaper = useCallback(
    async (paperId: string, title: string, key: string) => {
      if (busy) return;
      setBusy(key);
      setError("");
      try {
        const session = await create([paperId], title);
        refreshSessions();
        router.push(`/chat/${session.session_id}${mode !== "auto" ? `?mode=${mode}` : ""}`);
      } catch (err) {
        setError(describeApiError(err, t));
        setBusy("");
      }
    },
    [busy, create, mode, refreshSessions, router, t],
  );

  const send = useCallback(async () => {
    const content = draft.trim();
    if (!content || busy) return;
    setBusy("send");
    setError("");
    try {
      if (!createdRef.current) createdRef.current = (await create([], "")).session_id;
      await postMessage(createdRef.current, {
        content,
        clientRequestId: crypto.randomUUID(),
        mode,
      });
      refreshSessions();
      // The conversation page attaches to the run that is now in flight.
      router.push(`/chat/${createdRef.current}`);
    } catch (err) {
      setError(describeApiError(err, t));
      setBusy("");
    }
  }, [draft, busy, create, mode, refreshSessions, router, t]);

  const attach = useCallback(
    async (file: File) => {
      const uploaded = await uploadPaper(file);
      await openOnPaper(uploaded.paper_id, file.name.replace(/\.pdf$/i, ""), "upload");
    },
    [openOnPaper],
  );

  // Arriving with ?paper=… means the choice is already made.
  useEffect(() => {
    if (presetPaperId) void openOnPaper(presetPaperId, "", presetPaperId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetPaperId]);

  return (
    <div
      className="relative flex h-full flex-col"
      onDragOver={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          setError(t("upload.drop_error"));
          return;
        }
        attach(file).catch((err) => setError(describeApiError(err, t)));
      }}
    >
      <div className="flex h-12 shrink-0 items-center px-3 lg:hidden">
        <button
          type="button"
          onClick={openNav}
          title={t("chat.nav.open")}
          aria-label={t("chat.nav.open")}
          className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <IconMenu className="text-[18px]" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-4 sm:px-6">
        {/* Sits a little above centre: the eye lands on the box, not the middle of the page. */}
        <div className="w-full max-w-2xl pb-10 pt-[16vh]">
          <h1 className="font-display text-center text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
            {t("chat.entry.greeting")}
          </h1>
          <p className="mx-auto mt-3 max-w-md text-center text-sm leading-relaxed text-muted-foreground">
            {t("chat.entry.greeting_sub")}
          </p>

          <div className="mt-8">
            {presetProject && (
              <p className="mb-2 rounded-lg bg-accent px-3 py-2 text-xs text-accent-foreground">
                {t("chat.entry.in_project", { title: presetProject.title || t("project.untitled") })}
              </p>
            )}
            {error && (
              <p className="mb-2 rounded-lg border border-destructive/25 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                {error}
              </p>
            )}
            <Composer
              value={draft}
              onChange={setDraft}
              mode={mode}
              onModeChange={setMode}
              onSend={() => void send()}
              sending={Boolean(busy)}
              onAttach={attach}
              model={llmModel}
              models={models}
              onModelChange={setLlmModel}
              autoFocus
            />
          </div>

          {papers.length > 0 && (
            <section className="mt-8">
              <h2 className="mb-2 px-1 text-xs text-muted-foreground">{t("chat.entry.recent_papers")}</h2>
              <div className="grid gap-2 sm:grid-cols-2">
                {papers.map((paper) => (
                  <button
                    key={paper.paper_id}
                    type="button"
                    disabled={Boolean(busy)}
                    onClick={() => void openOnPaper(paper.paper_id, paper.paper_title, paper.paper_id)}
                    className="flex items-start gap-2.5 rounded-xl border border-border bg-card px-3 py-2.5 text-left transition-colors hover:border-foreground/20 disabled:opacity-60"
                  >
                    {busy === paper.paper_id ? (
                      <span className="mt-0.5 inline-block h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
                    ) : (
                      <IconBook className="mt-0.5 shrink-0 text-[14px] text-muted-foreground" />
                    )}
                    <span className="line-clamp-2 min-w-0 text-[0.8125rem] leading-snug text-foreground/90">
                      {paper.paper_title || paper.paper_id.slice(0, 16)}
                    </span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>

      {dragOver && <DropOverlay label={t("chat.entry.drop")} />}
    </div>
  );
}

function DropOverlay({ label }: { label: string }) {
  return (
    <div className="pointer-events-none absolute inset-3 z-20 flex items-center justify-center rounded-2xl border-2 border-dashed border-primary bg-accent/80 text-sm font-medium text-accent-foreground">
      {label}
    </div>
  );
}
