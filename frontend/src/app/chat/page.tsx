"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  attachPapers,
  createSession,
  describeApiError,
  listOwnerPapers,
  postMessage,
} from "@/lib/agent";
import type { AgentMode, SessionPaper } from "@/lib/agent";
import { listModels, listRecentRuns, uploadPaper } from "@/lib/api";
import type { RecentRunResponse } from "@/lib/types";
import { mergeLibraryItems, resolveTurnPrompt } from "@/lib/libraryHub";
import { useTranslation } from "@/lib/i18n";
import { useChatShell } from "@/components/agent/ChatShell";
import Composer from "@/components/agent/Composer";
import { IconBook, IconMenu, IconSearch } from "@/components/icons";

const MODES: AgentMode[] = ["auto", "snap", "lens", "sphere"];
const RECENT_PAPERS = 6;

function makeStagedPaper(paperId: string, title: string): SessionPaper {
  return {
    session_id: "",
    literature_id: paperId,
    paper_id: paperId,
    availability: "pdf_ready",
    added_by: "user",
    note: "",
    title,
    year: 0,
    year_known: false,
    venue: "",
    doi: "",
    arxiv_id: "",
    updated_at: new Date().toISOString(),
  };
}

/**
 * Primary workspace entry (`/chat`).
 *
 * Readers can type a question directly, stage one or more PDFs without losing
 * their draft text, assign a research project inline, or launch any of the
 * three whole-paper report modes in a single click.
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
  const presetPaperParam = searchParams.get("paper") || "";
  const presetPapersParam = searchParams.get("papers") || "";
  const presetMode = (searchParams.get("mode") || "auto") as AgentMode;
  const presetProjectParam = searchParams.get("project") || "";

  const { t, locale } = useTranslation();
  const { projects, refreshSessions, refreshRuns, openNav, openSearch, setCurrentProject } =
    useChatShell();

  const [ownerPapers, setOwnerPapers] = useState<SessionPaper[]>([]);
  const [runs, setRuns] = useState<RecentRunResponse[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [llmModel, setLlmModel] = useState("");
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<AgentMode>(MODES.includes(presetMode) ? presetMode : "auto");
  const [projectId, setProjectId] = useState<string>(presetProjectParam);
  const [stagedPapers, setStagedPapers] = useState<SessionPaper[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const createdRef = useRef("");

  useEffect(() => {
    setProjectId(presetProjectParam);
  }, [presetProjectParam]);

  useEffect(() => {
    if (MODES.includes(presetMode)) setMode(presetMode);
  }, [presetMode]);

  useEffect(() => {
    listOwnerPapers(30)
      .then((res) => setOwnerPapers(res.papers))
      .catch(() => {});
    listRecentRuns(30, false, 0)
      .then(setRuns)
      .catch(() => {});
    listModels()
      .then((res) => {
        const offered = res.agent_models ?? res.models;
        setModels(offered);
        setLlmModel(res.agent_default || offered[0] || "");
      })
      .catch(() => {});
  }, []);

  const selectedProject = projects.find((p) => p.project_id === projectId) ?? null;
  useEffect(() => {
    setCurrentProject(selectedProject);
  }, [selectedProject, setCurrentProject]);

  const recentPapers = useMemo(() => {
    return mergeLibraryItems(ownerPapers, runs)
      .filter((item) => Boolean(item.paperId))
      .slice(0, RECENT_PAPERS);
  }, [ownerPapers, runs]);

  const create = useCallback(
    (paperIds: string[], title: string) =>
      createSession({
        title,
        language: locale === "zh" ? "zh" : "en",
        llm_model: llmModel,
        paper_ids: paperIds,
        project_id: projectId,
      }),
    [locale, llmModel, projectId],
  );

  const openOnPapers = useCallback(
    async (paperIds: string[], title: string, key: string) => {
      if (busy || paperIds.length === 0) return;
      setBusy(key);
      setError("");
      try {
        const session = await create(paperIds, title);
        refreshSessions();
        router.push(`/chat/${session.session_id}${mode !== "auto" ? `?mode=${mode}` : ""}`);
      } catch (err) {
        setError(describeApiError(err, t));
        setBusy("");
      }
    },
    [busy, create, mode, refreshSessions, router, t],
  );

  const submitTurn = useCallback(
    async (chosenMode: AgentMode) => {
      const content = resolveTurnPrompt(draft, chosenMode, (m) =>
        t(`chat.mode.${m}.default_prompt`),
      );
      if (!content || busy) return;
      if (chosenMode !== "auto" && !draft.trim() && stagedPapers.length === 0) return;

      setBusy("send");
      setError("");
      try {
        const paperIds = stagedPapers.map((p) => p.paper_id).filter(Boolean);
        const initialTitle =
          stagedPapers.length === 1 && !draft.trim() ? stagedPapers[0].title : "";
        if (!createdRef.current) {
          createdRef.current = (await create(paperIds, initialTitle)).session_id;
        } else if (paperIds.length > 0) {
          await attachPapers(createdRef.current, paperIds);
        }
        await postMessage(createdRef.current, {
          content,
          clientRequestId: crypto.randomUUID(),
          mode: chosenMode,
        });
        refreshSessions();
        if (chosenMode !== "auto") refreshRuns();
        router.push(`/chat/${createdRef.current}`);
      } catch (err) {
        setError(describeApiError(err, t));
        setBusy("");
      }
    },
    [draft, busy, stagedPapers, create, refreshSessions, refreshRuns, router, t],
  );

  const stageFile = useCallback(async (file: File) => {
    const uploaded = await uploadPaper(file);
    const cleanTitle = file.name.replace(/\.pdf$/i, "");
    setStagedPapers((prev) =>
      prev.some((p) => p.paper_id === uploaded.paper_id)
        ? prev
        : [...prev, makeStagedPaper(uploaded.paper_id, cleanTitle)],
    );
  }, []);

  const removeStaged = useCallback((paperId: string) => {
    setStagedPapers((prev) => prev.filter((p) => p.paper_id !== paperId));
  }, []);

  const handlePickRecent = useCallback(
    (paperId: string, title: string) => {
      // Preserve any typed draft or staged multi-paper set instead of navigating away.
      if (draft.trim() || stagedPapers.length > 0) {
        setStagedPapers((prev) =>
          prev.some((p) => p.paper_id === paperId)
            ? prev
            : [...prev, makeStagedPaper(paperId, title)],
        );
        return;
      }
      void openOnPapers([paperId], title, paperId);
    },
    [draft, stagedPapers.length, openOnPapers],
  );

  // Arriving with `?paper=...` or `?papers=id1,id2` opens directly on those papers.
  const presetHandledRef = useRef(false);
  useEffect(() => {
    if (presetHandledRef.current) return;
    const ids = [
      ...(presetPaperParam ? [presetPaperParam] : []),
      ...presetPapersParam
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    ];
    const unique = [...new Set(ids)];
    if (unique.length === 0) return;
    presetHandledRef.current = true;
    void openOnPapers(unique, "", unique[0]);
  }, [presetPaperParam, presetPapersParam, openOnPapers]);

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
        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;
        if (files.some((f) => !f.name.toLowerCase().endsWith(".pdf"))) {
          setError(t("upload.drop_error"));
          return;
        }
        Promise.all(files.map((f) => stageFile(f))).catch((err) =>
          setError(describeApiError(err, t)),
        );
      }}
    >
      <div className="flex h-12 shrink-0 items-center justify-between px-3 lg:hidden">
        <button
          type="button"
          onClick={openNav}
          title={t("chat.nav.open")}
          aria-label={t("chat.nav.open")}
          className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <IconMenu className="text-[18px]" />
        </button>
        <button
          type="button"
          onClick={openSearch}
          title={t("search.trigger")}
          aria-label={t("search.trigger")}
          className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <IconSearch className="text-[16px]" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-4 sm:px-6">
        <div className="w-full max-w-2xl pb-12 pt-[14vh] sm:pt-[16vh]">
          <h1 className="font-display text-center text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
            {t("chat.entry.greeting")}
          </h1>
          <p className="mx-auto mt-3 max-w-md text-center text-sm leading-relaxed text-muted-foreground">
            {t("chat.entry.greeting_sub")}
          </p>

          <div className="mt-8">
            {error && (
              <p className="mb-2 rounded-xl border border-destructive/25 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                {error}
              </p>
            )}
            <Composer
              value={draft}
              onChange={setDraft}
              mode={mode}
              onModeChange={setMode}
              onSend={() => void submitTurn(mode)}
              onQuickMode={(m) => void submitTurn(m)}
              sending={Boolean(busy)}
              onAttach={stageFile}
              papers={stagedPapers}
              onRemovePaper={removeStaged}
              projects={projects}
              projectId={projectId}
              onProjectChange={setProjectId}
              model={llmModel}
              models={models}
              onModelChange={setLlmModel}
              autoFocus
            />
          </div>

          {recentPapers.length > 0 && stagedPapers.length === 0 && (
            <section className="mt-8">
              <h2 className="mb-2.5 px-1 text-xs font-medium text-muted-foreground">
                {t("chat.entry.recent_papers")}
              </h2>
              <div className="grid gap-2 sm:grid-cols-2">
                {recentPapers.map((paper) => (
                  <button
                    key={paper.paperId}
                    type="button"
                    disabled={Boolean(busy)}
                    onClick={() => handlePickRecent(paper.paperId, paper.title)}
                    className="flex items-start gap-2.5 rounded-xl border border-border bg-card px-3 py-2.5 text-left transition-colors hover:border-foreground/20 disabled:opacity-60"
                  >
                    {busy === paper.paperId ? (
                      <span className="mt-0.5 inline-block h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
                    ) : (
                      <IconBook className="mt-0.5 shrink-0 text-[14px] text-muted-foreground" />
                    )}
                    <span className="line-clamp-2 min-w-0 text-[0.8125rem] leading-snug text-foreground/90">
                      {paper.title || paper.paperId.slice(0, 16)}
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
