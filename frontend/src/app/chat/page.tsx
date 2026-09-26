"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { createSession, listSessions } from "@/lib/agent";
import type { AgentMode, AgentSession } from "@/lib/agent";
import { listModels, listRecentRuns, uploadPaper } from "@/lib/api";
import type { RecentRunResponse } from "@/lib/types";
import { useTranslation } from "@/lib/i18n";
import {
  IconArrowRight,
  IconLens,
  IconPlus,
  IconSnap,
  IconSparkles,
  IconSphere,
  IconUpload,
} from "@/components/icons";
import type { ComponentType } from "react";

const MODES: AgentMode[] = ["auto", "snap", "lens", "sphere"];
const MODE_ICON: Record<AgentMode, ComponentType<{ className?: string }>> = {
  auto: IconSparkles,
  snap: IconSnap,
  lens: IconLens,
  sphere: IconSphere,
};

/**
 * The entry to the platform: start a conversation.
 *
 * Three ways in. Drop a PDF and a session opens on it; pick a paper already in
 * the workspace; or open an empty conversation and let the agent find papers.
 * `?paper=<id>` (from a report page) and `?new=1` (from the sidebar) skip the
 * choice; `?mode=` carries a preselected mode into the new session.
 *
 * Reading the query string opts the page out of prerendering, so it needs a
 * Suspense boundary of its own.
 */
export default function ChatEntryPage() {
  return (
    <Suspense fallback={<main className="mx-auto max-w-3xl px-6 py-12" />}>
      <ChatEntry />
    </Suspense>
  );
}

function ChatEntry() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const presetPaperId = searchParams.get("paper") || "";
  const presetMode = (searchParams.get("mode") || "auto") as AgentMode;
  const wantsNew = searchParams.get("new") === "1";
  const { t, locale } = useTranslation();

  const [sessions, setSessions] = useState<AgentSession[] | null>(null);
  const [runs, setRuns] = useState<RecentRunResponse[] | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [llmModel, setLlmModel] = useState("");
  const [mode, setMode] = useState<AgentMode>(MODES.includes(presetMode) ? presetMode : "auto");
  const [creating, setCreating] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listSessions()
      .then((data) => setSessions(data.sessions))
      .catch(() => setSessions([]));
    listRecentRuns(40, false)
      .then(setRuns)
      .catch(() => setRuns([]));
    listModels()
      .then((res) => {
        setModels(res.models);
        setLlmModel(res.default || res.models[0] || "");
      })
      .catch(() => setModels([]));
  }, []);

  // One row per paper: the same PDF is usually analysed several times, and the
  // agent reads the paper, not the run.
  const papers = useMemo(() => {
    const seen = new Map<string, RecentRunResponse>();
    for (const run of runs ?? []) {
      if (run.paper_id && !seen.has(run.paper_id)) seen.set(run.paper_id, run);
    }
    return [...seen.values()];
  }, [runs]);

  const open = useCallback(
    async (paperIds: string[], title: string, key: string) => {
      if (creating) return;
      setCreating(key);
      setError("");
      try {
        const session = await createSession({
          title,
          language: locale === "zh" ? "zh" : "en",
          llm_model: llmModel,
          paper_ids: paperIds,
        });
        const query = mode !== "auto" ? `?mode=${mode}` : "";
        router.push(`/chat/${session.session_id}${query}`);
      } catch (err) {
        setError(String(err));
        setCreating("");
      }
    },
    [creating, locale, llmModel, mode, router],
  );

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file || creating) return;
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setError(t("upload.drop_error"));
        return;
      }
      setCreating("upload");
      setError("");
      try {
        const uploaded = await uploadPaper(file);
        await open([uploaded.paper_id], file.name.replace(/\.pdf$/i, ""), "upload");
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setCreating("");
      }
    },
    [creating, open, t],
  );

  // Arriving with ?paper=… or ?new=1 means the choice is already made.
  useEffect(() => {
    if (creating) return;
    if (presetPaperId) void open([presetPaperId], "", presetPaperId);
    else if (wantsNew) void open([], "", "new");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetPaperId, wantsNew]);

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8">
        <p className="mb-2 inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-accent-foreground">
          <IconSparkles className="h-4 w-4" />
          {t("chat.entry.eyebrow")}
        </p>
        <h1 className="font-display text-3xl font-bold text-foreground">
          {t("chat.entry.title")}
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          {t("chat.entry.subtitle")}
        </p>
      </div>

      {error && (
        <p className="mb-6 rounded-lg border border-border bg-muted px-3 py-2 text-xs text-destructive">
          {error}
        </p>
      )}

      {/* Drop zone: the fastest way in. */}
      <section
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void handleFile(e.dataTransfer.files[0]);
        }}
        onClick={() => !creating && fileRef.current?.click()}
        className={`mb-6 cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
          dragOver ? "border-primary bg-accent" : "border-border bg-card hover:border-foreground/20"
        } ${creating ? "opacity-60" : ""}`}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,application/pdf"
          className="hidden"
          onChange={(e) => void handleFile(e.target.files?.[0])}
        />
        <span className="mx-auto flex h-11 w-11 items-center justify-center rounded-2xl bg-muted text-xl text-muted-foreground">
          {creating === "upload" ? (
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-border border-t-primary" />
          ) : (
            <IconUpload />
          )}
        </span>
        <p className="mt-3 font-medium">
          {creating === "upload" ? t("chat.entry.opening") : t("chat.entry.drop")}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">{t("chat.entry.drop_hint")}</p>
      </section>

      {/* Mode + model for the session about to be created. */}
      <section className="mb-8 grid gap-3 sm:grid-cols-[1fr_auto]">
        <div className="flex flex-wrap gap-2">
          {MODES.map((m) => {
            const Icon = MODE_ICON[m];
            const active = m === mode;
            return (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                title={t(`chat.mode.${m}.desc`)}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors ${
                  active
                    ? "border-primary bg-accent text-accent-foreground"
                    : "border-border text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="text-[13px]" />
                {t(`chat.mode.${m}.label`)}
              </button>
            );
          })}
        </div>
        {models.length > 0 && (
          <select
            value={llmModel}
            onChange={(e) => setLlmModel(e.target.value)}
            title={t("upload.model_label")}
            className="rounded-lg border border-border bg-card px-3 py-1.5 text-xs text-foreground focus:border-primary focus:outline-none"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}
      </section>

      <section className="mb-10">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
            {t("chat.entry.pick_paper")}
          </h2>
          <button
            type="button"
            disabled={Boolean(creating)}
            onClick={() => void open([], "", "new")}
            className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground disabled:opacity-60"
          >
            <IconPlus className="text-[12px]" />
            {creating === "new" ? t("chat.entry.opening") : t("chat.entry.start_empty")}
          </button>
        </div>

        {papers.length === 0 && (
          <div className="rounded-xl border border-dashed border-border px-5 py-6 text-center">
            <p className="text-sm text-muted-foreground">{t("chat.entry.no_papers")}</p>
          </div>
        )}

        <div className="space-y-2">
          {papers.map((paper) => (
            <button
              key={paper.paper_id}
              type="button"
              disabled={Boolean(creating)}
              onClick={() => void open([paper.paper_id], paper.paper_title, paper.paper_id)}
              className="lift flex w-full items-center justify-between gap-4 rounded-xl border border-border bg-card px-4 py-3 text-left disabled:opacity-60"
            >
              <span className="min-w-0">
                <span className="line-clamp-2 block text-sm font-medium text-foreground">
                  {paper.paper_title || paper.paper_id.slice(0, 16)}
                </span>
              </span>
              <span className="shrink-0 text-xs text-muted-foreground">
                {creating === paper.paper_id ? (
                  t("chat.entry.opening")
                ) : (
                  <IconArrowRight className="h-4 w-4" />
                )}
              </span>
            </button>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {t("chat.sessions.heading")}
        </h2>
        {sessions !== null && sessions.length === 0 && (
          <p className="text-xs text-muted-foreground">{t("chat.sessions.empty")}</p>
        )}
        <div className="space-y-1">
          {(sessions ?? []).map((session) => (
            <Link
              key={session.session_id}
              href={`/chat/${session.session_id}`}
              className="block truncate rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              {session.title || t("chat.sessions.untitled")}
            </Link>
          ))}
        </div>
      </section>

      <p className="mt-10 text-center text-[0.7rem] text-muted-foreground">
        {t("chat.entry.classic_hint")}{" "}
        <Link href="/upload" className="underline hover:text-foreground">
          {t("chat.entry.classic_link")}
        </Link>
      </p>
    </main>
  );
}
