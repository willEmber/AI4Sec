"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { deleteMemory, getProject, searchConversations, updateProject } from "@/lib/agent";
import type { ProjectDetail, RecalledTurn, SessionPaper } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import { IconChevronLeft, IconPlus } from "@/components/icons";

const AVAILABILITY_CLASS: Record<SessionPaper["availability"], string> = {
  candidate: "bg-muted text-muted-foreground",
  downloading: "bg-accent text-accent-foreground",
  pdf_ready: "bg-accent text-accent-foreground",
  parsed: "bg-muted text-success",
  unavailable: "bg-muted text-destructive",
};

/**
 * One research project: what it is about, its conversations, the papers they
 * read, what is remembered for it, and a search over its earlier answers.
 *
 * The search is the same one the agent's recall runs, so what the reader finds
 * here is what the agent can find.
 */
export default function ProjectPage() {
  const params = useParams();
  const projectId = params.projectId as string;
  const { t } = useTranslation();

  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RecalledTurn[] | null>(null);
  const [searching, setSearching] = useState(false);

  const reload = useCallback(async () => {
    try {
      const data = await getProject(projectId);
      setDetail(data);
      setTitle(data.project.title);
      setDescription(data.project.description);
    } catch {
      setNotFound(true);
    }
  }, [projectId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const save = useCallback(
    async (changes: { title?: string; description?: string; status?: "active" | "archived" }) => {
      setBusy(true);
      setError("");
      try {
        await updateProject(projectId, changes);
        await reload();
        setEditing(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [projectId, reload],
  );

  const search = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    try {
      const data = await searchConversations(q, projectId);
      setResults(data.turns);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSearching(false);
    }
  }, [query, projectId]);

  const forget = useCallback(
    async (memoryId: string) => {
      await deleteMemory(memoryId).catch(() => {});
      await reload();
    },
    [reload],
  );

  if (notFound) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-12">
        <p className="text-sm text-muted-foreground">{t("project.detail.not_found")}</p>
        <Link href="/projects" className="mt-4 inline-block text-sm underline">
          {t("project.detail.back")}
        </Link>
      </main>
    );
  }

  const project = detail?.project;
  const sectionHeading = "mb-3 text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground";

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <Link
        href="/projects"
        className="mb-6 inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
      >
        <IconChevronLeft className="h-3.5 w-3.5" />
        {t("project.detail.back")}
      </Link>

      {/* Title, description and the actions on the project itself. */}
      <section className="mb-8">
        {editing ? (
          <div className="space-y-2">
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={200}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-lg font-semibold text-foreground focus:outline-none focus:ring-2 focus:ring-ring/40"
            />
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
              maxLength={4000}
              placeholder={t("project.create.description_placeholder")}
              className="w-full resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40"
            />
            <div className="flex gap-2">
              <button
                type="button"
                disabled={busy || !title.trim()}
                onClick={() => void save({ title: title.trim(), description: description.trim() })}
                className="rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-40"
              >
                {t("project.detail.save")}
              </button>
              <button
                type="button"
                onClick={() => {
                  setEditing(false);
                  setTitle(project?.title ?? "");
                  setDescription(project?.description ?? "");
                }}
                className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground"
              >
                {t("project.detail.cancel")}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <h1 className="font-display text-2xl font-bold text-foreground">
                {project ? project.title || t("project.untitled") : "…"}
                {project?.status === "archived" && (
                  <span className="ml-2 rounded bg-muted px-1.5 py-[1px] align-middle text-[0.7rem] font-normal text-muted-foreground">
                    {t("project.archived")}
                  </span>
                )}
              </h1>
              <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">
                {project?.description || t("project.detail.no_description")}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Link
                href={`/chat?new=1&project=${projectId}`}
                className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
              >
                <IconPlus className="text-[12px]" />
                {t("project.detail.new_chat")}
              </Link>
              <button
                type="button"
                onClick={() => setEditing(true)}
                className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground"
              >
                {t("project.detail.edit")}
              </button>
              {project && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    void save({ status: project.status === "archived" ? "active" : "archived" })
                  }
                  className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground disabled:opacity-40"
                >
                  {project.status === "archived"
                    ? t("project.detail.unarchive")
                    : t("project.detail.archive")}
                </button>
              )}
            </div>
          </div>
        )}
        {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
      </section>

      {/* Search over earlier answers — the agent's recall, for the reader. */}
      <section className="mb-10">
        <h2 className={sectionHeading}>{t("project.detail.search")}</h2>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void search();
              }
            }}
            maxLength={500}
            placeholder={t("project.detail.search_placeholder")}
            className="min-w-0 flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40"
          />
          <button
            type="button"
            disabled={!query.trim() || searching}
            onClick={() => void search()}
            className="rounded-lg border border-border px-4 text-sm text-foreground transition-colors hover:bg-muted disabled:opacity-40"
          >
            {t("project.detail.search_submit")}
          </button>
        </div>
        {results !== null && results.length === 0 && (
          <p className="mt-3 text-xs text-muted-foreground">{t("project.detail.search_empty")}</p>
        )}
        <div className="mt-3 space-y-2">
          {(results ?? []).map((turn) => (
            <Link
              key={`${turn.session_id}-${turn.run_id}`}
              href={`/chat/${turn.session_id}`}
              className="block rounded-xl border border-border bg-card px-4 py-3 transition-colors hover:border-foreground/20"
            >
              <span className="flex items-center justify-between gap-3 text-[0.7rem] text-muted-foreground">
                <span className="truncate">{turn.session_title || t("chat.sessions.untitled")}</span>
                <span className="shrink-0">{turn.asked_at.slice(0, 10)}</span>
              </span>
              <span className="mt-1 block text-sm font-medium text-foreground">{turn.question}</span>
              <span className="mt-1 line-clamp-3 block text-xs leading-relaxed text-muted-foreground">
                {turn.answer_excerpt}
              </span>
              {turn.evidence.length > 0 && (
                <span className="mt-1.5 block text-[0.7rem] text-accent-foreground">
                  {t("project.detail.cited", { count: turn.evidence.length })}
                </span>
              )}
            </Link>
          ))}
        </div>
      </section>

      <div className="grid gap-10 md:grid-cols-2">
        <section>
          <h2 className={sectionHeading}>{t("project.detail.sessions")}</h2>
          {detail && detail.sessions.length === 0 && (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {t("project.detail.sessions_empty")}
            </p>
          )}
          <div className="space-y-1">
            {(detail?.sessions ?? []).map((session) => (
              <Link
                key={session.session_id}
                href={`/chat/${session.session_id}`}
                className="flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <span className="truncate">{session.title || t("chat.sessions.untitled")}</span>
                <span className="shrink-0 text-[0.7rem]">{session.updated_at.slice(0, 10)}</span>
              </Link>
            ))}
          </div>
        </section>

        <section>
          <h2 className={sectionHeading}>{t("project.detail.papers")}</h2>
          {detail && detail.papers.length === 0 && (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {t("project.detail.papers_empty")}
            </p>
          )}
          <div className="space-y-1.5">
            {(detail?.papers ?? []).map((paper) => (
              <div key={paper.literature_id} className="rounded-lg px-3 py-2">
                <span className="line-clamp-2 block text-xs font-medium leading-snug text-foreground">
                  {paper.title || t("chat.papers.untitled")}
                </span>
                <span className="mt-1 flex items-center gap-1.5">
                  <span
                    className={`rounded px-1.5 py-[1px] text-[0.65rem] ${AVAILABILITY_CLASS[paper.availability]}`}
                  >
                    {t(`chat.availability.${paper.availability}`)}
                  </span>
                  {paper.year_known && (
                    <span className="text-[0.65rem] text-muted-foreground">{paper.year}</span>
                  )}
                </span>
              </div>
            ))}
          </div>

          <h2 className={`${sectionHeading} mt-8`}>{t("project.detail.memories")}</h2>
          {detail && detail.memories.length === 0 && (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {t("project.detail.memories_empty")}
            </p>
          )}
          {(detail?.memories ?? []).map((memory) => (
            <div
              key={memory.memory_id}
              className="group mb-1 flex items-start gap-2 rounded-lg px-2 py-1.5 hover:bg-muted/60"
            >
              <span className="mt-0.5 shrink-0 rounded bg-muted px-1 py-[1px] text-[0.6rem] uppercase text-muted-foreground">
                {t(`chat.memory.kind.${memory.kind}`)}
              </span>
              <span className="min-w-0 flex-1 text-xs leading-snug text-foreground">{memory.content}</span>
              <button
                type="button"
                onClick={() => void forget(memory.memory_id)}
                title={t("chat.memory.forget")}
                className="shrink-0 text-[0.7rem] text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
              >
                ✕
              </button>
            </div>
          ))}
        </section>
      </div>
    </main>
  );
}
