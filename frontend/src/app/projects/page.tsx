"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createProject, listProjects } from "@/lib/agent";
import type { AgentProject } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import { IconArrowRight, IconDocument, IconPlus } from "@/components/icons";

/**
 * Research projects: groups of conversations about one question.
 *
 * Creating one is the only action here; everything else happens on the
 * project's own page, which is where a new conversation starts already filed
 * under it.
 */
export default function ProjectsPage() {
  const router = useRouter();
  const { t } = useTranslation();
  const [projects, setProjects] = useState<AgentProject[] | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    listProjects(showArchived)
      .then((data) => setProjects(data.projects))
      .catch(() => setProjects([]));
  }, [showArchived]);

  const create = useCallback(async () => {
    if (!title.trim() || creating) return;
    setCreating(true);
    setError("");
    try {
      const { project } = await createProject({ title: title.trim(), description: description.trim() });
      router.push(`/projects/${project.project_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setCreating(false);
    }
  }, [title, description, creating, router]);

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8">
        <p className="mb-2 inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-accent-foreground">
          <IconDocument className="h-4 w-4" />
          {t("project.list.eyebrow")}
        </p>
        <h1 className="font-display text-3xl font-bold text-foreground">{t("project.list.title")}</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{t("project.list.subtitle")}</p>
      </div>

      <section className="mb-10 rounded-2xl border border-border bg-card p-5">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void create();
            }
          }}
          maxLength={200}
          placeholder={t("project.create.title_placeholder")}
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40"
        />
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={2}
          maxLength={4000}
          placeholder={t("project.create.description_placeholder")}
          className="mt-2 w-full resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40"
        />
        <div className="mt-3 flex items-center justify-between gap-3">
          {error ? <p className="text-xs text-destructive">{error}</p> : <span />}
          <button
            type="button"
            disabled={!title.trim() || creating}
            onClick={() => void create()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            <IconPlus className="text-[13px]" />
            {creating ? t("project.create.creating") : t("project.create.submit")}
          </button>
        </div>
      </section>

      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {t("project.list.title")}
        </h2>
        <button
          type="button"
          onClick={() => setShowArchived((v) => !v)}
          className="text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          {showArchived ? t("project.list.hide_archived") : t("project.list.show_archived")}
        </button>
      </div>

      {projects !== null && projects.length === 0 && (
        <div className="rounded-xl border border-dashed border-border px-5 py-6 text-center">
          <p className="text-sm text-muted-foreground">{t("project.list.empty")}</p>
        </div>
      )}

      <div className="space-y-2">
        {(projects ?? []).map((project) => (
          <Link
            key={project.project_id}
            href={`/projects/${project.project_id}`}
            className="lift flex items-center justify-between gap-4 rounded-xl border border-border bg-card px-4 py-3"
          >
            <span className="min-w-0">
              <span className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-foreground">
                  {project.title || t("project.untitled")}
                </span>
                {project.status === "archived" && (
                  <span className="shrink-0 rounded bg-muted px-1.5 py-[1px] text-[0.65rem] text-muted-foreground">
                    {t("project.archived")}
                  </span>
                )}
              </span>
              {project.description && (
                <span className="mt-0.5 line-clamp-1 block text-xs text-muted-foreground">
                  {project.description}
                </span>
              )}
            </span>
            <span className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground">
              {t("project.sessions", { count: project.session_count })}
              <IconArrowRight className="h-4 w-4" />
            </span>
          </Link>
        ))}
      </div>
    </main>
  );
}
