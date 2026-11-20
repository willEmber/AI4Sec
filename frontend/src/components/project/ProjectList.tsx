"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createProject, describeApiError, listProjects } from "@/lib/agent";
import type { AgentProject } from "@/lib/agent";
import { getMe } from "@/lib/auth";
import { useTranslation } from "@/lib/i18n";
import { useChatShell } from "@/components/agent/ChatShell";
import { IconArrowRight, IconFolder, IconPlus } from "@/components/icons";

/**
 * Research projects list with a clean toggleable creation card.
 *
 * Opens the creation card automatically when there are no projects yet or when
 * navigated with `?new=1` from the sidebar's quick `+` button.
 */
export default function ProjectList({ initialCreateOpen = false }: { initialCreateOpen?: boolean }) {
  const router = useRouter();
  const { t } = useTranslation();
  const { refreshProjects } = useChatShell();
  const [projects, setProjects] = useState<AgentProject[] | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [createOpen, setCreateOpen] = useState(initialCreateOpen);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (initialCreateOpen) setCreateOpen(true);
  }, [initialCreateOpen]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const me = await getMe();
      if (cancelled) return;
      if (!me?.principal_id) {
        setProjects([]);
        if (!showArchived) setCreateOpen(true);
        return;
      }
      try {
        const data = await listProjects(showArchived);
        if (cancelled) return;
        setProjects(data.projects);
        if (data.projects.length === 0 && !showArchived) setCreateOpen(true);
      } catch {
        if (!cancelled) setProjects([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [showArchived]);

  const create = useCallback(async () => {
    if (!title.trim() || creating) return;
    setCreating(true);
    setError("");
    try {
      const { project } = await createProject({
        title: title.trim(),
        description: description.trim(),
      });
      refreshProjects();
      router.push(`/projects/${project.project_id}`);
    } catch (err) {
      setError(describeApiError(err, t));
      setCreating(false);
    }
  }, [title, description, creating, refreshProjects, router, t]);

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => setShowArchived((v) => !v)}
          className="text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          {showArchived ? t("project.list.hide_archived") : t("project.list.show_archived")}
        </button>

        {!createOpen && (
          <button
            type="button"
            onClick={() => setCreateOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-3.5 py-2 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
          >
            <IconPlus className="text-[13px]" />
            {t("project.create.toggle")}
          </button>
        )}
      </div>

      {createOpen && (
        <section className="mb-7 rounded-2xl border border-border bg-card p-5 soft-shadow">
          <input
            autoFocus
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                e.preventDefault();
                void create();
              }
              if (e.key === "Escape" && (projects?.length ?? 0) > 0) {
                setCreateOpen(false);
              }
            }}
            maxLength={200}
            placeholder={t("project.create.title_placeholder")}
            className="w-full rounded-xl border border-border bg-background px-3.5 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/50 focus:outline-none"
          />
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={2}
            maxLength={4000}
            placeholder={t("project.create.description_placeholder")}
            className="mt-2.5 w-full resize-y rounded-xl border border-border bg-background px-3.5 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/50 focus:outline-none"
          />
          <div className="mt-3 flex items-center justify-between gap-3">
            {error ? <p className="text-xs text-destructive">{error}</p> : <span />}
            <div className="flex items-center gap-2">
              {(projects?.length ?? 0) > 0 && (
                <button
                  type="button"
                  onClick={() => setCreateOpen(false)}
                  className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
                >
                  {t("project.detail.cancel")}
                </button>
              )}
              <button
                type="button"
                disabled={!title.trim() || creating}
                onClick={() => void create()}
                className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
              >
                <IconPlus className="text-[13px]" />
                {creating ? t("project.create.creating") : t("project.create.submit")}
              </button>
            </div>
          </div>
        </section>
      )}

      {projects !== null && projects.length === 0 && (
        <div className="rounded-2xl border border-dashed border-border bg-card/50 px-5 py-8 text-center">
          <p className="text-sm text-muted-foreground">{t("project.list.empty")}</p>
        </div>
      )}

      <div className="space-y-2.5">
        {(projects ?? []).map((project) => (
          <Link
            key={project.project_id}
            href={`/projects/${project.project_id}`}
            className="lift flex items-center justify-between gap-4 rounded-2xl border border-border bg-card px-4 py-3.5 soft-shadow"
          >
            <span className="flex min-w-0 items-start gap-3">
              <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent text-primary">
                <IconFolder className="text-[15px]" />
              </span>
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
            </span>
            <span className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground">
              {t("project.sessions", { count: project.session_count })}
              <IconArrowRight className="h-4 w-4" />
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
