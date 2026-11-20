"use client";

import { useCallback, useMemo, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { deleteSession, describeApiError, updateSession } from "@/lib/agent";
import type { AgentProject, AgentSession } from "@/lib/agent";
import { parseServerTime } from "@/lib/agentEvents";
import { LanguageToggle, ThemeToggle, useTranslation } from "@/lib/i18n";
import { dateBucket } from "@/lib/turns";
import type { DateBucket } from "@/lib/turns";
import { usePersistentState } from "@/hooks/usePersistentState";
import { AuthMenu } from "@/components/AuthMenu";
import Menu, { MenuItem } from "@/components/Menu";
import {
  IconBook,
  IconChevronDown,
  IconChevronRight,
  IconFolder,
  IconMore,
  IconPanelLeftClose,
  IconPanelLeftOpen,
  IconPencil,
  IconPlus,
  IconSearch,
  IconTrash,
  IconUser,
  IconX,
} from "@/components/icons";

interface Props {
  sessions: AgentSession[];
  loaded: boolean;
  currentSessionId: string;
  projects: AgentProject[];
  /** The open conversation's project, when it has one. */
  project: AgentProject | null;
  scope: "all" | "project";
  onScope: (scope: "all" | "project") => void;
  activeRunsCount: number;
  onChanged: () => void;
  onOpenSearch: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  asDrawer?: boolean;
}

const BUCKETS: DateBucket[] = ["today", "week", "earlier"];
const SIDEBAR_PROJECTS_LIMIT = 5;

/**
 * Unified workspace sidebar across conversations, projects, library and account.
 *
 * Replaces the old split between the top 5-item `NavBar` and the 2x2 escape
 * grid at the foot of `/chat`: everything stays inside one calm shell with
 * client-side transitions.
 */
export default function ChatNav({
  sessions,
  loaded,
  currentSessionId,
  projects,
  project,
  scope,
  onScope,
  activeRunsCount,
  onChanged,
  onOpenSearch,
  collapsed,
  onToggleCollapse,
  asDrawer = false,
}: Props) {
  const { t } = useTranslation();
  const pathname = usePathname();
  const router = useRouter();
  const [projectsExpanded, setProjectsExpanded] = usePersistentState(
    "scholar.nav.projects_open",
    true,
    true,
  );
  const [renaming, setRenaming] = useState("");
  const [titleDraft, setTitleDraft] = useState("");
  const [error, setError] = useState("");

  const scopedProjectId = scope === "project" && project ? project.project_id : "";
  const newChatHref = scopedProjectId ? `/chat?project=${scopedProjectId}` : "/chat";
  const inLibrary = pathname.startsWith("/library") || pathname.startsWith("/paper/");
  const inProjectsRoot = pathname === "/projects";

  const grouped = useMemo(() => {
    const now = Date.now();
    const map = new Map<DateBucket, AgentSession[]>();
    for (const session of sessions) {
      const bucket = dateBucket(parseServerTime(session.updated_at), now);
      map.set(bucket, [...(map.get(bucket) ?? []), session]);
    }
    return map;
  }, [sessions]);

  const rename = useCallback(
    async (sessionId: string) => {
      const title = titleDraft.trim();
      setRenaming("");
      if (!title) return;
      try {
        await updateSession(sessionId, { title });
        onChanged();
      } catch (err) {
        setError(describeApiError(err, t));
      }
    },
    [titleDraft, onChanged, t],
  );

  const remove = useCallback(
    async (sessionId: string) => {
      if (!window.confirm(t("chat.sessions.delete_confirm"))) return;
      try {
        await deleteSession(sessionId);
        onChanged();
        if (sessionId === currentSessionId) router.push("/chat");
      } catch (err) {
        setError(describeApiError(err, t));
      }
    },
    [currentSessionId, onChanged, router, t],
  );

  if (collapsed) {
    const railItem = (active = false) =>
      `relative flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
        active
          ? "bg-card font-medium text-foreground soft-shadow"
          : "text-muted-foreground hover:bg-card hover:text-foreground"
      }`;
    return (
      <aside className="flex h-full w-13 shrink-0 flex-col items-center gap-1 border-r border-border bg-muted/45 py-2.5">
        <button
          type="button"
          onClick={onToggleCollapse}
          title={t("chat.sidebar.expand")}
          className={railItem()}
        >
          <IconPanelLeftOpen className="text-[17px]" />
        </button>
        <Link href={newChatHref} title={t("chat.nav.new")} className={railItem(pathname === "/chat")}>
          <IconPlus className="text-[17px] text-primary" />
        </Link>
        <button type="button" onClick={onOpenSearch} title={`${t("search.trigger")} (⌘K)`} className={railItem()}>
          <IconSearch className="text-[16px]" />
        </button>
        <div className="my-1 h-px w-6 bg-border" />
        <Link href="/library" title={t("nav.library_hub")} className={railItem(inLibrary)}>
          <IconBook className="text-[16px]" />
          {activeRunsCount > 0 && (
            <span className="absolute right-1.5 top-1.5 h-2 w-2 animate-pulse rounded-full bg-primary" />
          )}
        </Link>
        <Link href="/projects" title={t("nav.projects")} className={railItem(pathname.startsWith("/projects"))}>
          <IconFolder className="text-[16px]" />
        </Link>
        <div className="flex-1" />
        <ThemeToggle compact />
        <Link
          href="/account"
          title={t("account.title")}
          className={railItem(pathname.startsWith("/account"))}
        >
          <IconUser className="text-[16px]" />
        </Link>
      </aside>
    );
  }

  const shownProjects = projects.slice(0, SIDEBAR_PROJECTS_LIMIT);

  return (
    <aside
      className={`flex h-full shrink-0 flex-col border-r border-border ${
        asDrawer ? "w-72 max-w-[85vw] bg-background" : "w-64 bg-muted/45"
      }`}
    >
      {/* Brand & collapse */}
      <div className="flex h-12 shrink-0 items-center justify-between pl-4 pr-2">
        <Link href="/chat" className="flex items-center gap-2 font-semibold tracking-tight">
          <Image src="/logo.png" alt="Scholar" width={26} height={26} className="h-6 w-6 object-contain" />
          <span className="text-[15px]">{t("nav.brand")}</span>
        </Link>
        <button
          type="button"
          onClick={onToggleCollapse}
          title={t("chat.sidebar.collapse")}
          aria-label={t("chat.sidebar.collapse")}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
        >
          {asDrawer ? <IconX className="text-[16px]" /> : <IconPanelLeftClose className="text-[16px]" />}
        </button>
      </div>

      {/* Primary actions: New chat + Search + Library */}
      <div className="shrink-0 space-y-1.5 px-2.5 pb-2">
        <Link
          href={newChatHref}
          className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-sm font-medium transition-colors ${
            pathname === "/chat"
              ? "border-primary/35 bg-card text-foreground soft-shadow"
              : "border-border bg-card text-foreground soft-shadow hover:border-foreground/20"
          }`}
        >
          <IconPlus className="text-[15px] text-primary" />
          <span className="flex-1 truncate">{t("chat.nav.new")}</span>
        </Link>

        <button
          type="button"
          onClick={onOpenSearch}
          className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-[0.8125rem] text-muted-foreground transition-colors hover:bg-card/80 hover:text-foreground"
        >
          <IconSearch className="shrink-0 text-[14px]" />
          <span className="min-w-0 flex-1 truncate text-left">{t("search.trigger")}</span>
          <kbd className="hidden rounded border border-border/80 bg-card/70 px-1.5 py-[1px] font-mono text-[10px] text-muted-foreground sm:inline">
            ⌘K
          </kbd>
        </button>

        <Link
          href="/library"
          className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-[0.8125rem] transition-colors ${
            inLibrary
              ? "bg-card font-medium text-foreground soft-shadow"
              : "text-foreground/80 hover:bg-card/80 hover:text-foreground"
          }`}
        >
          <IconBook className="shrink-0 text-[15px] text-muted-foreground" />
          <span className="min-w-0 flex-1 truncate">{t("nav.library_hub")}</span>
          {activeRunsCount > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-accent px-1.5 py-[1px] text-[11px] font-medium text-accent-foreground">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
              {activeRunsCount}
            </span>
          )}
        </Link>

        {error && <p className="px-1 text-xs text-destructive">{error}</p>}
      </div>

      {/* Scrollable middle: Projects + Recent conversations */}
      <div className="relative min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {/* Projects section */}
        <section className="mb-3">
          <div className="group flex items-center justify-between px-1.5 py-1">
            <button
              type="button"
              onClick={() => setProjectsExpanded((v) => !v)}
              className="flex min-w-0 items-center gap-1 rounded px-1 py-0.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              {projectsExpanded ? (
                <IconChevronDown className="text-[12px]" />
              ) : (
                <IconChevronRight className="text-[12px]" />
              )}
              <span>{t("nav.projects")}</span>
            </button>
            <div className="flex items-center gap-0.5">
              <Link
                href="/projects?new=1"
                title={t("nav.new_project")}
                aria-label={t("nav.new_project")}
                className="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
              >
                <IconPlus className="text-[13px]" />
              </Link>
            </div>
          </div>

          {projectsExpanded && (
            <div className="mt-0.5 space-y-0.5">
              {shownProjects.length === 0 ? (
                <Link
                  href="/projects"
                  className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-[0.8125rem] transition-colors ${
                    inProjectsRoot
                      ? "bg-card font-medium text-foreground soft-shadow"
                      : "text-muted-foreground hover:bg-card/70 hover:text-foreground"
                  }`}
                >
                  <IconFolder className="shrink-0 text-[14px]" />
                  <span className="truncate">{t("project.create.toggle")}</span>
                </Link>
              ) : (
                <>
                  {shownProjects.map((item) => {
                    const active =
                      pathname === `/projects/${item.project_id}` ||
                      project?.project_id === item.project_id;
                    return (
                      <Link
                        key={item.project_id}
                        href={`/projects/${item.project_id}`}
                        className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-[0.8125rem] transition-colors ${
                          active
                            ? "bg-card font-medium text-foreground soft-shadow"
                            : "text-foreground/75 hover:bg-card/70 hover:text-foreground"
                        }`}
                      >
                        <IconFolder
                          className={`shrink-0 text-[14px] ${
                            active ? "text-primary" : "text-muted-foreground"
                          }`}
                        />
                        <span className="min-w-0 flex-1 truncate">
                          {item.title || t("project.untitled")}
                        </span>
                      </Link>
                    );
                  })}
                  {projects.length > SIDEBAR_PROJECTS_LIMIT || !inProjectsRoot ? (
                    <Link
                      href="/projects"
                      className={`block rounded-lg px-2.5 py-1 text-xs transition-colors ${
                        inProjectsRoot
                          ? "font-medium text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("nav.all_projects", { count: projects.length })}
                    </Link>
                  ) : null}
                </>
              )}
            </div>
          )}
        </section>

        {/* Scope switch inside a project's conversation */}
        {project && currentSessionId && (
          <div className="mb-2 flex rounded-lg bg-card/70 p-0.5 text-xs">
            {(["all", "project"] as const).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => onScope(key)}
                aria-pressed={scope === key}
                title={key === "project" ? project.title : undefined}
                className={`min-w-0 flex-1 truncate rounded-md px-2 py-1 transition-colors ${
                  scope === key ? "bg-card font-medium text-foreground soft-shadow" : "text-muted-foreground"
                }`}
              >
                {key === "all" ? t("chat.nav.scope_all") : t("chat.nav.scope_project")}
              </button>
            ))}
          </div>
        )}

        {/* Recent conversations */}
        {loaded && sessions.length === 0 && (
          <p className="px-2.5 py-2 text-xs text-muted-foreground">{t("chat.sessions.empty")}</p>
        )}
        {BUCKETS.filter((bucket) => grouped.has(bucket)).map((bucket) => (
          <section key={bucket} className="mb-2">
            <h2 className="px-2.5 pb-1 pt-1.5 text-xs font-medium text-muted-foreground/80">
              {t(`chat.nav.${bucket}`)}
            </h2>
            {(grouped.get(bucket) ?? []).map((session) => {
              const current = session.session_id === currentSessionId;
              if (renaming === session.session_id) {
                return (
                  <input
                    key={session.session_id}
                    autoFocus
                    value={titleDraft}
                    onChange={(e) => setTitleDraft(e.target.value)}
                    onBlur={() => void rename(session.session_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.nativeEvent.isComposing) e.currentTarget.blur();
                      if (e.key === "Escape") setRenaming("");
                    }}
                    maxLength={200}
                    className="mb-0.5 w-full rounded-lg border border-primary/50 bg-card px-2.5 py-1.5 text-[0.8125rem] text-foreground focus:outline-none"
                  />
                );
              }
              return (
                <div
                  key={session.session_id}
                  className={`group mb-0.5 flex items-center rounded-lg text-[0.8125rem] transition-colors ${
                    current
                      ? "bg-card font-medium text-foreground soft-shadow"
                      : "text-foreground/75 hover:bg-card/70 hover:text-foreground"
                  }`}
                >
                  <Link
                    href={`/chat/${session.session_id}`}
                    title={session.title}
                    className="min-w-0 flex-1 truncate py-1.5 pl-2.5"
                  >
                    {session.title || t("chat.sessions.untitled")}
                  </Link>
                  <Menu
                    label={t("chat.nav.more")}
                    trigger={<IconMore className="text-[15px]" />}
                    width={168}
                    buttonClassName="mx-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-opacity hover:bg-muted hover:text-foreground focus:opacity-100 group-hover:opacity-100 aria-expanded:opacity-100 [@media(hover:hover)]:opacity-0"
                  >
                    {(close) => (
                      <>
                        <MenuItem
                          onSelect={() => {
                            close();
                            setTitleDraft(session.title);
                            setRenaming(session.session_id);
                          }}
                        >
                          <IconPencil className="text-[14px] text-muted-foreground" />
                          {t("chat.nav.rename")}
                        </MenuItem>
                        <MenuItem
                          danger
                          onSelect={() => {
                            close();
                            void remove(session.session_id);
                          }}
                        >
                          <IconTrash className="text-[14px]" />
                          {t("chat.sessions.delete")}
                        </MenuItem>
                      </>
                    )}
                  </Menu>
                </div>
              );
            })}
          </section>
        ))}
      </div>

      {/* Footer: Account, Theme, Language */}
      <div className="shrink-0 border-t border-border px-2.5 py-2.5">
        <div className="flex items-center gap-1.5">
          <div className="min-w-0 flex-1">
            <AuthMenu placement="above" />
          </div>
          <ThemeToggle />
          <LanguageToggle />
        </div>
      </div>
    </aside>
  );
}
