"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { listProjects, listSessions } from "@/lib/agent";
import type { AgentProject, AgentSession } from "@/lib/agent";
import { listModels, listRecentRuns } from "@/lib/api";
import { getMe } from "@/lib/auth";
import { useTranslation } from "@/lib/i18n";
import { usePersistentState } from "@/hooks/usePersistentState";
import ChatNav from "@/components/agent/ChatNav";
import SearchModal from "@/components/agent/SearchModal";
import { IconMenu, IconSearch } from "@/components/icons";

interface ChatShellValue {
  sessions: AgentSession[];
  projects: AgentProject[];
  libraryEnabled: boolean;
  activeRunsCount: number;
  refreshSessions: () => void;
  refreshProjects: () => void;
  refreshRuns: () => void;
  setCurrentProject: (project: AgentProject | null) => void;
  openNav: () => void;
  openSearch: () => void;
}

const ChatShellContext = createContext<ChatShellValue>({
  sessions: [],
  projects: [],
  libraryEnabled: false,
  activeRunsCount: 0,
  refreshSessions: () => {},
  refreshProjects: () => {},
  refreshRuns: () => {},
  setCurrentProject: () => {},
  openNav: () => {},
  openSearch: () => {},
});

export function useChatShell(): ChatShellValue {
  return useContext(ChatShellContext);
}

/**
 * Global workspace shell shared across `/chat`, `/projects`, `/library`,
 * `/account`, and standalone report views.
 *
 * Mounted once above workspace routes so moving between conversations, projects
 * and the library keeps the sidebar stable with zero layout shift.
 */
export default function ChatShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { t } = useTranslation();
  const inChat = pathname === "/chat" || pathname.startsWith("/chat/");
  const currentSessionId = pathname.startsWith("/chat/") ? pathname.split("/")[2] ?? "" : "";

  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [projects, setProjects] = useState<AgentProject[]>([]);
  const [libraryEnabled, setLibraryEnabled] = useState(false);
  const [activeRunsCount, setActiveRunsCount] = useState(0);
  const [currentProject, setCurrentProject] = useState<AgentProject | null>(null);
  const [scope, setScope] = useState<"all" | "project">("all");
  const [sessionsToken, setSessionsToken] = useState(0);
  const [projectsToken, setProjectsToken] = useState(0);
  const [runsToken, setRunsToken] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [collapsed, setCollapsed] = usePersistentState("scholar.chat.nav_collapsed", false, false);

  const projectId = currentProject?.project_id ?? "";
  const scoped = scope === "project" && projectId && Boolean(currentSessionId);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const me = await getMe();
      if (cancelled) return;
      if (!me?.principal_id) {
        setSessions([]);
        setLoaded(true);
        return;
      }
      try {
        const data = await listSessions(scoped ? projectId : undefined);
        if (!cancelled) setSessions(data.sessions);
      } catch {
        // Unreachable or unauthorized: keep current state.
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [scoped, projectId, sessionsToken]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const me = await getMe();
      if (cancelled || !me?.principal_id) {
        if (!cancelled) setProjects([]);
        return;
      }
      try {
        const data = await listProjects();
        if (!cancelled) setProjects(data.projects);
      } catch {
        // Ignore transient failure.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectsToken, sessionsToken]);

  useEffect(() => {
    let cancelled = false;
    listModels()
      .then((res) => {
        if (!cancelled) setLibraryEnabled(Boolean(res.library_enabled));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    listRecentRuns(20, true)
      .then((runs) => {
        if (!cancelled) setActiveRunsCount(runs.length);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runsToken, sessionsToken]);

  useEffect(() => {
    if (!projectId || !currentSessionId) setScope("all");
  }, [projectId, currentSessionId]);

  useEffect(() => {
    if (!inChat) setCurrentProject(null);
    setDrawerOpen(false);
  }, [pathname, inChat]);

  // Global keyboard shortcut: Cmd/Ctrl + K opens the search palette.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const refreshSessions = useCallback(() => setSessionsToken((n) => n + 1), []);
  const refreshProjects = useCallback(() => setProjectsToken((n) => n + 1), []);
  const refreshRuns = useCallback(() => setRunsToken((n) => n + 1), []);
  const openNav = useCallback(() => setDrawerOpen(true), []);
  const openSearch = useCallback(() => setSearchOpen(true), []);

  const value = useMemo(
    () => ({
      sessions,
      projects,
      libraryEnabled,
      activeRunsCount,
      refreshSessions,
      refreshProjects,
      refreshRuns,
      setCurrentProject,
      openNav,
      openSearch,
    }),
    [
      sessions,
      projects,
      libraryEnabled,
      activeRunsCount,
      refreshSessions,
      refreshProjects,
      refreshRuns,
      openNav,
      openSearch,
    ],
  );

  const nav = (asDrawer: boolean) => (
    <ChatNav
      sessions={sessions}
      loaded={loaded}
      currentSessionId={currentSessionId}
      projects={projects}
      project={currentProject}
      scope={scope}
      onScope={setScope}
      activeRunsCount={activeRunsCount}
      onChanged={refreshSessions}
      onOpenSearch={() => {
        setDrawerOpen(false);
        setSearchOpen(true);
      }}
      collapsed={!asDrawer && collapsed}
      onToggleCollapse={asDrawer ? () => setDrawerOpen(false) : () => setCollapsed((v) => !v)}
      asDrawer={asDrawer}
    />
  );

  return (
    <ChatShellContext.Provider value={value}>
      <div className="flex h-dvh overflow-hidden bg-background">
        <div className="hidden lg:block">{nav(false)}</div>
        {drawerOpen && (
          <div className="fixed inset-0 z-40 lg:hidden">
            <button
              type="button"
              aria-label="Close"
              onClick={() => setDrawerOpen(false)}
              className="animate-fade-in absolute inset-0 bg-foreground/25"
            />
            <div className="absolute inset-y-0 left-0 shadow-xl">{nav(true)}</div>
          </div>
        )}
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          {!inChat && (
            <header className="flex h-12 shrink-0 items-center justify-between border-b border-border px-3 lg:hidden">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={openNav}
                  title={t("chat.nav.open")}
                  aria-label={t("chat.nav.open")}
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <IconMenu className="text-[18px]" />
                </button>
                <Link href="/chat" className="text-sm font-semibold tracking-tight">
                  {t("nav.brand")}
                </Link>
              </div>
              <button
                type="button"
                onClick={openSearch}
                title={t("search.trigger")}
                aria-label={t("search.trigger")}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <IconSearch className="text-[16px]" />
              </button>
            </header>
          )}
          <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
        </div>
      </div>

      <SearchModal
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        sessions={sessions}
        projects={projects}
      />
    </ChatShellContext.Provider>
  );
}
