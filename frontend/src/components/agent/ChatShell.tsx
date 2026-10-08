"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { listSessions } from "@/lib/agent";
import type { AgentProject, AgentSession } from "@/lib/agent";
import { usePersistentState } from "@/hooks/usePersistentState";
import ChatNav from "@/components/agent/ChatNav";

interface ChatShellValue {
  /** The reader's conversations, as the sidebar lists them. */
  sessions: AgentSession[];
  /** Refetch them: a turn ended, a title changed, a conversation moved or went. */
  refreshSessions: () => void;
  /** The open conversation's project, so the sidebar can narrow to it. */
  setCurrentProject: (project: AgentProject | null) => void;
  /** Open the sidebar where it is a drawer (narrow screens). */
  openNav: () => void;
}

const ChatShellContext = createContext<ChatShellValue>({
  sessions: [],
  refreshSessions: () => {},
  setCurrentProject: () => {},
  openNav: () => {},
});

export function useChatShell(): ChatShellValue {
  return useContext(ChatShellContext);
}

/**
 * The frame every `/chat` page sits in: the conversation list on the left, the
 * page on the right.
 *
 * It lives in the route's layout, so moving between conversations keeps the
 * list where it was instead of rebuilding it — and it is where the pages tell
 * the list that something changed.
 */
export default function ChatShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const currentSessionId = pathname.startsWith("/chat/") ? pathname.split("/")[2] ?? "" : "";

  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [currentProject, setCurrentProject] = useState<AgentProject | null>(null);
  const [scope, setScope] = useState<"all" | "project">("all");
  const [refreshToken, setRefreshToken] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [collapsed, setCollapsed] = usePersistentState("scholar.chat.nav_collapsed", false, false);

  const projectId = currentProject?.project_id ?? "";
  const scoped = scope === "project" && projectId;

  useEffect(() => {
    let cancelled = false;
    listSessions(scoped ? projectId : undefined)
      .then((data) => {
        if (!cancelled) setSessions(data.sessions);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, [scoped, projectId, refreshToken]);

  // Leaving a project's conversation leaves nothing to narrow to.
  useEffect(() => {
    if (!projectId) setScope("all");
  }, [projectId]);

  // A drawer that stays open over the page it just navigated to is in the way.
  useEffect(() => {
    setDrawerOpen(false);
  }, [pathname]);

  const refreshSessions = useCallback(() => setRefreshToken((n) => n + 1), []);
  const openNav = useCallback(() => setDrawerOpen(true), []);

  const value = useMemo(
    () => ({ sessions, refreshSessions, setCurrentProject, openNav }),
    [sessions, refreshSessions, openNav],
  );

  const nav = (asDrawer: boolean) => (
    <ChatNav
      sessions={sessions}
      loaded={loaded}
      currentSessionId={currentSessionId}
      project={currentProject}
      scope={scope}
      onScope={setScope}
      onChanged={refreshSessions}
      collapsed={!asDrawer && collapsed}
      onToggleCollapse={asDrawer ? () => setDrawerOpen(false) : () => setCollapsed((v) => !v)}
      asDrawer={asDrawer}
    />
  );

  return (
    <ChatShellContext.Provider value={value}>
      <div className="flex h-dvh overflow-hidden">
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
        <div className="min-w-0 flex-1">{children}</div>
      </div>
    </ChatShellContext.Provider>
  );
}
