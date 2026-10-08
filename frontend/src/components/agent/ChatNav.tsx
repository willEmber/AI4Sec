"use client";

import { useCallback, useMemo, useState } from "react";
import type { FormEvent } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { deleteSession, describeApiError, searchConversations, updateSession } from "@/lib/agent";
import type { AgentProject, AgentSession, RecalledTurn } from "@/lib/agent";
import { parseServerTime } from "@/lib/agentEvents";
import { LanguageToggle, useTranslation } from "@/lib/i18n";
import { dateBucket } from "@/lib/turns";
import type { DateBucket } from "@/lib/turns";
import { AuthMenu } from "@/components/AuthMenu";
import Menu, { MenuItem } from "@/components/Menu";
import {
  IconBook,
  IconDocument,
  IconFolder,
  IconMore,
  IconPanelLeftClose,
  IconPanelLeftOpen,
  IconPencil,
  IconPlus,
  IconSearch,
  IconTable,
  IconTrash,
  IconX,
} from "@/components/icons";

interface Props {
  sessions: AgentSession[];
  /** False until the first list has arrived, so "no conversations" is not shown early. */
  loaded: boolean;
  currentSessionId: string;
  /** The open conversation's project, when it has one. */
  project: AgentProject | null;
  scope: "all" | "project";
  onScope: (scope: "all" | "project") => void;
  /** A conversation was renamed or deleted here. */
  onChanged: () => void;
  /** Folded to a narrow icon rail. */
  collapsed: boolean;
  /** Fold / unfold; as a drawer, close it. */
  onToggleCollapse: () => void;
  asDrawer?: boolean;
}

const BUCKETS: DateBucket[] = ["today", "week", "earlier"];

/**
 * The conversation list: start one, find one, and the way out to the rest of
 * the app.
 *
 * Papers, sources and memory are not here. They belong to the conversation
 * that is open and sit beside it; this column is only for moving between
 * conversations, which is why it can stay put while the page changes.
 */
export default function ChatNav({
  sessions,
  loaded,
  currentSessionId,
  project,
  scope,
  onScope,
  onChanged,
  collapsed,
  onToggleCollapse,
  asDrawer = false,
}: Props) {
  const { t } = useTranslation();
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RecalledTurn[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [renaming, setRenaming] = useState("");
  const [titleDraft, setTitleDraft] = useState("");
  const [error, setError] = useState("");

  const scopedProjectId = scope === "project" && project ? project.project_id : "";
  const newChatHref = scopedProjectId ? `/chat?project=${scopedProjectId}` : "/chat";

  const grouped = useMemo(() => {
    const now = Date.now();
    const map = new Map<DateBucket, AgentSession[]>();
    for (const session of sessions) {
      const bucket = dateBucket(parseServerTime(session.updated_at), now);
      map.set(bucket, [...(map.get(bucket) ?? []), session]);
    }
    return map;
  }, [sessions]);

  const search = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const q = query.trim();
      if (!q) {
        setResults(null);
        return;
      }
      setSearching(true);
      setError("");
      try {
        const data = await searchConversations(q, scopedProjectId || undefined);
        setResults(data.turns);
      } catch (err) {
        setError(describeApiError(err, t));
      } finally {
        setSearching(false);
      }
    },
    [query, scopedProjectId, t],
  );

  const clearSearch = useCallback(() => {
    setQuery("");
    setResults(null);
  }, []);

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
        // The conversation on screen is the one that is gone.
        if (sessionId === currentSessionId) router.push("/chat");
      } catch (err) {
        setError(describeApiError(err, t));
      }
    },
    [currentSessionId, onChanged, router, t],
  );

  if (collapsed) {
    const rail =
      "flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-card hover:text-foreground";
    return (
      <aside className="flex h-full w-12 shrink-0 flex-col items-center gap-1 border-r border-border bg-muted/40 py-2.5">
        <button type="button" onClick={onToggleCollapse} title={t("chat.sidebar.expand")} className={rail}>
          <IconPanelLeftOpen className="text-[17px]" />
        </button>
        <Link href={newChatHref} title={t("chat.nav.new")} className={rail}>
          <IconPlus className="text-[17px]" />
        </Link>
        <button type="button" onClick={onToggleCollapse} title={t("chat.nav.search")} className={rail}>
          <IconSearch className="text-[16px]" />
        </button>
      </aside>
    );
  }

  const outLinks: [string, string, typeof IconFolder][] = [
    ["/projects", t("nav.projects"), IconFolder],
    ["/upload", t("nav.upload"), IconDocument],
    ["/compare", t("nav.compare"), IconTable],
    ["/library", t("nav.library"), IconBook],
  ];

  return (
    <aside
      className={`flex h-full shrink-0 flex-col border-r border-border ${
        asDrawer ? "w-72 max-w-[85vw] bg-background" : "w-64 bg-muted/40"
      }`}
    >
      <div className="flex h-12 shrink-0 items-center justify-between pl-4 pr-2">
        {/* A plain link, like the rest of the top-level navigation: a full load. */}
        <a href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <Image src="/scholar.png" alt="" width={24} height={24} className="h-6 w-6 rounded-md object-contain" />
          <span className="text-[15px]">{t("nav.brand")}</span>
        </a>
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

      <div className="shrink-0 space-y-2 px-2.5 pb-2">
        <Link
          href={newChatHref}
          className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm font-medium text-foreground soft-shadow transition-colors hover:border-foreground/20"
        >
          <IconPlus className="text-[15px] text-primary" />
          {t("chat.nav.new")}
        </Link>
        <form onSubmit={(e) => void search(e)} className="relative">
          <IconSearch className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[14px] text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("chat.nav.search")}
            aria-label={t("chat.nav.search")}
            enterKeyHint="search"
            className="w-full rounded-lg border border-transparent bg-card/70 py-1.5 pl-8 pr-7 text-[0.8125rem] text-foreground placeholder:text-muted-foreground focus:border-border focus:bg-card focus:outline-none"
          />
          {(query || results) && (
            <button
              type="button"
              onClick={clearSearch}
              title={t("chat.nav.search_clear")}
              aria-label={t("chat.nav.search_clear")}
              className="absolute right-1.5 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded text-muted-foreground hover:text-foreground"
            >
              <IconX className="text-[12px]" />
            </button>
          )}
        </form>
        {project && (
          <div className="flex rounded-lg bg-card/70 p-0.5 text-xs">
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
        {error && <p className="px-1 text-xs text-destructive">{error}</p>}
      </div>

      <div className="relative min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {results !== null || searching ? (
          <>
            {searching && <p className="px-2 py-2 text-xs text-muted-foreground">{t("chat.cite.loading")}</p>}
            {!searching && results?.length === 0 && (
              <p className="px-2 py-2 text-xs text-muted-foreground">{t("chat.nav.search_empty")}</p>
            )}
            {(results ?? []).map((turn) => (
              <Link
                key={turn.run_id}
                href={`/chat/${turn.session_id}?run=${turn.run_id}`}
                className="mb-1 block rounded-lg px-2.5 py-2 transition-colors hover:bg-card"
              >
                <span className="block truncate text-xs text-muted-foreground">
                  {turn.session_title || t("chat.sessions.untitled")}
                </span>
                <span className="mt-0.5 line-clamp-1 text-[0.8125rem] font-medium text-foreground">
                  {turn.question}
                </span>
                <span className="mt-0.5 line-clamp-2 text-xs leading-snug text-muted-foreground">
                  {turn.answer_excerpt}
                </span>
              </Link>
            ))}
          </>
        ) : (
          <>
            {loaded && sessions.length === 0 && (
              <p className="px-2 py-2 text-xs text-muted-foreground">{t("chat.sessions.empty")}</p>
            )}
            {BUCKETS.filter((bucket) => grouped.has(bucket)).map((bucket) => (
              <section key={bucket} className="mb-2">
                <h2 className="px-2.5 pb-1 pt-2 text-xs text-muted-foreground/80">
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
          </>
        )}
      </div>

      <div className="shrink-0 border-t border-border px-2 py-2">
        <div className="grid grid-cols-2 gap-0.5">
          {outLinks.map(([href, label, Icon]) => (
            <a
              key={href}
              href={href}
              className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
            >
              <Icon className="shrink-0 text-[14px]" />
              <span className="truncate">{label}</span>
            </a>
          ))}
        </div>
        <div className="mt-2 flex items-center gap-2 px-1">
          <AuthMenu placement="above" />
          <div className="flex-1" />
          <LanguageToggle />
        </div>
      </div>
    </aside>
  );
}
