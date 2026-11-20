"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { listOwnerPapers, searchConversations } from "@/lib/agent";
import type { AgentProject, AgentSession, RecalledTurn, SessionPaper } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import { IconBook, IconChat, IconFolder, IconHistory, IconSearch, IconX } from "@/components/icons";

interface Props {
  open: boolean;
  onClose: () => void;
  sessions: AgentSession[];
  projects: AgentProject[];
}

interface Entry {
  key: string;
  section: "projects" | "chats" | "papers" | "turns";
  href: string;
  title: string;
  subtitle?: string;
}

/**
 * Global command & recall palette (`⌘K` / `Ctrl+K`).
 *
 * Filters recent conversations, projects and papers locally on the first
 * keystroke, and queries server-side conversation recall after a short pause
 * so earlier Q&A turns can be jumped straight back to.
 */
export default function SearchModal({ open, onClose, sessions, projects }: Props) {
  const { t } = useTranslation();
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState<SessionPaper[]>([]);
  const [turns, setTurns] = useState<RecalledTurn[]>([]);
  const [searching, setSearching] = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setTurns([]);
    setActiveIdx(0);
    const frame = requestAnimationFrame(() => inputRef.current?.focus());
    listOwnerPapers(80)
      .then((res) => setPapers(res.papers))
      .catch(() => {});
    return () => cancelAnimationFrame(frame);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const q = query.trim();
    if (!q) {
      setTurns([]);
      setSearching(false);
      return;
    }
    let cancelled = false;
    setSearching(true);
    const timer = setTimeout(() => {
      searchConversations(q)
        .then((res) => {
          if (!cancelled) setTurns(res.turns.slice(0, 6));
        })
        .catch(() => {
          if (!cancelled) setTurns([]);
        })
        .finally(() => {
          if (!cancelled) setSearching(false);
        });
    }, 220);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [open, query]);

  const entries = useMemo<Entry[]>(() => {
    const q = query.trim().toLowerCase();
    const list: Entry[] = [];

    const matchedProjects = q
      ? projects.filter(
          (p) =>
            p.title.toLowerCase().includes(q) ||
            (p.description || "").toLowerCase().includes(q),
        )
      : projects.slice(0, 4);
    for (const p of matchedProjects.slice(0, 4)) {
      list.push({
        key: `proj-${p.project_id}`,
        section: "projects",
        href: `/projects/${p.project_id}`,
        title: p.title || t("project.untitled"),
        subtitle: p.description || undefined,
      });
    }

    const matchedSessions = q
      ? sessions.filter((s) => (s.title || "").toLowerCase().includes(q))
      : sessions.slice(0, 6);
    for (const s of matchedSessions.slice(0, 6)) {
      list.push({
        key: `sess-${s.session_id}`,
        section: "chats",
        href: `/chat/${s.session_id}`,
        title: s.title || t("chat.sessions.untitled"),
      });
    }

    if (q) {
      const matchedPapers = papers.filter(
        (p) =>
          (p.title || "").toLowerCase().includes(q) ||
          (p.venue || "").toLowerCase().includes(q),
      );
      for (const p of matchedPapers.slice(0, 5)) {
        const href = p.session_id
          ? `/chat/${p.session_id}`
          : p.paper_id
            ? `/chat?paper=${p.paper_id}`
            : "/library";
        list.push({
          key: `paper-${p.literature_id || p.paper_id}`,
          section: "papers",
          href,
          title: p.title || t("chat.papers.untitled"),
          subtitle: [p.venue, p.year_known ? String(p.year) : ""].filter(Boolean).join(" · ") || undefined,
        });
      }

      for (const turn of turns) {
        list.push({
          key: `turn-${turn.run_id}`,
          section: "turns",
          href: `/chat/${turn.session_id}?run=${turn.run_id}`,
          title: turn.question,
          subtitle: `${turn.session_title || t("chat.sessions.untitled")} · ${turn.answer_excerpt}`,
        });
      }
    }

    return list;
  }, [query, projects, sessions, papers, turns, t]);

  useEffect(() => {
    setActiveIdx(0);
  }, [entries.length, query]);

  const go = useCallback(
    (href: string) => {
      onClose();
      router.push(href);
    },
    [onClose, router],
  );

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowDown" && entries.length > 0) {
        e.preventDefault();
        setActiveIdx((i) => (i + 1) % entries.length);
      } else if (e.key === "ArrowUp" && entries.length > 0) {
        e.preventDefault();
        setActiveIdx((i) => (i - 1 + entries.length) % entries.length);
      } else if (e.key === "Enter" && !e.isComposing && entries[activeIdx]) {
        e.preventDefault();
        go(entries[activeIdx].href);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, entries, activeIdx, go, onClose]);

  if (!open || typeof document === "undefined") return null;

  const sections: Entry["section"][] = ["projects", "chats", "papers", "turns"];
  const sectionIcon = {
    projects: IconFolder,
    chats: IconChat,
    papers: IconBook,
    turns: IconHistory,
  } as const;

  let cursor = -1;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]">
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="animate-fade-in fixed inset-0 bg-foreground/25 backdrop-blur-[2px]"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t("search.trigger")}
        className="animate-fade-in relative z-10 flex max-h-[72vh] w-full max-w-xl flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl"
      >
        <div className="flex shrink-0 items-center gap-2.5 border-b border-border px-4 py-3">
          <IconSearch className="shrink-0 text-[16px] text-muted-foreground" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("search.placeholder")}
            className="min-w-0 flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          {searching && (
            <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
          )}
          {query ? (
            <button
              type="button"
              onClick={() => setQuery("")}
              aria-label={t("chat.nav.search_clear")}
              className="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <IconX className="text-[13px]" />
            </button>
          ) : (
            <kbd className="hidden rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground sm:inline">
              ESC
            </kbd>
          )}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {entries.length === 0 && !searching ? (
            <p className="px-3 py-8 text-center text-xs text-muted-foreground">
              {t("chat.nav.search_empty")}
            </p>
          ) : (
            sections.map((sec) => {
              const group = entries.filter((e) => e.section === sec);
              if (group.length === 0) return null;
              const Icon = sectionIcon[sec];
              return (
                <div key={sec} className="mb-2 last:mb-0">
                  <p className="px-2.5 pb-1 pt-1.5 text-xs font-medium text-muted-foreground">
                    {t(`search.section.${sec}`)}
                  </p>
                  {group.map((entry) => {
                    cursor += 1;
                    const idx = cursor;
                    const active = idx === activeIdx;
                    return (
                      <button
                        key={entry.key}
                        type="button"
                        onMouseEnter={() => setActiveIdx(idx)}
                        onClick={() => go(entry.href)}
                        className={`flex w-full items-start gap-2.5 rounded-xl px-2.5 py-2 text-left transition-colors ${
                          active ? "bg-muted text-foreground" : "text-foreground/85 hover:bg-muted/60"
                        }`}
                      >
                        <Icon className="mt-0.5 shrink-0 text-[15px] text-muted-foreground" />
                        <span className="min-w-0 flex-1">
                          <span className="line-clamp-1 block text-[0.8125rem] font-medium">
                            {entry.title}
                          </span>
                          {entry.subtitle && (
                            <span className="mt-0.5 line-clamp-1 block text-xs text-muted-foreground">
                              {entry.subtitle}
                            </span>
                          )}
                        </span>
                      </button>
                    );
                  })}
                </div>
              );
            })
          )}
        </div>

        <div className="hidden shrink-0 items-center justify-between border-t border-border bg-muted/40 px-4 py-2 text-[11px] text-muted-foreground sm:flex">
          <span>{t("search.hint")}</span>
          <span className="font-mono">⌘K</span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
