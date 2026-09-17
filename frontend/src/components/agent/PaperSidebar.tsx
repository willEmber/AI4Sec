"use client";

import Link from "next/link";
import { useTranslation } from "@/lib/i18n";
import type { AgentSession, SessionPaper } from "@/lib/agent";

// Readable and not-yet-readable must be distinguishable at a glance, not only
// by reading the label: which papers an answer could actually have drawn on is
// the first thing to check when several are attached.
const AVAILABILITY_CLASS: Record<SessionPaper["availability"], string> = {
  candidate: "bg-muted text-muted-foreground",
  downloading: "bg-accent text-accent-foreground",
  pdf_ready: "bg-accent text-accent-foreground",
  parsed: "bg-muted text-success",
  unavailable: "bg-muted text-destructive",
};

interface Props {
  papers: SessionPaper[];
  sessions: AgentSession[];
  currentSessionId: string;
  activePaperId: string;
  onSelectPaper: (paperId: string) => void;
}

/**
 * The papers this session can read, plus the reader's other sessions.
 *
 * `availability` is shown rather than hidden: "only the abstract is available"
 * changes how an answer should be read, so it belongs on screen next to the
 * paper it qualifies.
 */
export default function PaperSidebar({
  papers,
  sessions,
  currentSessionId,
  activePaperId,
  onSelectPaper,
}: Props) {
  const { t } = useTranslation();

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-muted/40">
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {t("chat.papers.heading")}
        </h2>
      </div>

      <div className="max-h-[50%] overflow-y-auto p-2">
        {papers.length === 0 && (
          <p className="px-2 py-3 text-xs leading-relaxed text-muted-foreground">
            {t("chat.papers.empty")}
          </p>
        )}

        {papers.map((paper) => (
          <button
            key={paper.literature_id}
            type="button"
            disabled={!paper.paper_id}
            onClick={() => paper.paper_id && onSelectPaper(paper.paper_id)}
            className={`mb-1 w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
              activePaperId && activePaperId === paper.paper_id
                ? "bg-card soft-shadow"
                : "hover:bg-card/70"
            } ${paper.paper_id ? "" : "cursor-default opacity-70"}`}
          >
            <span className="line-clamp-2 block text-xs font-medium leading-snug text-foreground">
              {paper.title || t("chat.papers.untitled")}
            </span>
            <span className="mt-1.5 flex items-center gap-1.5">
              <span
                className={`rounded px-1.5 py-[1px] text-[0.65rem] ${AVAILABILITY_CLASS[paper.availability]}`}
              >
                {t(`chat.availability.${paper.availability}`)}
              </span>
              {paper.year_known && (
                <span className="text-[0.65rem] text-muted-foreground">{paper.year}</span>
              )}
            </span>
          </button>
        ))}
      </div>

      <div className="border-t border-border px-4 py-3">
        <h2 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {t("chat.sessions.heading")}
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {sessions.length === 0 && (
          <p className="px-2 py-3 text-xs text-muted-foreground">
            {t("chat.sessions.empty")}
          </p>
        )}
        {sessions.map((session) => (
          <Link
            key={session.session_id}
            href={`/chat/${session.session_id}`}
            className={`mb-0.5 block truncate rounded-lg px-3 py-2 text-xs transition-colors ${
              session.session_id === currentSessionId
                ? "bg-card font-medium text-foreground soft-shadow"
                : "text-muted-foreground hover:bg-card/70 hover:text-foreground"
            }`}
          >
            {session.title || t("chat.sessions.untitled")}
          </Link>
        ))}
      </div>
    </aside>
  );
}
