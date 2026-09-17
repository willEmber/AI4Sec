"use client";

import { useState } from "react";
import { CITATION_PATTERN, getEvidence } from "@/lib/agent";
import type { Evidence } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";

interface Props {
  role: "user" | "assistant" | "system";
  content: string;
  onJumpToPage?: (paperId: string, page: number) => void;
}

/**
 * One message, with `[ev_...]` rendered as a clickable citation.
 *
 * A citation here is a real reference, not decoration: clicking it fetches the
 * evidence and shows the excerpt the agent actually read, with its page. That
 * is what makes an answer checkable rather than merely plausible.
 */
export default function ChatMessage({ role, content, onJumpToPage }: Props) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-primary-foreground">
          <p className="whitespace-pre-wrap text-[0.9rem] leading-relaxed">{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-[92%] whitespace-pre-wrap text-[0.9rem] leading-[1.75] text-foreground">
      {renderWithCitations(content, onJumpToPage)}
    </div>
  );
}

function renderWithCitations(
  text: string,
  onJumpToPage?: (paperId: string, page: number) => void,
) {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  let key = 0;

  // A fresh regex per call: the exported one is global, so a shared lastIndex
  // would make every other render start mid-string.
  const pattern = new RegExp(CITATION_PATTERN.source, "g");
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      parts.push(<span key={key++}>{text.slice(cursor, match.index)}</span>);
    }
    parts.push(
      <CitationBadge key={key++} evidenceId={match[1]} onJumpToPage={onJumpToPage} />,
    );
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push(<span key={key++}>{text.slice(cursor)}</span>);
  return parts;
}

function CitationBadge({
  evidenceId,
  onJumpToPage,
}: {
  evidenceId: string;
  onJumpToPage?: (paperId: string, page: number) => void;
}) {
  const { t } = useTranslation();
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);

  async function toggle() {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (evidence || failed) return;
    try {
      setEvidence(await getEvidence(evidenceId));
    } catch {
      setFailed(true);
    }
  }

  const page = evidence?.locator.page_label;

  return (
    <span className="relative inline-block">
      <button
        type="button"
        onClick={toggle}
        title={evidenceId}
        className="mx-[1px] inline-flex items-baseline rounded bg-accent px-1.5 py-[1px] align-baseline text-[0.7rem] font-medium text-accent-foreground transition-colors hover:bg-primary hover:text-primary-foreground"
      >
        {page ? t("chat.cite.page", { page }) : t("chat.cite.source")}
      </button>

      {open && (
        <span className="absolute bottom-full left-0 z-30 mb-1.5 block w-80 rounded-xl border border-border bg-card p-3 text-xs soft-shadow">
          {failed && (
            <span className="block text-muted-foreground">{t("chat.cite.failed")}</span>
          )}
          {!failed && !evidence && (
            <span className="block text-muted-foreground">{t("chat.cite.loading")}</span>
          )}
          {evidence && (
            <>
              <span className="mb-1 block font-medium text-foreground">
                {evidence.literature?.title || evidence.paper_id.slice(0, 12)}
              </span>
              <span className="mb-2 block text-muted-foreground">
                {evidence.locator.section_path || "—"}
                {page ? ` · ${t("chat.cite.page", { page })}` : ""}
                {evidence.source_level !== "fulltext"
                  ? ` · ${t(`chat.cite.level.${evidence.source_level}`)}`
                  : ""}
              </span>
              <span className="block max-h-40 overflow-y-auto whitespace-pre-wrap border-l-2 border-border pl-2 leading-relaxed text-muted-foreground">
                {evidence.quote}
              </span>
              {evidence.parse_status && !evidence.parse_status.is_current_version && (
                <span className="mt-2 block text-[0.7rem] text-accent-foreground">
                  {evidence.parse_status.still_present
                    ? t("chat.cite.stale_present")
                    : t("chat.cite.stale_missing")}
                </span>
              )}
              {page && evidence.paper_id && onJumpToPage && (
                <button
                  type="button"
                  onClick={() => onJumpToPage(evidence.paper_id, page)}
                  className="mt-2 block font-medium text-primary hover:underline"
                >
                  {t("chat.cite.jump", { page })}
                </button>
              )}
            </>
          )}
        </span>
      )}
    </span>
  );
}
