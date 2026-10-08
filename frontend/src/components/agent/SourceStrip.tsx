"use client";

import { useCallback, useMemo, useState } from "react";
import type { Evidence } from "@/lib/agent";
import { loadEvidenceMany } from "@/lib/evidenceCache";
import { useTranslation } from "@/lib/i18n";
import { citationNumbers } from "@/lib/turns";
import { CopyButton } from "@/components/agent/ChatMessage";
import { IconChevronRight, IconExternal, IconQuote } from "@/components/icons";

interface Props {
  /** The answer's Markdown: what is copied, and where the citations are read from. */
  content: string;
  onJumpToPage?: (paperId: string, page: number) => void;
}

export function hostOf(url: string): string {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, "") : "";
  } catch {
    return "";
  }
}

/** What a piece of evidence is from, in a few words. */
export function evidenceTitle(evidence: Evidence): string {
  return evidence.literature?.title || hostOf(evidence.source_url) || evidence.paper_id.slice(0, 12);
}

/**
 * The row under an answer: copy it, and see everything it cited in one list.
 *
 * A citation badge shows one source at a time and closes when the page moves;
 * checking an answer usually means looking at all of them. The numbers are the
 * badges' numbers, so the list reads as the answer's reference list.
 */
export default function SourceStrip({ content, onJumpToPage }: Props) {
  const { t } = useTranslation();
  const numbers = useMemo(() => citationNumbers(content), [content]);
  const [open, setOpen] = useState(false);
  const [sources, setSources] = useState<Map<string, Evidence | null> | null>(null);

  const toggle = useCallback(async () => {
    setOpen((v) => !v);
    if (sources) return;
    setSources(await loadEvidenceMany([...numbers.keys()]));
  }, [sources, numbers]);

  return (
    <div>
      <div className="flex items-center gap-1 text-muted-foreground">
        <CopyButton text={content} />
        {numbers.size > 0 && (
          <button
            type="button"
            onClick={() => void toggle()}
            aria-expanded={open}
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-xs transition-colors hover:bg-muted hover:text-foreground"
          >
            <IconQuote className="text-[13px]" />
            {t("chat.sources.count", { count: numbers.size })}
            <IconChevronRight className={`text-[12px] transition-transform ${open ? "rotate-90" : ""}`} />
          </button>
        )}
      </div>

      {open && (
        <ol className="animate-fade-in mt-1.5 space-y-0.5">
          {!sources && <li className="px-1.5 py-1 text-xs text-muted-foreground">{t("chat.cite.loading")}</li>}
          {sources &&
            [...numbers].map(([evidenceId, number]) => (
              <SourceRow
                key={evidenceId}
                number={number}
                evidence={sources.get(evidenceId) ?? null}
                onJumpToPage={onJumpToPage}
              />
            ))}
        </ol>
      )}
    </div>
  );
}

function SourceRow({
  number,
  evidence,
  onJumpToPage,
}: {
  number: number;
  evidence: Evidence | null;
  onJumpToPage?: (paperId: string, page: number) => void;
}) {
  const { t } = useTranslation();
  const badge = (
    <span className="mt-[1px] inline-flex h-[1.15rem] min-w-[1.15rem] shrink-0 items-center justify-center rounded-full bg-accent px-1 text-[0.68rem] font-semibold tabular-nums text-accent-foreground">
      {number}
    </span>
  );
  if (!evidence) {
    return (
      <li className="flex items-start gap-2 px-1.5 py-1 text-xs text-muted-foreground">
        {badge}
        {t("chat.cite.failed")}
      </li>
    );
  }

  const page = evidence.locator.page_label;
  const body = (
    <>
      {badge}
      <span className="min-w-0 flex-1">
        <span className="line-clamp-1 text-foreground/90">{evidenceTitle(evidence)}</span>
        <span className="line-clamp-1 text-muted-foreground">{evidence.quote}</span>
      </span>
      <span className="flex shrink-0 items-center gap-1.5 text-[0.7rem] text-muted-foreground">
        {evidence.source_level !== "fulltext" && (
          <span className="rounded bg-muted px-1.5 py-[1px]">
            {t(`chat.cite.level.${evidence.source_level}`)}
          </span>
        )}
        {page ? t("chat.cite.page", { page }) : evidence.source_url ? <IconExternal className="text-[12px]" /> : null}
      </span>
    </>
  );
  const rowClass =
    "flex w-full items-start gap-2 rounded-lg px-1.5 py-1.5 text-left text-xs leading-snug transition-colors hover:bg-muted/70";

  if (page && evidence.paper_id && onJumpToPage) {
    return (
      <li>
        <button type="button" onClick={() => onJumpToPage(evidence.paper_id, page)} className={rowClass}>
          {body}
        </button>
      </li>
    );
  }
  if (evidence.source_url) {
    return (
      <li>
        <a href={evidence.source_url} target="_blank" rel="noopener noreferrer" className={rowClass}>
          {body}
        </a>
      </li>
    );
  }
  return <li className={rowClass}>{body}</li>;
}
