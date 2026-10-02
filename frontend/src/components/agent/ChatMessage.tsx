"use client";

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { CITATION_PATTERN, getEvidence } from "@/lib/agent";
import type { Evidence } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import { IconCheck, IconCopy, IconExternal } from "@/components/icons";

interface Props {
  role: "user" | "assistant" | "system";
  content: string;
  onJumpToPage?: (paperId: string, page: number) => void;
  /** Text still arriving: shows a caret after the last block. */
  streaming?: boolean;
}

/**
 * One message. An assistant message is Markdown — headings, lists, tables,
 * formulas — with each `[ev_...]` rendered as a numbered, clickable citation.
 *
 * A citation here is a real reference, not decoration: clicking it fetches the
 * evidence and shows the excerpt the agent actually read, with its page. That
 * is what makes an answer checkable rather than merely plausible.
 */
function ChatMessage({ role, content, onJumpToPage, streaming }: Props) {
  // Numbered by first appearance, so the same source keeps one number
  // throughout an answer.
  const numbers = useMemo(() => {
    const map = new Map<string, number>();
    const pattern = new RegExp(CITATION_PATTERN.source, "g");
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(content)) !== null) {
      if (!map.has(match[1])) map.set(match[1], map.size + 1);
    }
    return map;
  }, [content]);

  const renderEvidence = useCallback(
    (evidenceId: string) => (
      <CitationBadge
        evidenceId={evidenceId}
        number={numbers.get(evidenceId) ?? 0}
        onJumpToPage={onJumpToPage}
      />
    ),
    [numbers, onJumpToPage],
  );

  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-foreground">
          <p className="whitespace-pre-wrap break-words text-[0.925rem] leading-relaxed">
            {content}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-w-0 ${streaming ? "streaming-caret" : ""}`}>
      <MarkdownRenderer content={content} variant="chat" renderEvidence={renderEvidence} />
    </div>
  );
}

// Memoised: while a turn streams, the page re-renders on every chunk, and
// re-parsing every earlier answer's Markdown and formulas each time is waste.
export default memo(ChatMessage);

/** Copies a message's Markdown source. */
export function CopyButton({ text, className = "" }: { text: string; className?: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard denied (insecure origin, permissions): nothing to undo.
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={() => void copy()}
      title={copied ? t("chat.copied") : t("chat.copy")}
      className={`inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[0.7rem] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground ${className}`}
    >
      {copied ? <IconCheck className="text-[13px] text-success" /> : <IconCopy className="text-[13px]" />}
      <span>{copied ? t("chat.copied") : t("chat.copy")}</span>
    </button>
  );
}

function hostOf(url: string): string {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, "") : "";
  } catch {
    return "";
  }
}

const POPOVER_WIDTH = 336;
const POPOVER_GAP = 6;

interface Placement {
  left: number;
  top?: number;
  bottom?: number;
}

function CitationBadge({
  evidenceId,
  number,
  onJumpToPage,
}: {
  evidenceId: string;
  number: number;
  onJumpToPage?: (paperId: string, page: number) => void;
}) {
  const { t } = useTranslation();
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [failed, setFailed] = useState(false);
  const [placement, setPlacement] = useState<Placement | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLSpanElement>(null);
  const open = placement !== null;

  // The popover is portalled and fixed, so it is never clipped by the scrolling
  // conversation or a table's overflow box. It opens toward the side of the
  // viewport with room for it.
  const place = useCallback(() => {
    const rect = buttonRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const width = Math.min(POPOVER_WIDTH, window.innerWidth - 16);
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
    return rect.top > 300
      ? { left, bottom: window.innerHeight - rect.top + POPOVER_GAP }
      : { left, top: rect.bottom + POPOVER_GAP };
  }, []);

  const toggle = useCallback(async () => {
    if (open) {
      setPlacement(null);
      return;
    }
    setPlacement(place());
    if (evidence || failed) return;
    try {
      setEvidence(await getEvidence(evidenceId));
    } catch {
      setFailed(true);
    }
  }, [open, place, evidence, failed, evidenceId]);

  // Close on a click elsewhere, Escape, or when the page under it moves.
  useEffect(() => {
    if (!open) return;
    const onPointer = (e: MouseEvent) => {
      const target = e.target as Node;
      if (buttonRef.current?.contains(target) || popoverRef.current?.contains(target)) return;
      setPlacement(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPlacement(null);
    };
    const onMove = (e: Event) => {
      // Scrolling inside the popover's own quote is not the page moving. A
      // resize's target is the window, which is not a Node.
      if (e.target instanceof Node && popoverRef.current?.contains(e.target)) return;
      setPlacement(null);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [open]);

  // Keep it on screen once its real height is known.
  useLayoutEffect(() => {
    const el = popoverRef.current;
    if (!open || !el || placement?.top === undefined) return;
    const overflow = el.getBoundingClientRect().bottom - window.innerHeight + 8;
    if (overflow > 0) el.style.transform = `translateY(-${overflow}px)`;
  }, [open, placement, evidence]);

  const page = evidence?.locator.page_label;
  const levelLabel =
    evidence && evidence.source_level !== "fulltext"
      ? t(`chat.cite.level.${evidence.source_level}`)
      : "";

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => void toggle()}
        title={page ? t("chat.cite.page", { page }) : t("chat.cite.source")}
        aria-expanded={open}
        className={`mx-[2px] inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-[1px] items-center justify-center rounded-full px-1 align-middle text-[0.65rem] font-semibold tabular-nums transition-colors ${
          open
            ? "bg-primary text-primary-foreground"
            : "bg-accent text-accent-foreground hover:bg-primary hover:text-primary-foreground"
        }`}
      >
        {number || "?"}
      </button>

      {open &&
        placement &&
        createPortal(
          <span
            ref={popoverRef}
            role="dialog"
            style={{
              position: "fixed",
              left: placement.left,
              top: placement.top,
              bottom: placement.bottom,
              width: Math.min(POPOVER_WIDTH, window.innerWidth - 16),
            }}
            className="animate-fade-in z-50 block rounded-xl border border-border bg-card p-3.5 text-xs shadow-xl"
          >
            {failed && <span className="block text-muted-foreground">{t("chat.cite.failed")}</span>}
            {!failed && !evidence && (
              <span className="flex items-center gap-2 text-muted-foreground">
                <span className="inline-block h-3 w-3 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
                {t("chat.cite.loading")}
              </span>
            )}
            {evidence && (
              <>
                <span className="mb-1.5 flex items-start gap-2">
                  <span className="mt-[1px] inline-flex h-4 min-w-4 shrink-0 items-center justify-center rounded-full bg-accent px-1 text-[0.6rem] font-semibold text-accent-foreground">
                    {number}
                  </span>
                  <span className="line-clamp-2 font-medium leading-snug text-foreground">
                    {evidence.literature?.title ||
                      hostOf(evidence.source_url) ||
                      evidence.paper_id.slice(0, 12)}
                  </span>
                </span>
                <span className="mb-2 flex flex-wrap items-center gap-1.5 text-[0.7rem] text-muted-foreground">
                  {evidence.locator.section_path && (
                    <span className="max-w-full truncate">{evidence.locator.section_path}</span>
                  )}
                  {page && (
                    <span className="rounded bg-muted px-1.5 py-[1px]">
                      {t("chat.cite.page", { page })}
                    </span>
                  )}
                  {levelLabel && (
                    <span className="rounded bg-muted px-1.5 py-[1px]">{levelLabel}</span>
                  )}
                </span>
                <span className="block max-h-48 overflow-y-auto whitespace-pre-wrap border-l-2 border-primary/40 pl-2.5 leading-relaxed text-foreground/80">
                  {evidence.quote}
                </span>
                {/* Only a PDF passage has a parse version to be out of date against. */}
                {evidence.paper_id &&
                  evidence.parse_status &&
                  !evidence.parse_status.is_current_version && (
                  <span className="mt-2 block text-[0.7rem] text-accent-foreground">
                    {evidence.parse_status.still_present
                      ? t("chat.cite.stale_present")
                      : t("chat.cite.stale_missing")}
                  </span>
                )}
                <span className="mt-2.5 flex items-center gap-3">
                  {page && evidence.paper_id && onJumpToPage && (
                    <button
                      type="button"
                      onClick={() => {
                        onJumpToPage(evidence.paper_id, page);
                        setPlacement(null);
                      }}
                      className="font-medium text-primary hover:underline"
                    >
                      {t("chat.cite.jump", { page })}
                    </button>
                  )}
                  {evidence.source_url && (
                    <a
                      href={evidence.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 font-medium text-primary hover:underline"
                    >
                      <IconExternal className="text-[12px]" />
                      {t("chat.cite.open_source")}
                    </a>
                  )}
                </span>
              </>
            )}
          </span>,
          document.body,
        )}
    </>
  );
}
