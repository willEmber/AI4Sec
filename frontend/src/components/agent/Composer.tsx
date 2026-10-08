"use client";

import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { AgentMode, SessionPaper } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import ModePicker from "@/components/agent/ModePicker";
import { AVAILABILITY_DOT } from "@/components/agent/PaperList";
import { IconArrowUp, IconPaperclip, IconStop } from "@/components/icons";

const MAX_CHIPS = 4;

interface Props {
  value: string;
  onChange: (value: string) => void;
  mode: AgentMode;
  onModeChange: (mode: AgentMode) => void;
  onSend: () => void;
  onStop?: () => void;
  /** A turn is running: the send button becomes stop. */
  streaming?: boolean;
  /** The question is on its way to the server. */
  sending?: boolean;
  /** Attach a PDF. Resolves once it is attached; rejects with the reason. */
  onAttach?: (file: File) => Promise<void>;
  /** The papers this turn can read, shown above the text. */
  papers?: SessionPaper[];
  onPickPaper?: (paperId: string) => void;
  /** The model answering. With `models` and `onModelChange` it can be chosen. */
  model?: string;
  models?: string[];
  onModelChange?: (model: string) => void;
  inputRef?: RefObject<HTMLTextAreaElement | null>;
  autoFocus?: boolean;
}

/**
 * The message box: text, an attachment, the mode for this turn, and what the
 * turn will run on and read.
 *
 * Shared by the entry page and a conversation, so starting a conversation and
 * continuing one are the same gesture.
 */
export default function Composer({
  value,
  onChange,
  mode,
  onModeChange,
  onSend,
  onStop,
  streaming = false,
  sending = false,
  onAttach,
  papers = [],
  onPickPaper,
  model = "",
  models,
  onModelChange,
  inputRef,
  autoFocus,
}: Props) {
  const { t } = useTranslation();
  const ownRef = useRef<HTMLTextAreaElement>(null);
  const textareaRef = inputRef ?? ownRef;
  const fileRef = useRef<HTMLInputElement>(null);
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState("");

  // Grows with its text, up to a limit, then scrolls.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [value, textareaRef]);

  const attach = useCallback(
    async (file: File | undefined) => {
      if (!file || !onAttach) return;
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setAttachError(t("upload.drop_error"));
        return;
      }
      setAttaching(true);
      setAttachError("");
      try {
        await onAttach(file);
      } catch (err) {
        setAttachError(err instanceof Error ? err.message : String(err));
      } finally {
        setAttaching(false);
        if (fileRef.current) fileRef.current.value = "";
      }
    },
    [onAttach, t],
  );

  const shown = papers.slice(0, MAX_CHIPS);
  const selectable = models && models.length > 0 && onModelChange;

  return (
    <div>
      <div className="rounded-2xl border border-border bg-card soft-shadow transition-[border-color,box-shadow] focus-within:border-primary/50 focus-within:shadow-[0_0_0_3px_color-mix(in_srgb,var(--primary)_12%,transparent)]">
        {shown.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 px-3 pt-2.5">
            {shown.map((paper) => (
              <button
                key={paper.literature_id}
                type="button"
                disabled={!paper.paper_id || !onPickPaper}
                onClick={() => paper.paper_id && onPickPaper?.(paper.paper_id)}
                title={`${paper.title || t("chat.papers.untitled")} · ${t(`chat.availability.${paper.availability}`)}`}
                className="inline-flex max-w-[14rem] items-center gap-1.5 rounded-md bg-muted px-2 py-1 text-xs text-foreground/85 transition-colors enabled:hover:bg-accent disabled:cursor-default"
              >
                <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${AVAILABILITY_DOT[paper.availability]}`} />
                <span className="truncate">{paper.title || t("chat.papers.untitled")}</span>
              </button>
            ))}
            {papers.length > MAX_CHIPS && (
              <span className="text-xs text-muted-foreground">+{papers.length - MAX_CHIPS}</span>
            )}
          </div>
        )}
        <textarea
          ref={textareaRef}
          value={value}
          autoFocus={autoFocus}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends; Shift+Enter is a newline, as in every chat box.
            // Not while an input method is composing: that Enter picks a candidate.
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              onSend();
            }
          }}
          rows={1}
          placeholder={mode === "auto" ? t("chat.placeholder") : t("chat.placeholder_mode")}
          className="block max-h-60 w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[0.925rem] leading-relaxed text-foreground placeholder:text-muted-foreground focus:outline-none focus-visible:outline-none"
        />
        <div className="flex items-center gap-1.5 px-2.5 pb-2.5 pt-1.5">
          {onAttach && (
            <>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,application/pdf"
                className="hidden"
                onChange={(e) => void attach(e.target.files?.[0])}
              />
              <button
                type="button"
                disabled={attaching}
                onClick={() => fileRef.current?.click()}
                title={t("chat.papers.upload")}
                aria-label={t("chat.papers.upload")}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-60"
              >
                {attaching ? (
                  <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-[1.5px] border-border border-t-primary" />
                ) : (
                  <IconPaperclip className="text-[16px]" />
                )}
              </button>
            </>
          )}
          <div className="no-scrollbar min-w-0 overflow-x-auto">
            <ModePicker value={mode} onChange={onModeChange} disabled={streaming} />
          </div>
          <div className="flex-1" />
          {selectable ? (
            <select
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
              title={t("upload.model_label")}
              className="hidden max-w-[11rem] truncate rounded-md bg-transparent px-1 py-1 text-xs text-muted-foreground hover:text-foreground focus:outline-none sm:block"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            model && (
              <span
                className="hidden max-w-[11rem] truncate px-1 text-xs text-muted-foreground sm:block"
                title={t("upload.model_label")}
              >
                {model}
              </span>
            )
          )}
          {streaming ? (
            <button
              type="button"
              onClick={onStop}
              title={t("chat.stop")}
              className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full border border-border bg-background px-3.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
            >
              <IconStop className="text-[13px]" />
              <span className="hidden sm:inline">{t("chat.stop")}</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={onSend}
              disabled={!value.trim() || sending}
              title={t("chat.send")}
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-35"
            >
              {sending ? (
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground/40 border-t-primary-foreground" />
              ) : (
                <IconArrowUp className="text-[17px]" />
              )}
            </button>
          )}
        </div>
      </div>
      <p className={`mt-1.5 truncate text-center text-xs ${attachError ? "text-destructive" : "text-muted-foreground"}`}>
        {attachError || (mode === "auto" ? t("chat.composer_hint") : t(`chat.mode.${mode}.desc`))}
      </p>
    </div>
  );
}
