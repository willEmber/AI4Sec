"use client";

import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { MAX_MESSAGE_CHARS } from "@/lib/agent";
import type { AgentMode, AgentProject, SessionPaper } from "@/lib/agent";
import { canSubmitTurn } from "@/lib/libraryHub";
import { useTranslation } from "@/lib/i18n";
import Menu from "@/components/Menu";
import ModePicker, { MODE_ICONS } from "@/components/agent/ModePicker";
import { AVAILABILITY_DOT } from "@/components/agent/PaperList";
import {
  IconArrowUp,
  IconCheck,
  IconChevronDown,
  IconFolder,
  IconPaperclip,
  IconStop,
  IconX,
} from "@/components/icons";

const MAX_CHIPS = 6;
const REPORT_MODES: Exclude<AgentMode, "auto">[] = ["snap", "lens", "sphere"];

interface Props {
  value: string;
  onChange: (value: string) => void;
  mode: AgentMode;
  onModeChange: (mode: AgentMode) => void;
  onSend: () => void;
  /** Run one of the three whole-paper report modes with a single click. */
  onQuickMode?: (mode: Exclude<AgentMode, "auto">) => void;
  onStop?: () => void;
  streaming?: boolean;
  sending?: boolean;
  onAttach?: (file: File) => Promise<void>;
  papers?: SessionPaper[];
  onPickPaper?: (paperId: string) => void;
  /** Allow detaching a staged paper chip on the entry page before session creation. */
  onRemovePaper?: (paperId: string) => void;
  /** Optional project selector for filing a new conversation on the entry page. */
  projects?: AgentProject[];
  projectId?: string;
  onProjectChange?: (projectId: string) => void;
  model?: string;
  models?: string[];
  onModelChange?: (model: string) => void;
  inputRef?: RefObject<HTMLTextAreaElement | null>;
  autoFocus?: boolean;
}

/**
 * Unified message & report composer shared by `/chat` and `/chat/[sessionId]`.
 *
 * Supports multi-PDF chips, compact mode & project popovers, and one-click
 * whole-paper report generation without requiring filler text when a PDF is
 * already attached.
 */
export default function Composer({
  value,
  onChange,
  mode,
  onModeChange,
  onSend,
  onQuickMode,
  onStop,
  streaming = false,
  sending = false,
  onAttach,
  papers = [],
  onPickPaper,
  onRemovePaper,
  projects,
  projectId = "",
  onProjectChange,
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
  const length = value.trim().length;
  const tooLong = length > MAX_MESSAGE_CHARS;

  const hasReadablePaper = papers.some((p) => Boolean(p.paper_id));
  const submittable = canSubmitTurn({
    draft: value,
    mode,
    hasReadablePaper,
    sending,
    streaming,
    attaching,
    tooLong,
  });

  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [value, textareaRef]);

  const attachFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0 || !onAttach) return;
      const list = Array.from(files);
      if (list.some((f) => !f.name.toLowerCase().endsWith(".pdf"))) {
        setAttachError(t("upload.drop_error"));
        if (fileRef.current) fileRef.current.value = "";
        return;
      }
      setAttaching(true);
      setAttachError("");
      try {
        for (const file of list) {
          await onAttach(file);
        }
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
  const selectedProject = projects?.find((p) => p.project_id === projectId) ?? null;
  const showQuickReports =
    hasReadablePaper && !value.trim() && mode === "auto" && !streaming && !sending && Boolean(onQuickMode);

  return (
    <div>
      <div className="rounded-2xl border border-border bg-card soft-shadow transition-[border-color,box-shadow] focus-within:border-primary/50 focus-within:shadow-[0_0_0_3px_color-mix(in_srgb,var(--primary)_12%,transparent)]">
        {shown.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 px-3 pt-2.5">
            {shown.map((paper) => (
              <span
                key={paper.literature_id || paper.paper_id}
                className="inline-flex max-w-[16rem] items-center gap-1.5 rounded-lg bg-muted px-2 py-1 text-xs text-foreground/85"
              >
                <button
                  type="button"
                  disabled={!paper.paper_id || !onPickPaper}
                  onClick={() => paper.paper_id && onPickPaper?.(paper.paper_id)}
                  title={`${paper.title || t("chat.papers.untitled")} · ${t(`chat.availability.${paper.availability}`)}`}
                  className="inline-flex min-w-0 items-center gap-1.5 enabled:hover:text-foreground disabled:cursor-default"
                >
                  <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${AVAILABILITY_DOT[paper.availability]}`} />
                  <span className="truncate">{paper.title || t("chat.papers.untitled")}</span>
                </button>
                {onRemovePaper && paper.paper_id && (
                  <button
                    type="button"
                    onClick={() => onRemovePaper(paper.paper_id)}
                    title={t("chat.composer.remove_paper")}
                    aria-label={t("chat.composer.remove_paper")}
                    className="ml-0.5 rounded p-0.5 text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
                  >
                    <IconX className="text-[11px]" />
                  </button>
                )}
              </span>
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
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              if (submittable) onSend();
            }
          }}
          rows={1}
          placeholder={mode === "auto" ? t("chat.placeholder") : t("chat.placeholder_mode")}
          className="block max-h-60 w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[0.925rem] leading-relaxed text-foreground placeholder:text-muted-foreground focus:outline-none focus-visible:outline-none"
        />

        <div className="flex items-center gap-1 px-2.5 pb-2.5 pt-1.5">
          {onAttach && (
            <>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".pdf,application/pdf"
                className="hidden"
                onChange={(e) => void attachFiles(e.target.files)}
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

          <ModePicker value={mode} onChange={onModeChange} disabled={streaming} />

          {projects && projects.length > 0 && onProjectChange && (
            <Menu
              label={t("chat.composer.project_pick")}
              align="start"
              width={220}
              disabled={streaming || sending}
              buttonClassName={`inline-flex h-8 max-w-[9.5rem] items-center gap-1.5 rounded-lg px-2.5 text-xs transition-colors ${
                selectedProject
                  ? "bg-muted font-medium text-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
              trigger={
                <>
                  <IconFolder className="shrink-0 text-[13px]" />
                  <span className="truncate">
                    {selectedProject
                      ? selectedProject.title || t("project.untitled")
                      : t("chat.composer.project_pick")}
                  </span>
                  <IconChevronDown className="shrink-0 text-[11px] opacity-70" />
                </>
              }
            >
              {(close) => (
                <div className="space-y-0.5 p-0.5">
                  <button
                    type="button"
                    onClick={() => {
                      onProjectChange("");
                      close();
                    }}
                    className="flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  >
                    <span>{t("chat.composer.project_none")}</span>
                    {!projectId && <IconCheck className="text-[13px] text-primary" />}
                  </button>
                  {projects.map((item) => {
                    const active = item.project_id === projectId;
                    return (
                      <button
                        key={item.project_id}
                        type="button"
                        onClick={() => {
                          onProjectChange(item.project_id);
                          close();
                        }}
                        className={`flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors ${
                          active ? "bg-accent/70 font-medium text-foreground" : "text-foreground hover:bg-muted"
                        }`}
                      >
                        <span className="truncate">{item.title || t("project.untitled")}</span>
                        {active && <IconCheck className="shrink-0 text-[13px] text-primary" />}
                      </button>
                    );
                  })}
                </div>
              )}
            </Menu>
          )}

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
              disabled={!submittable}
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

      {showQuickReports ? (
        <div className="mt-2.5 flex flex-wrap items-center justify-center gap-1.5 text-xs">
          <span className="text-muted-foreground">{t("chat.composer.quick_reports")}</span>
          {REPORT_MODES.map((m) => {
            const Icon = MODE_ICONS[m];
            return (
              <button
                key={m}
                type="button"
                onClick={() => onQuickMode?.(m)}
                title={t(`chat.mode.${m}.desc`)}
                className="inline-flex items-center gap-1 rounded-full border border-border bg-card px-2.5 py-1 text-xs text-foreground/85 transition-colors hover:border-primary/40 hover:bg-accent/60 hover:text-foreground"
              >
                <Icon className="text-[12px] text-primary" />
                <span>{t(`chat.mode.${m}.label`)}</span>
              </button>
            );
          })}
        </div>
      ) : (
        <p
          className={`mt-1.5 truncate text-center text-xs ${
            attachError || tooLong ? "text-destructive" : "text-muted-foreground"
          }`}
        >
          {attachError ||
            (tooLong
              ? t("chat.composer.too_long", { count: length, limit: MAX_MESSAGE_CHARS })
              : mode === "auto"
                ? t("chat.composer_hint")
                : hasReadablePaper && !value.trim()
                  ? t("chat.composer.mode_ready_hint", { mode: t(`chat.mode.${mode}.label`) })
                  : t(`chat.mode.${mode}.desc`))}
        </p>
      )}
    </div>
  );
}
