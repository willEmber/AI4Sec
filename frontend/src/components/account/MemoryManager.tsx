"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createMemory,
  deleteMemory,
  describeApiError,
  listMemories,
  listProjects,
  updateMemory,
} from "@/lib/agent";
import type { AgentMemory, AgentProject, MemoryKind } from "@/lib/agent";
import { parseServerTime } from "@/lib/agentEvents";
import { useTranslation } from "@/lib/i18n";
import { IconPencil, IconPlus, IconTrash } from "@/components/icons";

const KINDS: MemoryKind[] = ["preference", "instruction", "fact", "project"];
const MAX_CHARS = 500;

const field =
  "rounded-lg border border-border bg-background px-2.5 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/40";
const smallButton =
  "rounded-md border border-border px-2.5 py-1 text-xs text-foreground transition-colors hover:bg-muted disabled:opacity-40";

interface Draft {
  content: string;
  kind: MemoryKind;
  project_id: string;
}

/**
 * Everything the agent remembers about the reader, in one place.
 *
 * The panel beside a conversation shows only what that conversation works
 * with. Here every scope is listed, and a memory can be reworded, re-typed or
 * moved between "everywhere" and one project — which there needs deleting and
 * telling the agent again.
 */
export default function MemoryManager() {
  const { t, locale } = useTranslation();
  const [memories, setMemories] = useState<AgentMemory[] | null>(null);
  const [limit, setLimit] = useState(0);
  const [projects, setProjects] = useState<AgentProject[]>([]);
  // "all", "" (global) or a project id.
  const [filter, setFilter] = useState("all");
  const [draft, setDraft] = useState<Draft>({ content: "", kind: "preference", project_id: "" });
  const [editing, setEditing] = useState<{ id: string; draft: Draft } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const reload = useCallback(async () => {
    try {
      const data = await listMemories();
      setMemories(data.memories);
      setLimit(data.limit);
    } catch (err) {
      setMemories([]);
      setError(describeApiError(err, t));
    }
  }, [t]);

  useEffect(() => {
    void reload();
    listProjects(true)
      .then((data) => setProjects(data.projects))
      .catch(() => setProjects([]));
  }, [reload]);

  const projectTitle = useCallback(
    (projectId: string) =>
      projects.find((p) => p.project_id === projectId)?.title || t("project.untitled"),
    [projects, t],
  );

  const run = useCallback(
    async (action: () => Promise<unknown>): Promise<boolean> => {
      setBusy(true);
      setError("");
      try {
        await action();
        await reload();
        return true;
      } catch (err) {
        setError(describeApiError(err, t));
        return false;
      } finally {
        setBusy(false);
      }
    },
    [reload, t],
  );

  const add = async () => {
    const content = draft.content.trim();
    if (!content) return;
    if (await run(() => createMemory({ ...draft, content }))) {
      setDraft((d) => ({ ...d, content: "" }));
    }
  };

  const saveEdit = async () => {
    if (!editing || !editing.draft.content.trim()) return;
    const { id, draft: d } = editing;
    if (await run(() => updateMemory(id, { ...d, content: d.content.trim() }))) setEditing(null);
  };

  const all = useMemo(() => memories ?? [], [memories]);
  const pending = all.filter((m) => m.pending);
  const kept = all.filter((m) => !m.pending && (filter === "all" || m.project_id === filter));
  // Scopes that hold something, so the filter never offers an empty one.
  const scopes = useMemo(() => {
    const ids = new Set(all.filter((m) => !m.pending).map((m) => m.project_id));
    return ["all", ...[...ids].sort((a, b) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)))];
  }, [all]);

  const scopeLabel = (scope: string) =>
    scope === "all"
      ? t("account.memory.filter_all")
      : scope === ""
        ? t("account.memory.scope_global")
        : projectTitle(scope);

  const kindSelect = (value: MemoryKind, onChange: (kind: MemoryKind) => void) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as MemoryKind)}
      className={field}
      aria-label={t("account.memory.kind")}
    >
      {KINDS.map((kind) => (
        <option key={kind} value={kind}>
          {t(`account.memory.kind.${kind}`)}
        </option>
      ))}
    </select>
  );

  const scopeSelect = (value: string, onChange: (projectId: string) => void) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`${field} max-w-[14rem]`}
      aria-label={t("account.memory.scope")}
    >
      <option value="">{t("account.memory.scope_global")}</option>
      {projects.map((p) => (
        <option key={p.project_id} value={p.project_id}>
          {p.title || t("project.untitled")}
        </option>
      ))}
    </select>
  );

  const when = (value: string) => {
    const ms = parseServerTime(value);
    return Number.isNaN(ms)
      ? ""
      : new Date(ms).toLocaleDateString(locale === "zh" ? "zh-CN" : "en-US", {
          year: "numeric",
          month: "short",
          day: "numeric",
        });
  };

  const source = (memory: AgentMemory) =>
    memory.source_session_id ? (
      <a
        href={`/chat/${memory.source_session_id}${
          memory.source_run_id ? `?run=${memory.source_run_id}` : ""
        }`}
        className="underline underline-offset-2 hover:text-foreground"
      >
        {t("account.memory.source")}
      </a>
    ) : (
      <span>{t("account.memory.source_manual")}</span>
    );

  return (
    <div>
      <p className="mb-5 text-sm leading-relaxed text-muted-foreground">
        {t("account.memory.subtitle")}
      </p>

      {pending.length > 0 && (
        <section className="mb-6 rounded-2xl border border-accent-foreground/30 bg-accent p-4">
          <h2 className="text-sm font-medium text-foreground">
            {t("chat.memory.pending_heading")} · {pending.length}
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            {t("chat.memory.pending_hint")}
          </p>
          <div className="mt-3 space-y-2">
            {pending.map((memory) => (
              <div
                key={memory.memory_id}
                className="flex flex-wrap items-start gap-2 rounded-xl border border-border bg-background px-3 py-2.5"
              >
                <div className="min-w-0 flex-1">
                  <p className="text-sm leading-snug text-foreground">{memory.content}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {scopeLabel(memory.project_id)} · {source(memory)}
                  </p>
                </div>
                <button
                  type="button"
                  disabled={busy}
                  className={smallButton}
                  onClick={() => void run(() => updateMemory(memory.memory_id, { confirm: true }))}
                >
                  {t("chat.memory.confirm")}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  className={smallButton}
                  onClick={() => void run(() => deleteMemory(memory.memory_id))}
                >
                  {t("chat.memory.discard")}
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="mb-6 rounded-2xl border border-border bg-card p-4">
        <textarea
          value={draft.content}
          onChange={(e) => setDraft((d) => ({ ...d, content: e.target.value }))}
          rows={2}
          maxLength={MAX_CHARS}
          placeholder={t("chat.memory.add_placeholder")}
          className={`${field} w-full resize-y`}
        />
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {kindSelect(draft.kind, (kind) => setDraft((d) => ({ ...d, kind })))}
          {scopeSelect(draft.project_id, (project_id) => setDraft((d) => ({ ...d, project_id })))}
          <div className="flex-1" />
          <button
            type="button"
            disabled={busy || !draft.content.trim()}
            onClick={() => void add()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3.5 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            <IconPlus className="text-[13px]" />
            {t("chat.memory.add")}
          </button>
        </div>
      </section>

      {error && <p className="mb-3 text-xs text-destructive">{error}</p>}

      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        {scopes.map((scope) => (
          <button
            key={scope || "global"}
            type="button"
            onClick={() => setFilter(scope)}
            className={`max-w-[14rem] truncate rounded-full border px-3 py-1 text-xs transition-colors ${
              filter === scope
                ? "border-foreground/30 bg-muted font-medium text-foreground"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            {scopeLabel(scope)}
          </button>
        ))}
        <div className="flex-1" />
        {limit > 0 && (
          <span className="text-xs tabular-nums text-muted-foreground">
            {t("account.memory.count", { count: all.length, limit })}
          </span>
        )}
      </div>

      {memories !== null && kept.length === 0 && (
        <div className="rounded-xl border border-dashed border-border px-5 py-6 text-center">
          <p className="text-sm text-muted-foreground">{t("chat.memory.empty")}</p>
        </div>
      )}

      <div className="space-y-2">
        {kept.map((memory) =>
          editing?.id === memory.memory_id ? (
            <div key={memory.memory_id} className="rounded-xl border border-border bg-card p-3">
              <textarea
                value={editing.draft.content}
                onChange={(e) =>
                  setEditing({ id: editing.id, draft: { ...editing.draft, content: e.target.value } })
                }
                rows={2}
                maxLength={MAX_CHARS}
                autoFocus
                className={`${field} w-full resize-y`}
              />
              <div className="mt-2 flex flex-wrap items-center gap-2">
                {kindSelect(editing.draft.kind, (kind) =>
                  setEditing({ id: editing.id, draft: { ...editing.draft, kind } }),
                )}
                {scopeSelect(editing.draft.project_id, (project_id) =>
                  setEditing({ id: editing.id, draft: { ...editing.draft, project_id } }),
                )}
                <div className="flex-1" />
                <button type="button" className={smallButton} onClick={() => setEditing(null)}>
                  {t("account.cancel")}
                </button>
                <button
                  type="button"
                  disabled={busy || !editing.draft.content.trim()}
                  className={smallButton}
                  onClick={() => void saveEdit()}
                >
                  {t("account.save")}
                </button>
              </div>
            </div>
          ) : (
            <div
              key={memory.memory_id}
              className="flex items-start gap-3 rounded-xl border border-border bg-card px-4 py-3"
            >
              <span className="mt-0.5 shrink-0 rounded bg-muted px-1.5 py-[1px] text-[0.65rem] text-muted-foreground">
                {t(`account.memory.kind.${memory.kind}`)}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm leading-snug text-foreground">{memory.content}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {scopeLabel(memory.project_id)} · {source(memory)} · {when(memory.updated_at)}
                </p>
              </div>
              <button
                type="button"
                disabled={busy}
                title={t("account.memory.edit")}
                aria-label={t("account.memory.edit")}
                onClick={() =>
                  setEditing({
                    id: memory.memory_id,
                    draft: {
                      content: memory.content,
                      kind: memory.kind,
                      project_id: memory.project_id,
                    },
                  })
                }
                className="shrink-0 rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40"
              >
                <IconPencil className="text-[14px]" />
              </button>
              <button
                type="button"
                disabled={busy}
                title={t("chat.memory.forget")}
                aria-label={t("chat.memory.forget")}
                onClick={() => void run(() => deleteMemory(memory.memory_id))}
                className="shrink-0 rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive disabled:opacity-40"
              >
                <IconTrash className="text-[14px]" />
              </button>
            </div>
          ),
        )}
      </div>
    </div>
  );
}
