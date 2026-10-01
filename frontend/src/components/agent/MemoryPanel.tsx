"use client";

import { useCallback, useEffect, useState } from "react";
import { createMemory, deleteMemory, listMemories } from "@/lib/agent";
import type { AgentMemory } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import { IconPlus } from "@/components/icons";

interface Props {
  /** Bump to refetch — the agent can save a memory mid-turn. */
  refreshToken: number;
  /** The conversation's research project; "" outside one. */
  projectId?: string;
}

/**
 * What the agent remembers about this reader, across conversations.
 *
 * Shown rather than hidden: a memory changes how future answers are shaped,
 * so the reader must be able to see each one and remove it. Adding one by
 * hand is allowed too — telling the agent is the usual way, but a reader who
 * already knows what they want should not have to phrase it as a request.
 *
 * Inside a project the panel shows what the agent will actually see there:
 * global memories plus this project's, never another project's.
 */
export default function MemoryPanel({ refreshToken, projectId = "" }: Props) {
  const { t } = useTranslation();
  const [memories, setMemories] = useState<AgentMemory[] | null>(null);
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [projectOnly, setProjectOnly] = useState(false);

  const reload = useCallback(async () => {
    try {
      const data = await listMemories();
      setMemories(data.memories.filter((m) => !m.project_id || m.project_id === projectId));
    } catch {
      setMemories([]);
    }
  }, [projectId]);

  useEffect(() => {
    void reload();
  }, [reload, refreshToken]);

  const remove = useCallback(
    async (memoryId: string) => {
      setBusy(true);
      try {
        await deleteMemory(memoryId);
        setMemories((prev) => (prev ?? []).filter((m) => m.memory_id !== memoryId));
      } catch (err) {
        setError(String(err));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const add = useCallback(async () => {
    const content = draft.trim();
    if (!content) return;
    setBusy(true);
    setError("");
    try {
      await createMemory({ content, project_id: projectId && projectOnly ? projectId : "" });
      setDraft("");
      await reload();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }, [draft, reload, projectId, projectOnly]);

  const count = memories?.length ?? 0;

  return (
    <div className="border-t border-border">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <span className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {t("chat.memory.heading")}
          {count > 0 && <span className="ml-1.5 opacity-70">({count})</span>}
        </span>
        <span className="text-[0.7rem] text-muted-foreground">{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div className="max-h-56 overflow-y-auto px-2 pb-3">
          {memories !== null && memories.length === 0 && (
            <p className="px-2 pb-2 text-xs leading-relaxed text-muted-foreground">
              {t("chat.memory.empty")}
            </p>
          )}
          {(memories ?? []).map((memory) => (
            <div
              key={memory.memory_id}
              className="group mb-1 flex items-start gap-2 rounded-lg px-2 py-1.5 hover:bg-card/70"
            >
              <span className="mt-0.5 shrink-0 rounded bg-muted px-1 py-[1px] text-[0.6rem] uppercase text-muted-foreground">
                {t(`chat.memory.kind.${memory.kind}`)}
              </span>
              <span className="min-w-0 flex-1 text-xs leading-snug text-foreground">
                {memory.content}
                {memory.project_id && (
                  <span className="ml-1 text-[0.6rem] text-accent-foreground">
                    · {t("chat.memory.scope.project")}
                  </span>
                )}
              </span>
              <button
                type="button"
                disabled={busy}
                onClick={() => void remove(memory.memory_id)}
                title={t("chat.memory.forget")}
                className="shrink-0 text-[0.7rem] text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100 disabled:opacity-40"
              >
                ✕
              </button>
            </div>
          ))}

          <div className="mt-1 flex items-center gap-1 px-1">
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void add();
                }
              }}
              placeholder={t("chat.memory.add_placeholder")}
              maxLength={500}
              className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring/40"
            />
            <button
              type="button"
              disabled={busy || !draft.trim()}
              onClick={() => void add()}
              title={t("chat.memory.add")}
              className="rounded-md border border-border p-1 text-muted-foreground transition-colors hover:bg-card hover:text-foreground disabled:opacity-40"
            >
              <IconPlus className="text-[12px]" />
            </button>
          </div>
          {projectId && (
            <label className="mt-1 flex items-center gap-1.5 px-2 text-[0.7rem] text-muted-foreground">
              <input
                type="checkbox"
                checked={projectOnly}
                onChange={(e) => setProjectOnly(e.target.checked)}
              />
              {t("chat.memory.only_project")}
            </label>
          )}
          {error && <p className="mt-1 px-2 text-[0.7rem] text-destructive">{error}</p>}
          <p className="mt-2 px-2 text-[0.65rem] leading-relaxed text-muted-foreground">
            {t("chat.memory.hint")}
          </p>
        </div>
      )}
    </div>
  );
}
