"use client";

import { useState } from "react";
import { deleteMemory, describeApiError, updateMemory } from "@/lib/agent";
import type { SavedMemory } from "@/hooks/useAgentStream";
import { useTranslation } from "@/lib/i18n";
import { IconBookmark, IconX } from "@/components/icons";

interface Props {
  memory: SavedMemory;
  /** The memory was confirmed or removed here. */
  onChanged: () => void;
  onDismiss: () => void;
}

/**
 * Says, where the reader is looking, that the agent just kept something about
 * them — with the way back in the same line.
 *
 * A memory shapes every later conversation, so saving one must not be a row
 * in a folded process panel. One the reader never said is not in use yet and
 * asks to be confirmed instead of undone.
 */
export default function MemoryNotice({ memory, onChanged, onDismiss }: Props) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const act = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError("");
    try {
      await action();
      onChanged();
      onDismiss();
    } catch (err) {
      setError(describeApiError(err, t));
      setBusy(false);
    }
  };

  const button =
    "shrink-0 rounded-md border border-border px-2 py-0.5 text-xs text-foreground transition-colors hover:bg-card disabled:opacity-40";

  return (
    <div
      className={`flex items-start gap-2 rounded-xl border px-3 py-2 text-xs leading-relaxed ${
        memory.pending ? "border-accent-foreground/30 bg-accent" : "border-border bg-muted/60"
      }`}
    >
      <IconBookmark className="mt-[2px] shrink-0 text-[14px] text-muted-foreground" />
      <div className="min-w-0 flex-1 text-foreground">
        <span className="text-muted-foreground">
          {t(memory.pending ? "chat.memory.notice.pending" : "chat.memory.notice.saved")}
        </span>{" "}
        {memory.preview}
        {error && <span className="ml-2 text-destructive">{error}</span>}
      </div>
      {memory.pending ? (
        <>
          <button
            type="button"
            disabled={busy}
            className={button}
            onClick={() => void act(() => updateMemory(memory.memory_id, { confirm: true }))}
          >
            {t("chat.memory.confirm")}
          </button>
          <button
            type="button"
            disabled={busy}
            className={button}
            onClick={() => void act(() => deleteMemory(memory.memory_id))}
          >
            {t("chat.memory.discard")}
          </button>
        </>
      ) : (
        <button
          type="button"
          disabled={busy}
          className={button}
          onClick={() => void act(() => deleteMemory(memory.memory_id))}
        >
          {t("chat.memory.undo")}
        </button>
      )}
      <button
        type="button"
        onClick={onDismiss}
        title={t("chat.memory.dismiss")}
        aria-label={t("chat.memory.dismiss")}
        className="shrink-0 rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground"
      >
        <IconX className="text-[13px]" />
      </button>
    </div>
  );
}
