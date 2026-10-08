"use client";

import { useTranslation } from "@/lib/i18n";

interface Props {
  turns: { index: number; label: string }[];
  /** The turn currently at the top of the viewport. */
  active: number;
  onPick: (index: number) => void;
}

/**
 * The conversation's turns as a column of ticks along the edge; hovering it
 * lists the questions.
 *
 * Twenty turns in, the only way back to "what did it say about the ablation"
 * was to scroll and read. The questions are the reader's own words, so they
 * are the index that needs no explaining.
 */
export default function TurnRail({ turns, active, onPick }: Props) {
  const { t } = useTranslation();
  if (turns.length < 3) return null;

  return (
    <nav
      aria-label={t("chat.turns.label")}
      className="group/rail absolute right-3 top-1/2 z-10 hidden -translate-y-1/2 md:block"
    >
      <button
        type="button"
        aria-label={t("chat.turns.label")}
        className="flex max-h-[60vh] flex-col items-end gap-[6px] overflow-hidden rounded-md py-2 pl-4"
      >
        {turns.map((turn) => (
          <span
            key={turn.index}
            className={`h-[2px] rounded-full transition-all ${
              turn.index === active ? "w-4 bg-primary" : "w-2.5 bg-foreground/20"
            }`}
          />
        ))}
      </button>

      <ol className="invisible absolute right-0 top-1/2 max-h-[60vh] w-72 -translate-y-1/2 overflow-y-auto rounded-xl border border-border bg-card p-1.5 opacity-0 shadow-lg transition-opacity group-focus-within/rail:visible group-focus-within/rail:opacity-100 group-hover/rail:visible group-hover/rail:opacity-100">
        {turns.map((turn) => (
          <li key={turn.index}>
            <button
              type="button"
              onClick={() => onPick(turn.index)}
              className={`flex w-full items-baseline gap-2 rounded-lg px-2 py-1.5 text-left text-xs transition-colors hover:bg-muted ${
                turn.index === active ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              <span className="w-4 shrink-0 text-right tabular-nums opacity-60">{turn.index + 1}</span>
              <span className={`min-w-0 flex-1 truncate ${turn.index === active ? "font-medium" : ""}`}>
                {turn.label}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}
