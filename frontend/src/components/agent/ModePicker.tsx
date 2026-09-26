"use client";

import type { ComponentType } from "react";
import { AGENT_MODES } from "@/lib/agent";
import type { AgentMode } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import { IconLens, IconSnap, IconSparkles, IconSphere } from "@/components/icons";

const ICONS: Record<AgentMode, ComponentType<{ className?: string }>> = {
  auto: IconSparkles,
  snap: IconSnap,
  lens: IconLens,
  sphere: IconSphere,
};

interface Props {
  value: AgentMode;
  onChange: (mode: AgentMode) => void;
  disabled?: boolean;
}

/**
 * The composer's mode switch.
 *
 * `auto` is the conversation as usual — the agent reads what the question
 * needs. The other three make the next turn produce a whole report through the
 * matching tool, and the switch resets to `auto` once that turn is sent, so a
 * follow-up question is not mistaken for a request to run the report again.
 */
export default function ModePicker({ value, onChange, disabled }: Props) {
  const { t } = useTranslation();
  return (
    <div
      role="radiogroup"
      aria-label={t("chat.mode.label")}
      className="inline-flex items-center gap-0.5 rounded-lg border border-border bg-background p-0.5"
    >
      {AGENT_MODES.map((mode) => {
        const Icon = ICONS[mode];
        const active = mode === value;
        return (
          <button
            key={mode}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled}
            title={t(`chat.mode.${mode}.desc`)}
            onClick={() => onChange(mode)}
            className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs transition-colors disabled:opacity-50 ${
              active
                ? "bg-accent font-medium text-accent-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Icon className="text-[13px]" />
            <span>{t(`chat.mode.${mode}.label`)}</span>
          </button>
        );
      })}
    </div>
  );
}
