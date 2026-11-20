"use client";

import type { ComponentType } from "react";
import { AGENT_MODES } from "@/lib/agent";
import type { AgentMode } from "@/lib/agent";
import { useTranslation } from "@/lib/i18n";
import Menu from "@/components/Menu";
import {
  IconCheck,
  IconChevronDown,
  IconLens,
  IconSnap,
  IconSparkles,
  IconSphere,
  IconX,
} from "@/components/icons";

export const MODE_ICONS: Record<AgentMode, ComponentType<{ className?: string }>> = {
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
 * Compact popover mode picker for the composer.
 *
 * Collapses the four reading modes into a single calm pill (`Auto` by default,
 * highlighted with a one-click `×` reset when a whole-paper report mode is
 * chosen) so the composer toolbar stays uncluttered on any screen width.
 */
export default function ModePicker({ value, onChange, disabled }: Props) {
  const { t } = useTranslation();
  const ActiveIcon = MODE_ICONS[value];
  const isReportMode = value !== "auto";

  return (
    <div className="inline-flex items-center">
      <Menu
        label={t("chat.mode.label")}
        align="start"
        width={284}
        disabled={disabled}
        buttonClassName={`inline-flex h-8 items-center gap-1.5 text-xs transition-colors disabled:opacity-50 ${
          isReportMode
            ? "rounded-l-lg bg-accent pl-2.5 pr-1.5 font-medium text-accent-foreground"
            : "rounded-lg px-2.5 text-muted-foreground hover:bg-muted hover:text-foreground"
        }`}
        trigger={
          <>
            <ActiveIcon className="shrink-0 text-[13px]" />
            <span className="truncate">{t(`chat.mode.${value}.label`)}</span>
            <IconChevronDown className="shrink-0 text-[11px] opacity-70" />
          </>
        }
      >
        {(close) => (
          <div className="space-y-0.5 p-0.5">
            {AGENT_MODES.map((mode) => {
              const Icon = MODE_ICONS[mode];
              const active = mode === value;
              return (
                <button
                  key={mode}
                  type="button"
                  role="menuitemradio"
                  aria-checked={active}
                  onClick={() => {
                    onChange(mode);
                    close();
                  }}
                  className={`flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors ${
                    active ? "bg-accent/70 text-foreground" : "hover:bg-muted"
                  }`}
                >
                  <Icon
                    className={`mt-0.5 shrink-0 text-[15px] ${
                      active ? "text-primary" : "text-muted-foreground"
                    }`}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center justify-between gap-2 text-xs font-medium text-foreground">
                      <span>{t(`chat.mode.${mode}.label`)}</span>
                      {active && <IconCheck className="text-[13px] text-primary" />}
                    </span>
                    <span className="mt-0.5 block text-[11.5px] leading-snug text-muted-foreground">
                      {t(`chat.mode.${mode}.desc`)}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </Menu>

      {isReportMode && (
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange("auto")}
          title={t("chat.mode.reset")}
          aria-label={t("chat.mode.reset")}
          className="inline-flex h-8 items-center justify-center rounded-r-lg border-l border-border/60 bg-accent px-1.5 text-accent-foreground transition-colors hover:bg-accent/80 disabled:opacity-50"
        >
          <IconX className="text-[12px]" />
        </button>
      )}
    </div>
  );
}
