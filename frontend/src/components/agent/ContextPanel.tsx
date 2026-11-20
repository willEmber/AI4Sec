"use client";

import type { ReactNode } from "react";
import { useTranslation } from "@/lib/i18n";
import { IconX } from "@/components/icons";

export type PanelTab = "pdf" | "report" | "sources";

interface Props {
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  onClose: () => void;
  /** Shown whenever the conversation holds at least one mode report. */
  hasReport: boolean;
  paperCount: number;
  /** Kept mounted while another tab shows, so the PDF document does not reload. */
  pdf: ReactNode;
  report: ReactNode;
  sources: ReactNode;
}

/**
 * Streamlined 3-tab reading & reference panel beside the conversation:
 *   1. Paper (`pdf`) — PDF reader with inline paper switcher & upload
 *   2. Report (`report`) — structured / Markdown mode reports when available
 *   3. Sources & Papers (`sources`) — cited passages and session/project papers
 *
 * Low-frequency long-term memory management lives in the conversation options
 * modal and account settings rather than competing with reading tabs.
 */
export default function ContextPanel({
  tab,
  onTab,
  onClose,
  hasReport,
  paperCount,
  pdf,
  report,
  sources,
}: Props) {
  const { t } = useTranslation();
  const tabs: [PanelTab, string][] = [["pdf", t("chat.panel.pdf")]];
  if (hasReport) tabs.push(["report", t("chat.panel.report")]);
  tabs.push([
    "sources",
    paperCount > 0 ? `${t("chat.panel.refs")} · ${paperCount}` : t("chat.panel.refs"),
  ]);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex h-12 shrink-0 items-center gap-1 border-b border-border pl-2.5 pr-1.5">
        <div role="tablist" className="no-scrollbar flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
          {tabs.map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              onClick={() => onTab(key)}
              className={`shrink-0 rounded-lg px-2.5 py-1.5 text-[0.8125rem] transition-colors ${
                tab === key
                  ? "bg-muted font-medium text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={onClose}
          title={t("chat.panel.close")}
          aria-label={t("chat.panel.close")}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <IconX className="text-[15px]" />
        </button>
      </div>

      <div className="relative min-h-0 flex-1">
        <div className={tab === "pdf" ? "h-full" : "hidden"}>{pdf}</div>
        {tab === "report" && report}
        {tab === "sources" && sources}
      </div>
    </div>
  );
}
