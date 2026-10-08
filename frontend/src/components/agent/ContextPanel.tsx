"use client";

import type { ReactNode } from "react";
import { useTranslation } from "@/lib/i18n";
import { IconX } from "@/components/icons";

export type PanelTab = "pdf" | "papers" | "sources" | "memory" | "report";

interface Props {
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  onClose: () => void;
  /** The report tab exists only while a report is open. */
  hasReport: boolean;
  paperCount: number;
  /** Kept mounted while another tab shows, so the document does not reload. */
  pdf: ReactNode;
  papers: ReactNode;
  sources: ReactNode;
  memory: ReactNode;
  report: ReactNode;
}

/**
 * What this conversation is working with: the open paper, the papers it can
 * read, the passages it has cited, what the agent remembers, and a report when
 * one is open.
 *
 * These used to be spread over a left sidebar and a PDF pane. They are one
 * thing — the conversation's context — and the conversation list is another.
 */
export default function ContextPanel({
  tab,
  onTab,
  onClose,
  hasReport,
  paperCount,
  pdf,
  papers,
  sources,
  memory,
  report,
}: Props) {
  const { t } = useTranslation();
  const tabs: [PanelTab, string][] = [
    ["pdf", t("chat.panel.pdf")],
    ["papers", paperCount > 0 ? `${t("chat.panel.papers")} ${paperCount}` : t("chat.panel.papers")],
    ["sources", t("chat.panel.sources")],
    ["memory", t("chat.memory.heading")],
  ];
  if (hasReport) tabs.push(["report", t("chat.panel.report")]);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex h-12 shrink-0 items-center gap-1 border-b border-border pl-2 pr-1.5">
        <div role="tablist" className="no-scrollbar flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto">
          {tabs.map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              onClick={() => onTab(key)}
              className={`shrink-0 rounded-md px-2.5 py-1.5 text-[0.8125rem] transition-colors ${
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
        {tab === "papers" && papers}
        {tab === "sources" && sources}
        {tab === "memory" && memory}
        {tab === "report" && report}
      </div>
    </div>
  );
}
