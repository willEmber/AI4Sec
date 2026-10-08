"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { SessionArtifact } from "@/lib/agent";
import { getMarkdownExportUrl, getRunOutput, getZoteroBundleUrl } from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import { defaultLensView, parseLensData } from "@/lib/lens";
import { parseSnapData } from "@/lib/snap";
import { parseSphereData } from "@/lib/sphere";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import LensReport from "@/components/lens/LensReport";
import SnapReport from "@/components/snap/SnapReport";
import SphereReport from "@/components/sphere/SphereReport";
import {
  IconArrowRight,
  IconCards,
  IconDocument,
  IconDownload,
  IconExternal,
  IconLens,
  IconSnap,
  IconSphere,
} from "@/components/icons";

const MODE_ICON = { snap: IconSnap, lens: IconLens, sphere: IconSphere } as const;

/**
 * A mode report in the conversation, as a card that opens it.
 *
 * The report itself is rendered beside the conversation (`ArtifactView`), not
 * inside it: a report is pages long, and unfolded in the message list it was a
 * scrolling box inside a scrolling box. The card is the same while the turn
 * runs and after it is stored, so nothing moves when the turn ends.
 */
export default function ArtifactCard({
  artifact,
  onOpen,
  active = false,
}: {
  artifact: SessionArtifact;
  onOpen: (artifact: SessionArtifact) => void;
  /** This report is the one open in the side panel. */
  active?: boolean;
}) {
  const { t } = useTranslation();
  const Icon = MODE_ICON[artifact.mode as keyof typeof MODE_ICON] ?? IconCards;

  return (
    <div
      className={`flex items-center gap-3 rounded-xl border bg-card px-3 py-2.5 transition-colors ${
        active ? "border-primary/50" : "border-border"
      }`}
    >
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent text-primary">
        <Icon className="text-[16px]" />
      </span>
      <button type="button" onClick={() => onOpen(artifact)} className="min-w-0 flex-1 text-left">
        <span className="block truncate text-sm font-medium text-foreground">
          {t(`chat.mode.${artifact.mode}.label`)}
        </span>
        <span className="block truncate text-xs text-muted-foreground">
          {artifact.paper_title || artifact.paper_id.slice(0, 12)}
        </span>
      </button>
      <a
        href={getMarkdownExportUrl(artifact.run_id)}
        title={t("run.export_md")}
        aria-label={t("run.export_md")}
        className="hidden rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground sm:block"
      >
        <IconDownload className="text-[14px]" />
      </a>
      <button
        type="button"
        onClick={() => onOpen(artifact)}
        className="inline-flex shrink-0 items-center gap-1 rounded-lg border border-border px-2.5 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted"
      >
        {t("chat.artifact.view")}
        <IconArrowRight className="text-[12px]" />
      </button>
    </div>
  );
}

/**
 * The report itself — the same run the report page renders, same output, same
 * structured view. The full page, the compare matrix and the exports remain
 * one click away.
 */
export function ArtifactView({
  artifact,
  onJumpToPage,
}: {
  artifact: SessionArtifact;
  onJumpToPage?: (paperId: string, page: number) => void;
}) {
  const { t } = useTranslation();
  const [markdown, setMarkdown] = useState("");
  const [jsonData, setJsonData] = useState("");
  const [failed, setFailed] = useState(false);
  // Null until the reader picks a view, so Lens can open on its prose.
  const [viewChoice, setView] = useState<"structured" | "markdown" | null>(null);

  useEffect(() => {
    let cancelled = false;
    setMarkdown("");
    setJsonData("");
    setFailed(false);
    setView(null);
    getRunOutput(artifact.run_id)
      .then((o) => {
        if (cancelled) return;
        setMarkdown(o.markdown);
        setJsonData(o.json_data);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [artifact.run_id]);

  const sphereData = useMemo(() => parseSphereData(jsonData), [jsonData]);
  const snapData = useMemo(() => parseSnapData(jsonData), [jsonData]);
  const lensData = useMemo(() => parseLensData(jsonData), [jsonData]);
  const structured = sphereData ?? snapData ?? lensData;
  const view = viewChoice ?? defaultLensView(lensData);
  const showStructured = structured !== null && view === "structured";
  const jump = (page: number) => onJumpToPage?.(artifact.paper_id, page);
  const tool =
    "inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground";

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-border bg-card/60 px-3 py-2">
        <p className="min-w-0 flex-1 truncate text-sm text-foreground">
          <span className="font-medium">{t(`chat.mode.${artifact.mode}.label`)}</span>
          <span className="mx-1.5 opacity-40">·</span>
          <span className="text-muted-foreground">
            {artifact.paper_title || artifact.paper_id.slice(0, 12)}
          </span>
        </p>
        {structured && (
          <div className="flex shrink-0 items-center rounded-md border border-border p-0.5">
            {(
              [
                ["structured", IconCards, t("report.view.structured")],
                ["markdown", IconDocument, t("report.view.markdown")],
              ] as const
            ).map(([key, ViewIcon, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setView(key)}
                title={label}
                aria-pressed={view === key}
                className={`rounded px-1.5 py-1 text-xs transition-colors ${
                  view === key
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <ViewIcon className="text-[13px]" />
              </button>
            ))}
          </div>
        )}
        <a href={getMarkdownExportUrl(artifact.run_id)} title={t("run.export_md")} className={tool}>
          <IconDownload className="text-[13px]" />
        </a>
        <a
          href={getZoteroBundleUrl(artifact.run_id)}
          title={t("run.export_zotero_title")}
          className={`${tool} hidden sm:inline-flex`}
        >
          Zotero
        </a>
        <Link
          href={`/paper/${artifact.paper_id}/run/${artifact.run_id}`}
          title={t("chat.artifact.open")}
          className={tool}
        >
          <IconExternal className="text-[13px]" />
          <span className="hidden sm:inline">{t("chat.artifact.open")}</span>
        </Link>
      </div>

      <div className="relative min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {failed && <p className="text-xs text-destructive">{t("chat.artifact.failed")}</p>}
        {!failed && !markdown && (
          <p className="text-xs text-muted-foreground">{t("chat.artifact.loading")}</p>
        )}
        {showStructured && sphereData && <SphereReport data={sphereData} />}
        {showStructured && !sphereData && snapData && (
          <SnapReport data={snapData} onCitationClick={jump} />
        )}
        {showStructured && !sphereData && !snapData && lensData && (
          <LensReport data={lensData} markdown={markdown} onCitationClick={jump} />
        )}
        {!showStructured && markdown && <MarkdownRenderer content={markdown} onCitationClick={jump} />}
      </div>
    </div>
  );
}
