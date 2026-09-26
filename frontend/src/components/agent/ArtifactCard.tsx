"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { SessionArtifact } from "@/lib/agent";
import { getMarkdownExportUrl, getRunOutput, getZoteroBundleUrl } from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import { parseLensData } from "@/lib/lens";
import { parseSnapData } from "@/lib/snap";
import { parseSphereData } from "@/lib/sphere";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import LensReport from "@/components/lens/LensReport";
import SnapReport from "@/components/snap/SnapReport";
import SphereReport from "@/components/sphere/SphereReport";
import {
  IconCards,
  IconDocument,
  IconDownload,
  IconExternal,
  IconLens,
  IconSnap,
  IconSphere,
} from "@/components/icons";

interface Props {
  artifact: SessionArtifact;
  onJumpToPage?: (paperId: string, page: number) => void;
  /** Sphere reports are long; they start folded, the other two start open. */
  defaultOpen?: boolean;
}

const MODE_ICON = { snap: IconSnap, lens: IconLens, sphere: IconSphere } as const;

/**
 * A mode report inside the conversation.
 *
 * This is the same run the report page renders — same output, same structured
 * view — shown in place so the reader does not have to leave the conversation
 * to see what the agent just produced. The full page, the compare matrix and
 * the exports remain one click away.
 */
export default function ArtifactCard({ artifact, onJumpToPage, defaultOpen }: Props) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(defaultOpen ?? artifact.mode !== "sphere");
  const [markdown, setMarkdown] = useState("");
  const [jsonData, setJsonData] = useState("");
  const [failed, setFailed] = useState(false);
  const [view, setView] = useState<"structured" | "markdown">("structured");

  useEffect(() => {
    if (!open || markdown || failed) return;
    let cancelled = false;
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
  }, [open, markdown, failed, artifact.run_id]);

  const sphereData = useMemo(() => parseSphereData(jsonData), [jsonData]);
  const snapData = useMemo(() => parseSnapData(jsonData), [jsonData]);
  const lensData = useMemo(() => parseLensData(jsonData), [jsonData]);
  const structured = sphereData ?? snapData ?? lensData;
  const showStructured = structured !== null && view === "structured";

  const Icon = MODE_ICON[artifact.mode as keyof typeof MODE_ICON] ?? IconCards;
  const jump = (page: number) => onJumpToPage?.(artifact.paper_id, page);

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card soft-shadow">
      <div className="flex items-center gap-3 border-b border-border bg-muted/40 px-4 py-2.5">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent text-primary">
          <Icon className="text-[15px]" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-foreground">
            {t(`chat.mode.${artifact.mode}.label`)}
            <span className="mx-1.5 opacity-40">·</span>
            <span className="text-muted-foreground">
              {artifact.paper_title || artifact.paper_id.slice(0, 12)}
            </span>
          </p>
          <p className="text-[0.7rem] text-muted-foreground">{t("chat.artifact.hint")}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {structured && open && (
            <div className="flex items-center rounded-md border border-border p-0.5">
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
          <a
            href={getMarkdownExportUrl(artifact.run_id)}
            title={t("run.export_md")}
            className="rounded-md border border-border p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <IconDownload className="text-[13px]" />
          </a>
          <a
            href={getZoteroBundleUrl(artifact.run_id)}
            title={t("run.export_zotero_title")}
            className="hidden rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground sm:inline-flex"
          >
            Zotero
          </a>
          <Link
            href={`/paper/${artifact.paper_id}/run/${artifact.run_id}`}
            title={t("chat.artifact.open")}
            className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <IconExternal className="text-[13px]" />
            <span className="hidden sm:inline">{t("chat.artifact.open")}</span>
          </Link>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            {open ? t("chat.artifact.collapse") : t("chat.artifact.expand")}
          </button>
        </div>
      </div>

      {open && (
        <div className="max-h-[70vh] overflow-y-auto px-5 py-4">
          {failed && (
            <p className="text-xs text-destructive">{t("chat.artifact.failed")}</p>
          )}
          {!failed && !markdown && (
            <p className="text-xs text-muted-foreground">{t("chat.artifact.loading")}</p>
          )}
          {showStructured && sphereData && <SphereReport data={sphereData} />}
          {showStructured && !sphereData && snapData && (
            <SnapReport data={snapData} onCitationClick={jump} />
          )}
          {showStructured && !sphereData && !snapData && lensData && (
            <LensReport data={lensData} onCitationClick={jump} />
          )}
          {!showStructured && markdown && (
            <MarkdownRenderer content={markdown} onCitationClick={jump} />
          )}
        </div>
      )}
    </div>
  );
}
