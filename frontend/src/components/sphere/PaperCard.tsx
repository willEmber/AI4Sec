"use client";

import { BADGE_BASE, CCF_COLORS, SCI_COLORS } from "@/components/RankBadges";
import { IconExternal } from "@/components/icons";
import { useTranslation } from "@/lib/i18n";
import {
  DEFAULT_ACCENT,
  PARTITION_ACCENT,
  formatCitations,
  nodeAnchorId,
  nodeExternalUrl,
  type SphereNodeData,
} from "@/lib/sphere";

/** SCI / CCF chips for one node (shared styling with the header badges). */
export function NodeRankChips({ node }: { node: SphereNodeData }) {
  return (
    <>
      {node.sci_rank && SCI_COLORS[node.sci_rank] && (
        <span className={`${BADGE_BASE} ${SCI_COLORS[node.sci_rank]}`}>SCI {node.sci_rank}</span>
      )}
      {node.ccf_rank && CCF_COLORS[node.ccf_rank] && (
        <span className={`${BADGE_BASE} ${CCF_COLORS[node.ccf_rank]}`}>CCF {node.ccf_rank}</span>
      )}
    </>
  );
}

interface PaperCardProps {
  node: SphereNodeData;
  /** Partition key driving the left rail colour. */
  partitionKey?: string;
  /** Marks the paper being analysed (rendered with a primary ring). */
  isCenter?: boolean;
  /** Set when the card is the canonical anchor target for cross-links. */
  anchor?: boolean;
}

export default function PaperCard({
  node,
  partitionKey,
  isCenter = false,
  anchor = true,
}: PaperCardProps) {
  const { t } = useTranslation();
  const accent = (partitionKey && PARTITION_ACCENT[partitionKey]) || DEFAULT_ACCENT;

  return (
    <article
      id={anchor ? nodeAnchorId(node.node_id) : undefined}
      className={`group scroll-mt-24 rounded-xl border border-l-[3px] bg-card px-3.5 py-3 transition-colors ${
        isCenter ? "border-primary/40 border-l-primary bg-accent/40" : `border-border ${accent.rail}`
      } hover:border-foreground/20`}
    >
      <div className="flex items-start gap-2">
        <a
          href={nodeExternalUrl(node)}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[14px] font-medium leading-snug text-foreground decoration-primary/40 underline-offset-2 hover:text-primary hover:underline"
        >
          {node.title}
          <IconExternal className="ml-1 inline-block shrink-0 align-[-0.1em] text-[11px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-70" />
        </a>
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
        {isCenter && (
          <span className="rounded-md bg-primary px-1.5 py-0.5 text-[11px] font-semibold text-primary-foreground">
            {t("sphere.this_paper")}
          </span>
        )}
        {node.year > 0 && <span className="font-mono tabular-nums">{node.year}</span>}
        {node.venue && (
          <span className="min-w-0 max-w-[18rem] truncate" title={node.venue}>
            {node.venue}
          </span>
        )}
        <NodeRankChips node={node} />
        {node.cited_by_count > 0 && (
          <span className="tabular-nums" title={String(node.cited_by_count)}>
            {t("sphere.cited_n", { n: formatCitations(node.cited_by_count) })}
          </span>
        )}
        {node.influential && (
          <span className="rounded-md border border-primary/30 bg-accent px-1.5 py-0.5 text-[11px] font-medium text-accent-foreground">
            {t("sphere.influential")}
          </span>
        )}
      </div>

      {node.relation_reason && (
        <p className="mt-2 text-[13px] leading-relaxed text-foreground/75">{node.relation_reason}</p>
      )}
    </article>
  );
}

/** Inline reference to a node — used by evidence lists, themes and paths. */
export function PaperChip({
  node,
  index,
  onSelect,
}: {
  node: SphereNodeData;
  index?: number;
  onSelect?: (nodeId: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect?.(node.node_id)}
      className="inline-flex max-w-full items-center gap-1.5 rounded-lg border border-border bg-muted/50 px-2 py-1 text-left text-xs text-foreground/80 transition-colors hover:border-primary/40 hover:bg-accent hover:text-foreground sm:max-w-[26rem]"
      title={node.title}
    >
      {index !== undefined && (
        <span className="shrink-0 font-mono text-[11px] text-muted-foreground">{index}</span>
      )}
      <span className="truncate">{node.title}</span>
      {node.year > 0 && (
        <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
          {node.year}
        </span>
      )}
    </button>
  );
}
