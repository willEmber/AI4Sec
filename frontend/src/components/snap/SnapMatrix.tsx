"use client";

import { useMemo, useState } from "react";
import { BADGE_BASE, CCF_COLORS, SCI_COLORS } from "@/components/RankBadges";
import { IconExternal } from "@/components/icons";
import { useTranslation } from "@/lib/i18n";
import {
  TIER_ACCENT,
  formatCount,
  isUnknown,
  paperExternalUrl,
  repoLabel,
  type SnapData,
  type VerdictTier,
} from "@/lib/snap";

/**
 * Paper × attribute matrix over several completed Snap runs.
 *
 * This is the payoff of making the report structured: once a finding is
 * {metric, dataset, value, delta} rather than a sentence, the same metric
 * measured by different papers lines up in a column and can be compared. Free
 * prose can never be stacked this way, which is why the old Snap could only ever
 * describe one paper at a time.
 */

export interface MatrixEntry {
  runId: string;
  paperId: string;
  data: SnapData;
}

const TIER_LABEL_KEY: Record<VerdictTier, string> = {
  must_read: "snap.tier.must_read",
  selective: "snap.tier.selective",
  skip: "snap.tier.skip",
};

const TIER_ORDER: Record<VerdictTier, number> = { must_read: 0, selective: 1, skip: 2 };

type SortKey = "tier" | "citations" | "year" | "title";

/** Metrics measured by more than one paper, most widely shared first. */
function sharedMetrics(entries: MatrixEntry[]): string[] {
  const counts = new Map<string, number>();
  for (const e of entries) {
    const seen = new Set(e.data.report.findings.map((f) => f.metric.trim().toLowerCase()));
    for (const m of seen) counts.set(m, (counts.get(m) || 0) + 1);
  }
  return [...counts.entries()]
    .filter(([, n]) => n > 1)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([m]) => m);
}

function TierChip({ tier }: { tier: VerdictTier }) {
  const { t } = useTranslation();
  const accent = TIER_ACCENT[tier] || TIER_ACCENT.selective;
  return (
    <span className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-semibold ${accent.chip}`}>
      {t(TIER_LABEL_KEY[tier])}
    </span>
  );
}

export default function SnapMatrix({ entries }: { entries: MatrixEntry[] }) {
  const { t } = useTranslation();
  const [sortKey, setSortKey] = useState<SortKey>("tier");

  const metrics = useMemo(() => sharedMetrics(entries), [entries]);

  const sorted = useMemo(() => {
    const rows = [...entries];
    rows.sort((a, b) => {
      switch (sortKey) {
        case "citations":
          return b.data.signals.cited_by_count - a.data.signals.cited_by_count;
        case "year":
          return (b.data.signals.year || 0) - (a.data.signals.year || 0);
        case "title":
          return a.data.title.localeCompare(b.data.title);
        default:
          return (
            TIER_ORDER[a.data.verdict.tier] - TIER_ORDER[b.data.verdict.tier] ||
            b.data.verdict.combined_score - a.data.verdict.combined_score
          );
      }
    });
    return rows;
  }, [entries, sortKey]);

  const sortOptions: { key: SortKey; label: string }[] = [
    { key: "tier", label: t("matrix.sort.tier") },
    { key: "citations", label: t("matrix.sort.citations") },
    { key: "year", label: t("matrix.sort.year") },
    { key: "title", label: t("matrix.sort.title") },
  ];

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">{t("matrix.sort_by")}</span>
        <div className="flex items-center rounded-lg border border-border p-0.5">
          {sortOptions.map((o) => (
            <button
              key={o.key}
              onClick={() => setSortKey(o.key)}
              aria-pressed={sortKey === o.key}
              className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                sortKey === o.key
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full min-w-[54rem] border-collapse text-[13px]">
          <thead>
            <tr className="bg-muted/60 text-left text-xs text-muted-foreground">
              <th className="sticky left-0 z-10 bg-muted/60 px-3 py-2 font-medium">{t("matrix.col.paper")}</th>
              <th className="px-3 py-2 font-medium">{t("matrix.col.verdict")}</th>
              <th className="px-3 py-2 font-medium">{t("snap.sig.venue")}</th>
              <th className="px-3 py-2 text-right font-medium">{t("snap.sig.citations")}</th>
              <th className="px-3 py-2 font-medium">{t("snap.sig.code")}</th>
              {metrics.map((m) => (
                <th key={m} className="px-3 py-2 text-right font-medium capitalize">
                  {m}
                </th>
              ))}
              <th className="px-3 py-2 font-medium">{t("matrix.col.limitation")}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(({ runId, paperId, data }) => {
              const s = data.signals;
              const official = s.repos.filter((r) => r.is_official);
              return (
                <tr key={runId} className="border-t border-border/70 align-top">
                  <td className="sticky left-0 z-10 max-w-[18rem] bg-card px-3 py-2.5">
                    <a
                      href={`/paper/${paperId}/run/${runId}`}
                      className="font-medium leading-snug decoration-primary/40 underline-offset-2 hover:text-primary hover:underline"
                      title={data.title}
                    >
                      {data.title || paperId}
                    </a>
                    {data.report.one_liner && (
                      <p className="mt-1 line-clamp-2 text-[12px] leading-snug text-muted-foreground">
                        {data.report.one_liner}
                      </p>
                    )}
                  </td>

                  <td className="whitespace-nowrap px-3 py-2.5">
                    <TierChip tier={data.verdict.tier} />
                    <p className="mt-1 font-mono text-[11px] tabular-nums text-muted-foreground">
                      {data.verdict.combined_score.toFixed(2)}
                    </p>
                    {s.is_retracted && (
                      <p className="mt-1 text-[11px] font-semibold text-destructive">
                        {t("matrix.retracted_short")}
                      </p>
                    )}
                  </td>

                  <td className="px-3 py-2.5">
                    <span className="flex flex-wrap items-center gap-1">
                      <span className="text-foreground/85" title={s.venue}>
                        {s.venue_normalized || s.venue || "—"}
                      </span>
                      {s.year > 0 && <span className="tabular-nums text-muted-foreground">{s.year}</span>}
                      {s.sci_rank && SCI_COLORS[s.sci_rank] && (
                        <span className={`${BADGE_BASE} ${SCI_COLORS[s.sci_rank]}`}>SCI {s.sci_rank}</span>
                      )}
                      {s.ccf_rank && CCF_COLORS[s.ccf_rank] && (
                        <span className={`${BADGE_BASE} ${CCF_COLORS[s.ccf_rank]}`}>CCF {s.ccf_rank}</span>
                      )}
                    </span>
                  </td>

                  <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono tabular-nums">
                    {isUnknown(s, "citations") ? (
                      <span className="text-muted-foreground">{t("snap.unknown")}</span>
                    ) : (
                      formatCount(s.cited_by_count)
                    )}
                  </td>

                  <td className="px-3 py-2.5">
                    {official.length > 0 ? (
                      <a
                        href={official[0].url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 break-all font-mono text-[11.5px] hover:text-primary hover:underline"
                      >
                        {repoLabel(official[0])}
                        {official[0].probe_ok && official[0].stars > 0 && (
                          <span className="shrink-0 text-muted-foreground">
                            {formatCount(official[0].stars)}★
                          </span>
                        )}
                        <IconExternal className="shrink-0 text-[10px] text-muted-foreground" />
                      </a>
                    ) : (
                      <span className="text-muted-foreground">{t("snap.none")}</span>
                    )}
                  </td>

                  {metrics.map((m) => {
                    const match = data.report.findings.find(
                      (f) => f.metric.trim().toLowerCase() === m,
                    );
                    return (
                      <td key={m} className="whitespace-nowrap px-3 py-2.5 text-right">
                        {match ? (
                          <>
                            <span className="font-mono tabular-nums">{match.value}</span>
                            {match.delta && (
                              <span
                                className={`ml-1.5 font-mono text-[11px] tabular-nums ${
                                  match.delta.startsWith("+")
                                    ? "text-success"
                                    : match.delta.startsWith("-")
                                      ? "text-destructive"
                                      : "text-muted-foreground"
                                }`}
                              >
                                {match.delta}
                              </span>
                            )}
                            {match.dataset && (
                              <p className="mt-0.5 text-[11px] text-muted-foreground">{match.dataset}</p>
                            )}
                          </>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                    );
                  })}

                  <td className="max-w-[16rem] px-3 py-2.5 text-[12.5px] leading-snug text-foreground/75">
                    {data.report.limitations[0]?.text || "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {metrics.length === 0 && (
        <p className="mt-2 text-[12px] italic text-muted-foreground">{t("matrix.no_shared_metrics")}</p>
      )}
    </div>
  );
}
