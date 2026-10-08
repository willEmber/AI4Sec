"use client";

import { useEffect, useMemo, useState } from "react";
import type { Evidence } from "@/lib/agent";
import { loadEvidenceMany } from "@/lib/evidenceCache";
import { useTranslation } from "@/lib/i18n";
import { citationNumbers } from "@/lib/turns";
import type { Turn } from "@/lib/turns";
import { evidenceTitle } from "@/components/agent/SourceStrip";
import { IconExternal } from "@/components/icons";

interface Props {
  turns: Turn[];
  onJumpToPage: (paperId: string, page: number) => void;
  onPickTurn: (index: number) => void;
}

interface SourceGroup {
  title: string;
  items: { evidence: Evidence; turns: number[] }[];
}

/**
 * Everything this conversation has cited, grouped by where it came from.
 *
 * Citation numbers restart with every answer, so across a long conversation
 * they do not say which passages keep being relied on. This does: each passage
 * once, with the turns that cited it.
 */
export default function SourcesTab({ turns, onJumpToPage, onPickTurn }: Props) {
  const { t } = useTranslation();

  // evidence id → the turns whose answers cite it, in first-cited order.
  const citedBy = useMemo(() => {
    const map = new Map<string, number[]>();
    for (const turn of turns) {
      for (const reply of turn.replies) {
        if (reply.role !== "assistant") continue;
        for (const id of citationNumbers(reply.content).keys()) {
          const list = map.get(id) ?? [];
          if (!list.includes(turn.index)) list.push(turn.index);
          map.set(id, list);
        }
      }
    }
    return map;
  }, [turns]);

  const idsKey = [...citedBy.keys()].join(",");
  const [resolved, setResolved] = useState<Map<string, Evidence | null> | null>(null);

  useEffect(() => {
    let cancelled = false;
    const ids = idsKey ? idsKey.split(",") : [];
    loadEvidenceMany(ids).then((map) => {
      if (!cancelled) setResolved(map);
    });
    return () => {
      cancelled = true;
    };
  }, [idsKey]);

  const groups = useMemo(() => {
    const byTitle = new Map<string, SourceGroup>();
    for (const [id, turnList] of citedBy) {
      const evidence = resolved?.get(id);
      if (!evidence) continue;
      const title = evidenceTitle(evidence);
      const group = byTitle.get(title) ?? { title, items: [] };
      group.items.push({ evidence, turns: turnList });
      byTitle.set(title, group);
    }
    return [...byTitle.values()];
  }, [citedBy, resolved]);

  if (citedBy.size === 0) {
    return <p className="p-4 text-xs leading-relaxed text-muted-foreground">{t("chat.sources.empty")}</p>;
  }
  if (!resolved) {
    return <p className="p-4 text-xs text-muted-foreground">{t("chat.cite.loading")}</p>;
  }

  return (
    <div className="h-full space-y-5 overflow-y-auto p-3">
      {groups.map((group) => (
        <section key={group.title}>
          <h3 className="mb-1.5 flex items-baseline gap-2 px-1">
            <span className="line-clamp-2 min-w-0 flex-1 text-[0.8125rem] font-medium leading-snug text-foreground">
              {group.title}
            </span>
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{group.items.length}</span>
          </h3>
          <ul className="space-y-1">
            {group.items.map(({ evidence, turns: citing }) => {
              const page = evidence.locator.page_label;
              const canJump = Boolean(page && evidence.paper_id);
              return (
                <li key={evidence.evidence_id} className="rounded-lg border border-border/70 bg-card px-2.5 py-2">
                  <p className="line-clamp-3 border-l-2 border-primary/40 pl-2 text-xs leading-relaxed text-foreground/80">
                    {evidence.quote}
                  </p>
                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[0.7rem] text-muted-foreground">
                    {canJump && (
                      <button
                        type="button"
                        onClick={() => onJumpToPage(evidence.paper_id, page as number)}
                        className="rounded bg-muted px-1.5 py-[1px] font-medium text-primary hover:underline"
                      >
                        {t("chat.cite.page", { page: page as number })}
                      </button>
                    )}
                    {evidence.source_level !== "fulltext" && (
                      <span className="rounded bg-muted px-1.5 py-[1px]">
                        {t(`chat.cite.level.${evidence.source_level}`)}
                      </span>
                    )}
                    {!canJump && evidence.source_url && (
                      <a
                        href={evidence.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-primary hover:underline"
                      >
                        <IconExternal className="text-[11px]" />
                        {t("chat.cite.open_source")}
                      </a>
                    )}
                    <span className="flex-1" />
                    {citing.map((index) => (
                      <button
                        key={index}
                        type="button"
                        onClick={() => onPickTurn(index)}
                        title={t("chat.sources.turn_hint", { n: index + 1 })}
                        className="rounded border border-border px-1.5 py-[1px] tabular-nums transition-colors hover:bg-muted hover:text-foreground"
                      >
                        {t("chat.sources.turn", { n: index + 1 })}
                      </button>
                    ))}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
      {resolved.size > 0 && groups.length === 0 && (
        <p className="px-1 text-xs text-muted-foreground">{t("chat.cite.failed")}</p>
      )}
    </div>
  );
}
