"use client";

import { Fragment, useState } from "react";
import { IconCards, IconChevronRight, IconTable } from "@/components/icons";
import { useTranslation } from "@/lib/i18n";
import type { SphereComparisonRow, SphereNodeData } from "@/lib/sphere";
import { NodeRankChips } from "@/components/sphere/PaperCard";

type Field = "problem" | "method" | "dataset" | "metric" | "strength" | "weakness";

const FIELDS: Field[] = ["problem", "method", "dataset", "metric", "strength", "weakness"];

const FIELD_TONE: Record<Field, string> = {
  problem: "text-muted-foreground",
  method: "text-muted-foreground",
  dataset: "text-muted-foreground",
  metric: "text-muted-foreground",
  strength: "text-success",
  weakness: "text-destructive",
};

interface Props {
  rows: SphereComparisonRow[];
  nodes: Record<string, SphereNodeData>;
  centerNodeId: string;
}

export default function ComparisonMatrix({ rows, nodes, centerNodeId }: Props) {
  const { t } = useTranslation();
  const [view, setView] = useState<"cards" | "table">("cards");
  const [expanded, setExpanded] = useState<Set<number>>(() => new Set([0]));

  const toggle = (i: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const label = (f: Field) => t(`sphere.cmp.${f}`);

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          {t("sphere.cmp.intro", { n: String(rows.length) })}
        </p>
        <div className="flex shrink-0 items-center rounded-lg border border-border p-0.5">
          {(
            [
              ["cards", IconCards, t("sphere.view.cards")],
              ["table", IconTable, t("sphere.view.table")],
            ] as const
          ).map(([key, Icon, title]) => (
            <button
              key={key}
              type="button"
              onClick={() => setView(key)}
              title={title}
              aria-pressed={view === key}
              className={`rounded-md px-2 py-1 text-sm transition-colors ${
                view === key
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Icon />
            </button>
          ))}
        </div>
      </div>

      {view === "cards" ? (
        <div className="space-y-2.5">
          {rows.map((row, i) => {
            const node = nodes[row.node_id];
            const isCenter = row.node_id === centerNodeId;
            const isOpen = expanded.has(i);
            const title = node?.title || row.title;
            return (
              <div
                key={`${row.node_id}-${i}`}
                className={`overflow-hidden rounded-xl border bg-card ${
                  isCenter ? "border-primary/40" : "border-border"
                }`}
              >
                <button
                  type="button"
                  onClick={() => toggle(i)}
                  aria-expanded={isOpen}
                  className="flex w-full items-start gap-2.5 px-3.5 py-3 text-left transition-colors hover:bg-muted/50"
                >
                  <IconChevronRight
                    className={`mt-0.5 shrink-0 text-sm text-muted-foreground transition-transform ${
                      isOpen ? "rotate-90" : ""
                    }`}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="text-[14px] font-medium leading-snug">{title}</span>
                      {isCenter && (
                        <span className="rounded-md bg-primary px-1.5 py-0.5 text-[11px] font-semibold text-primary-foreground">
                          {t("sphere.this_paper")}
                        </span>
                      )}
                      {node && (
                        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          {node.year > 0 && <span className="font-mono tabular-nums">{node.year}</span>}
                          <NodeRankChips node={node} />
                        </span>
                      )}
                    </span>
                    {!isOpen && row.problem && (
                      <span className="mt-1 line-clamp-2 block text-[13px] leading-relaxed text-muted-foreground">
                        {row.problem}
                      </span>
                    )}
                  </span>
                </button>

                {isOpen && (
                  <dl className="grid grid-cols-[3.5rem_1fr] gap-x-3 gap-y-2 border-t border-border bg-muted/25 px-3.5 py-3 text-[13px] leading-relaxed">
                    {FIELDS.map((f) =>
                      row[f] ? (
                        <Fragment key={f}>
                          <dt className={`pt-px text-xs font-semibold ${FIELD_TONE[f]}`}>
                            {label(f)}
                          </dt>
                          <dd className="min-w-0 text-foreground/85">{row[f]}</dd>
                        </Fragment>
                      ) : null,
                    )}
                  </dl>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border">
          <table className="w-full min-w-[64rem] border-collapse text-[13px]">
            <thead>
              <tr className="bg-muted/60 text-left">
                <th className="sticky left-0 z-10 w-56 min-w-56 border-b border-border bg-muted/95 px-3 py-2 font-semibold backdrop-blur">
                  {t("sphere.cmp.paper")}
                </th>
                {FIELDS.map((f) => (
                  <th
                    key={f}
                    className="min-w-52 border-b border-l border-border px-3 py-2 font-semibold"
                  >
                    {label(f)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => {
                const node = nodes[row.node_id];
                const isCenter = row.node_id === centerNodeId;
                return (
                  <tr
                    key={`${row.node_id}-${i}`}
                    className={`align-top transition-colors hover:bg-muted/40 ${
                      isCenter ? "bg-accent/40" : ""
                    }`}
                  >
                    <th
                      scope="row"
                      className={`sticky left-0 z-10 border-b border-border px-3 py-2.5 text-left font-medium ${
                        isCenter ? "bg-accent" : "bg-card"
                      }`}
                    >
                      {node?.title || row.title}
                      {node?.year ? (
                        <span className="ml-1.5 font-mono text-xs font-normal tabular-nums text-muted-foreground">
                          {node.year}
                        </span>
                      ) : null}
                    </th>
                    {FIELDS.map((f) => (
                      <td
                        key={f}
                        className="border-b border-l border-border px-3 py-2.5 text-foreground/85"
                      >
                        {row[f]}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
