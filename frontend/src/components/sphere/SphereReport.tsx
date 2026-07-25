"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import ComparisonMatrix from "@/components/sphere/ComparisonMatrix";
import PaperCard, { PaperChip } from "@/components/sphere/PaperCard";
import { useTranslation } from "@/lib/i18n";
import {
  CLUSTER_ACCENT,
  DEFAULT_ACCENT,
  PARTITION_ACCENT,
  PARTITION_ORDER,
  focusNode,
  formatCitations,
  nodeExternalUrl,
  type SphereData,
  type SphereNodeData,
} from "@/lib/sphere";

/** Section ids in render order; the nav is built from the ones with content. */
type SectionId =
  | "overview"
  | "core"
  | "themes"
  | "timeline"
  | "comparison"
  | "hubs"
  | "gaps"
  | "paths"
  | "library";

function Section({
  id,
  title,
  subtitle,
  children,
}: {
  id: SectionId;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={`sphere-sec-${id}`} className="scroll-mt-16 pt-8 first:pt-2">
      <h2 className="font-display text-[19px] font-semibold tracking-tight">{title}</h2>
      {subtitle && <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{subtitle}</p>}
      <div className="mt-3.5">{children}</div>
    </section>
  );
}

export default function SphereReport({ data }: { data: SphereData }) {
  const { t } = useTranslation();
  const out = data.sphere_output;
  const nodes = data.nodes;
  const [activePartition, setActivePartition] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<SectionId | null>(null);

  const partitions = useMemo(() => {
    const present = data.partitions.filter((p) => p.node_ids.length > 0);
    return [...present].sort(
      (a, b) =>
        PARTITION_ORDER.indexOf(a.key as (typeof PARTITION_ORDER)[number]) -
        PARTITION_ORDER.indexOf(b.key as (typeof PARTITION_ORDER)[number]),
    );
  }, [data.partitions]);

  const coreCount = useMemo(
    () => partitions.reduce((sum, p) => sum + p.node_ids.length, 0),
    [partitions],
  );

  const yearSpan = useMemo(() => {
    const years = out.timeline.map((e) => e.year).filter((y) => y > 0);
    if (years.length === 0) return "";
    return `${Math.min(...years)}–${Math.max(...years)}`;
  }, [out.timeline]);

  const sections = useMemo(() => {
    const list: { id: SectionId; label: string }[] = [];
    if (out.sphere_overview) list.push({ id: "overview", label: t("sphere.sec.overview") });
    if (partitions.length) list.push({ id: "core", label: t("sphere.sec.core") });
    if (out.themes.length) list.push({ id: "themes", label: t("sphere.sec.themes") });
    if (out.timeline.length) list.push({ id: "timeline", label: t("sphere.sec.timeline") });
    if (out.comparison_table.length) list.push({ id: "comparison", label: t("sphere.sec.comparison") });
    if (out.key_hubs.length) list.push({ id: "hubs", label: t("sphere.sec.hubs") });
    if (out.gaps_and_ideas.length) list.push({ id: "gaps", label: t("sphere.sec.gaps") });
    if (out.reading_paths.length) list.push({ id: "paths", label: t("sphere.sec.paths") });
    if (data.library_matches.length) list.push({ id: "library", label: t("sphere.sec.library") });
    return list;
  }, [out, partitions.length, data.library_matches.length, t]);

  // Highlight the section currently occupying the top of the reading pane.
  useEffect(() => {
    const observed = sections
      .map((s) => document.getElementById(`sphere-sec-${s.id}`))
      .filter((el): el is HTMLElement => el !== null);
    if (observed.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) {
          setActiveSection(visible[0].target.id.replace("sphere-sec-", "") as SectionId);
        }
      },
      { rootMargin: "-64px 0px -65% 0px", threshold: 0 },
    );
    observed.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [sections]);

  /** Cross-link: scroll to the paper's card, or open it externally if unlisted. */
  const selectNode = (nodeId: string) => {
    setActivePartition(null);
    // Let the filter reset paint before measuring the anchor position.
    window.requestAnimationFrame(() => {
      if (!focusNode(nodeId)) {
        const node = nodes[nodeId];
        if (node) window.open(nodeExternalUrl(node), "_blank", "noopener,noreferrer");
      }
    });
  };

  const nodeOf = (id: string): SphereNodeData | undefined => nodes[id];

  // `t` echoes unknown keys, so keys derived from backend values (partition
  // keys, reading-path names) need an explicit fallback to the raw value.
  const tr = (key: string, fallback: string) => {
    const text = t(key);
    return text === key ? fallback : text;
  };

  const visiblePartitions = activePartition
    ? partitions.filter((p) => p.key === activePartition)
    : partitions;

  const maxPagerank = Math.max(...out.key_hubs.map((h) => h.pagerank), 0.0001);

  return (
    <div className="pb-16">
      {/* Section navigation — sticks to the top of the reading pane */}
      {sections.length > 1 && (
        <nav className="no-scrollbar sticky top-0 z-20 -mx-1 mb-1 flex gap-1 overflow-x-auto border-b border-border bg-background/85 px-1 py-2 backdrop-blur">
          {sections.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() =>
                document
                  .getElementById(`sphere-sec-${s.id}`)
                  ?.scrollIntoView({ behavior: "smooth", block: "start" })
              }
              className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                activeSection === s.id
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              {s.label}
            </button>
          ))}
        </nav>
      )}

      {/* ── Stats ── */}
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {[
          { label: t("sphere.stat.core"), value: String(coreCount) },
          { label: t("sphere.stat.candidates"), value: String(data.num_nodes) },
          { label: t("sphere.stat.themes"), value: String(out.themes.length) },
          { label: t("sphere.stat.span"), value: yearSpan || "—" },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-border bg-card px-3 py-2.5">
            <p className="font-display text-xl font-semibold tabular-nums leading-none">{s.value}</p>
            <p className="mt-1.5 text-xs text-muted-foreground">{s.label}</p>
          </div>
        ))}
      </div>

      {/* ── Overview ── */}
      {out.sphere_overview && (
        <Section id="overview" title={t("sphere.sec.overview")}>
          <div className="space-y-3 rounded-xl border border-border bg-card px-4 py-3.5 text-[14px] leading-relaxed text-foreground/85">
            {out.sphere_overview
              .split(/\n{2,}/)
              .map((p) => p.trim())
              .filter(Boolean)
              .map((p, i) => (
                <p key={i}>{p}</p>
              ))}
          </div>
        </Section>
      )}

      {/* ── Core literature partitions ── */}
      {partitions.length > 0 && (
        <Section
          id="core"
          title={t("sphere.sec.core")}
          subtitle={t("sphere.core.intro", { n: String(coreCount) })}
        >
          <div className="mb-3.5 flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => setActivePartition(null)}
              className={`rounded-full border px-2.5 py-1 text-xs font-medium transition-colors ${
                activePartition === null
                  ? "border-primary/40 bg-accent text-accent-foreground"
                  : "border-border text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              {t("sphere.core.all")} {coreCount}
            </button>
            {partitions.map((p) => {
              const accent = PARTITION_ACCENT[p.key] || DEFAULT_ACCENT;
              const active = activePartition === p.key;
              return (
                <button
                  key={p.key}
                  type="button"
                  onClick={() => setActivePartition(active ? null : p.key)}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors ${
                    active
                      ? accent.chip
                      : "border-border text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${accent.dot}`} />
                  {tr(`sphere.partition.${p.key}`, p.key)}
                  <span className="tabular-nums opacity-70">{p.node_ids.length}</span>
                </button>
              );
            })}
          </div>

          <div className="space-y-6">
            {visiblePartitions.map((p) => {
              const accent = PARTITION_ACCENT[p.key] || DEFAULT_ACCENT;
              return (
                <div key={p.key}>
                  <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
                    <span className={`h-2 w-2 rounded-full ${accent.dot}`} />
                    {tr(`sphere.partition.${p.key}`, p.key)}
                    <span className="font-normal tabular-nums text-muted-foreground">
                      {p.node_ids.length}
                    </span>
                  </h3>
                  <p className="mb-2.5 text-xs leading-relaxed text-muted-foreground">
                    {tr(`sphere.partition.${p.key}.desc`, "")}
                  </p>
                  <div className="space-y-2">
                    {p.node_ids.map((id) => {
                      const node = nodeOf(id);
                      return node ? (
                        <PaperCard key={id} node={node} partitionKey={p.key} />
                      ) : null;
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </Section>
      )}

      {/* ── Thematic clusters ── */}
      {out.themes.length > 0 && (
        <Section id="themes" title={t("sphere.sec.themes")} subtitle={t("sphere.themes.intro")}>
          <div className="space-y-3">
            {out.themes.map((theme, i) => (
              <div key={i} className="rounded-xl border border-border bg-card px-4 py-3.5">
                <h3 className="flex items-center gap-2 text-[14px] font-semibold">
                  <span
                    className={`h-2.5 w-2.5 shrink-0 rounded-full ${
                      CLUSTER_ACCENT[i % CLUSTER_ACCENT.length]
                    }`}
                  />
                  {theme.name}
                  <span className="ml-auto shrink-0 rounded-full bg-muted px-2 py-0.5 text-[11px] font-normal tabular-nums text-muted-foreground">
                    {theme.representative_node_ids.length}
                  </span>
                </h3>
                {theme.definition && (
                  <p className="mt-1.5 text-[13px] leading-relaxed text-foreground/75">
                    {theme.definition}
                  </p>
                )}
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {theme.representative_node_ids.map((id) => {
                    const node = nodeOf(id);
                    return node ? (
                      <PaperChip key={id} node={node} onSelect={selectNode} />
                    ) : null;
                  })}
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* ── Timeline ── */}
      {out.timeline.length > 0 && (
        <Section id="timeline" title={t("sphere.sec.timeline")} subtitle={t("sphere.timeline.intro")}>
          <div className="grid grid-cols-[2.75rem_1fr] gap-x-3">
            {out.timeline.map((entry) => {
              const entryNodes = entry.node_ids
                .map(nodeOf)
                .filter((n): n is SphereNodeData => n !== undefined);
              const hasCenter = entry.node_ids.includes(data.center_node_id);
              return (
                <Fragment key={entry.year}>
                  <div className="pt-0.5 text-right font-mono text-xs tabular-nums text-muted-foreground">
                    {entry.year}
                  </div>
                  <div className="relative border-l border-border pb-4 pl-4">
                    <span
                      className={`absolute -left-[4.5px] top-[7px] h-2 w-2 rounded-full ring-2 ring-background ${
                        hasCenter ? "bg-primary" : "bg-border"
                      }`}
                    />
                    <div className="flex flex-wrap gap-1.5">
                      {entryNodes.map((node) =>
                        node.node_id === data.center_node_id ? (
                          <span
                            key={node.node_id}
                            className="inline-flex max-w-full items-center gap-1.5 rounded-lg border border-primary/40 bg-accent px-2 py-1 text-xs font-medium text-accent-foreground"
                          >
                            <span className="truncate">{node.title}</span>
                            <span className="shrink-0 opacity-70">{t("sphere.this_paper")}</span>
                          </span>
                        ) : (
                          <PaperChip key={node.node_id} node={node} onSelect={selectNode} />
                        ),
                      )}
                    </div>
                  </div>
                </Fragment>
              );
            })}
          </div>
        </Section>
      )}

      {/* ── Comparison matrix ── */}
      {out.comparison_table.length > 0 && (
        <Section id="comparison" title={t("sphere.sec.comparison")}>
          <ComparisonMatrix
            rows={out.comparison_table}
            nodes={nodes}
            centerNodeId={data.center_node_id}
          />
        </Section>
      )}

      {/* ── Key hubs ── */}
      {out.key_hubs.length > 0 && (
        <Section id="hubs" title={t("sphere.sec.hubs")} subtitle={t("sphere.hubs.intro")}>
          <ol className="space-y-1.5">
            {out.key_hubs.map((hub, i) => {
              const node = nodeOf(hub.node_id);
              return (
                <li
                  key={`${hub.node_id}-${i}`}
                  className="group rounded-xl border border-border bg-card px-3.5 py-2.5"
                >
                  <div className="flex items-baseline gap-2.5">
                    <span className="w-4 shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                      {i + 1}
                    </span>
                    <button
                      type="button"
                      onClick={() => selectNode(hub.node_id)}
                      className="min-w-0 flex-1 text-left text-[14px] font-medium leading-snug hover:text-primary"
                    >
                      {hub.title}
                    </button>
                    {hub.cited_by_count > 0 && (
                      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                        {t("sphere.cited_n", { n: formatCitations(hub.cited_by_count) })}
                      </span>
                    )}
                  </div>
                  <div className="mt-2 flex items-center gap-2.5 pl-[26px]">
                    <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-muted">
                      <span
                        className="block h-full rounded-full bg-primary/70"
                        style={{ width: `${Math.max(4, (hub.pagerank / maxPagerank) * 100)}%` }}
                      />
                    </span>
                    <span
                      className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground"
                      title={t("sphere.hubs.pagerank")}
                    >
                      PR {hub.pagerank.toFixed(3)}
                    </span>
                  </div>
                  {node?.venue && (
                    <p className="mt-1.5 truncate pl-[26px] text-xs text-muted-foreground">
                      {node.venue}
                    </p>
                  )}
                </li>
              );
            })}
          </ol>
        </Section>
      )}

      {/* ── Research gaps ── */}
      {out.gaps_and_ideas.length > 0 && (
        <Section id="gaps" title={t("sphere.sec.gaps")}>
          <div className="space-y-3">
            {out.gaps_and_ideas.map((gap, i) => (
              <div
                key={i}
                className="rounded-xl border border-border border-l-[3px] border-l-primary/60 bg-card px-4 py-3.5"
              >
                <h3 className="flex gap-2.5 text-[14px] font-semibold leading-snug">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-[11px] text-primary">
                    {i + 1}
                  </span>
                  {gap.title}
                </h3>
                {gap.description && (
                  <p className="mt-2 text-[13px] leading-relaxed text-foreground/80">
                    {gap.description}
                  </p>
                )}
                {gap.evidence_node_ids.length > 0 && (
                  <div className="mt-3">
                    <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                      {t("sphere.gaps.evidence")}
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {gap.evidence_node_ids.map((id) => {
                        const node = nodeOf(id);
                        return node ? (
                          <PaperChip key={id} node={node} onSelect={selectNode} />
                        ) : null;
                      })}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* ── Reading paths ── */}
      {out.reading_paths.length > 0 && (
        <Section id="paths" title={t("sphere.sec.paths")}>
          <div className="grid gap-3 lg:grid-cols-2">
            {out.reading_paths.map((path) => (
              <div key={path.name} className="rounded-xl border border-border bg-card px-4 py-3.5">
                <h3 className="flex items-center gap-2 text-[14px] font-semibold">
                  <span className="rounded-md bg-accent px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-accent-foreground">
                    {tr(`sphere.path.${path.name}`, path.name)}
                  </span>
                  <span className="text-xs font-normal tabular-nums text-muted-foreground">
                    {t("sphere.path.steps", { n: String(path.node_ids.length) })}
                  </span>
                </h3>
                {path.description && (
                  <p className="mt-2 text-[13px] leading-relaxed text-foreground/75">
                    {path.description}
                  </p>
                )}
                <ol className="mt-3 space-y-1.5">
                  {path.node_ids.map((id, i) => {
                    const node = nodeOf(id);
                    if (!node) return null;
                    return (
                      <li key={id} className="relative flex gap-2.5 pb-1.5 last:pb-0">
                        <span className="relative flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-border bg-muted font-mono text-[11px] text-muted-foreground">
                          {i + 1}
                          {i < path.node_ids.length - 1 && (
                            <span className="absolute top-full left-1/2 h-[calc(100%-0.25rem)] w-px -translate-x-1/2 bg-border" />
                          )}
                        </span>
                        <button
                          type="button"
                          onClick={() => selectNode(id)}
                          className="min-w-0 flex-1 text-left text-[13px] leading-snug text-foreground/85 hover:text-primary"
                        >
                          {node.title}
                          {node.year > 0 && (
                            <span className="ml-1.5 font-mono text-[11px] tabular-nums text-muted-foreground">
                              {node.year}
                            </span>
                          )}
                        </button>
                      </li>
                    );
                  })}
                </ol>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* ── Related work in the user's library ── */}
      {data.library_matches.length > 0 && (
        <Section id="library" title={t("sphere.sec.library")} subtitle={t("sphere.library.intro")}>
          <div className="flex flex-wrap gap-1.5">
            {data.library_matches.map((m) => (
              <a
                key={m.document_id || m.title}
                href={m.document_id ? `/library?doc=${m.document_id}` : undefined}
                className="inline-flex max-w-full items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs text-foreground/80 transition-colors hover:border-primary/40 hover:bg-accent hover:text-foreground"
                title={m.title}
              >
                <span className="truncate">{m.title}</span>
                {m.score > 0 && (
                  <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
                    {m.score.toFixed(2)}
                  </span>
                )}
              </a>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}
