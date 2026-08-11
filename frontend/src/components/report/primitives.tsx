"use client";

/**
 * Shared building blocks for the structured report views.
 *
 * Insight Snap, Logic Lens and Research Sphere each render a different payload,
 * but a page citation, a claim, a quantified result and a stat tile should look
 * and behave identically in all three — the toggle in the run header is the same
 * control, so what it reveals has to read as one product. Keeping these here
 * means a styling change lands in every mode at once, instead of drifting into
 * three near-copies.
 */

import { useEffect, useMemo, useState } from "react";
import katex from "katex";
import { useTranslation } from "@/lib/i18n";

const INLINE_MATH_RE = /\$([^$\n]{1,160})\$/g;
/** LaTeX control characters — anything carrying one is math beyond doubt. */
const MATHY_RE = /[\\^_{}]/;

/**
 * Whether a `$...$` run is math rather than a pair of currency amounts.
 *
 * "$5 and $10" pairs into the body "5 and ", which has spaces and no LaTeX
 * control character; "$O(n)$" is short and unbroken, which prose between two
 * prices never is.
 */
function looksLikeMath(body: string): boolean {
  return MATHY_RE.test(body) || (body.length <= 16 && !/\s/.test(body));
}

/**
 * Text with inline `$...$` typeset by KaTeX.
 *
 * Report strings carry inline math ("a $d_k$-dimensional key"), which would
 * otherwise show its delimiters. A `$` pair that looks like prose or money is
 * left exactly as written — rendering "$5 and $10" as math is worse than
 * showing a dollar sign.
 */
export function RichText({ text }: { text: string }) {
  const parts = useMemo(() => {
    const out: { math: boolean; value: string }[] = [];
    let last = 0;
    for (const match of text.matchAll(INLINE_MATH_RE)) {
      const body = match[1].trim();
      if (!looksLikeMath(body)) continue;
      const start = match.index ?? 0;
      if (start > last) out.push({ math: false, value: text.slice(last, start) });
      out.push({ math: true, value: body });
      last = start + match[0].length;
    }
    if (last < text.length) out.push({ math: false, value: text.slice(last) });
    return out;
  }, [text]);

  return (
    <>
      {parts.map((part, i) =>
        part.math ? <InlineMath key={i} latex={part.value} /> : <span key={i}>{part.value}</span>,
      )}
    </>
  );
}

/** A single inline symbol or expression, e.g. a row of a formula's symbol table. */
export function InlineMath({ latex }: { latex: string }) {
  const html = useMemo(() => renderMath(latex, false), [latex]);
  if (!html) return <code className="font-mono text-[0.92em]">{latex}</code>;
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/** A display equation. Falls back to monospace source if KaTeX cannot parse it. */
export function BlockMath({ latex }: { latex: string }) {
  const html = useMemo(() => renderMath(latex, true), [latex]);
  if (!html) {
    return (
      <pre className="overflow-x-auto rounded-lg bg-muted/60 px-3 py-2 font-mono text-[12px] text-foreground/80">
        {latex}
      </pre>
    );
  }
  return (
    <div
      className="overflow-x-auto overflow-y-hidden py-1 text-[15px]"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function renderMath(latex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(latex, {
      displayMode,
      throwOnError: false,
      strict: "ignore",
      trust: false,
      output: "html",
    });
  } catch {
    return "";
  }
}

/** The clickable `p.X` chip that jumps the PDF pane to that page. */
export function PageBadge({
  page,
  onCite,
  variant = "inline",
}: {
  page: number;
  onCite?: (page: number) => void;
  variant?: "inline" | "cell";
}) {
  if (page <= 0) return null;
  return (
    <button
      type="button"
      onClick={() => onCite?.(page)}
      className={
        variant === "inline"
          ? "ml-1.5 rounded-md border border-primary/25 bg-accent/60 px-1 py-px align-[0.05em] font-mono text-[11px] text-primary transition-colors hover:bg-accent"
          : "rounded-md border border-primary/25 bg-accent/60 px-1.5 py-0.5 font-mono text-[11px] text-primary transition-colors hover:bg-accent"
      }
    >
      p.{page}
    </button>
  );
}

/** A bullet list of claims, each with a clickable page badge. */
export function ClaimList({
  claims,
  onCite,
}: {
  claims: { text: string; page: number }[];
  onCite?: (page: number) => void;
}) {
  return (
    <ul className="space-y-1.5">
      {claims.map((c, i) => (
        <li key={i} className="flex gap-2 text-[13.5px] leading-relaxed">
          <span className="mt-[0.45rem] h-1 w-1 shrink-0 rounded-full bg-muted-foreground/60" />
          <span className="min-w-0">
            <RichText text={c.text} />
            <PageBadge page={c.page} onCite={onCite} />
          </span>
        </li>
      ))}
    </ul>
  );
}

/** A titled block of the report. `id` makes it a scroll target for section nav. */
export function Section({
  id,
  title,
  subtitle,
  children,
}: {
  id?: string;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-16 pt-7 first:pt-1">
      <h2 className="font-display text-[17px] font-semibold tracking-tight">{title}</h2>
      {subtitle && <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </section>
  );
}

/**
 * Sticky in-report navigation that tracks the section under the reading pane's
 * top edge. Long reports (Sphere, Lens) get one; short ones do not need it.
 */
export function SectionNav({
  sections,
  idPrefix,
}: {
  sections: { id: string; label: string }[];
  idPrefix: string;
}) {
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const observed = sections
      .map((s) => document.getElementById(`${idPrefix}-sec-${s.id}`))
      .filter((el): el is HTMLElement => el !== null);
    if (observed.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) {
          setActive(visible[0].target.id.replace(`${idPrefix}-sec-`, ""));
        }
      },
      { rootMargin: "-64px 0px -65% 0px", threshold: 0 },
    );
    observed.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [sections, idPrefix]);

  if (sections.length < 2) return null;

  return (
    <nav className="no-scrollbar sticky top-0 z-20 -mx-1 mb-1 flex gap-1 overflow-x-auto border-b border-border bg-background/85 px-1 py-2 backdrop-blur">
      {sections.map((s) => (
        <button
          key={s.id}
          type="button"
          onClick={() =>
            document
              .getElementById(`${idPrefix}-sec-${s.id}`)
              ?.scrollIntoView({ behavior: "smooth", block: "start" })
          }
          className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
            active === s.id
              ? "bg-accent text-accent-foreground"
              : "text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
        >
          {s.label}
        </button>
      ))}
    </nav>
  );
}

/** Headline counts above a report — "3 formulas, 5 tables". */
export function StatTiles({ stats }: { stats: { label: string; value: string }[] }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {stats.map((s) => (
        <div key={s.label} className="rounded-xl border border-border bg-card px-3 py-2.5">
          <p className="font-display text-xl font-semibold tabular-nums leading-none">{s.value}</p>
          <p className="mt-1.5 text-xs text-muted-foreground">{s.label}</p>
        </div>
      ))}
    </div>
  );
}

/** A 0-3 rating as a labelled card with segment bars. */
export function MeterCard({
  label,
  value,
  max = 3,
  invert = false,
}: {
  label: string;
  value: number;
  max?: number;
  invert?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-card px-2.5 py-2">
      <p className="truncate text-[11px] text-muted-foreground" title={label}>
        {label}
      </p>
      <p className="mt-0.5 font-mono text-[15px] tabular-nums">
        {value}
        <span className="text-[11px] text-muted-foreground">/{max}</span>
      </p>
      <div className="mt-1 flex gap-0.5" aria-hidden>
        {Array.from({ length: max }, (_, i) => (
          <span
            key={i}
            className={`h-1 flex-1 rounded-full ${
              i < value ? (invert ? "bg-destructive/60" : "bg-primary/70") : "bg-border"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

/** One quantified result. Snap and Lens share the shape so they share the table. */
export interface ReportFinding {
  metric: string;
  dataset: string;
  value: string;
  baseline: string;
  delta: string;
  page: number;
  note: string;
}

/** Quantified results as a real table — the numbers, not "significantly better". */
export function FindingsTable({
  findings,
  emptyText,
  onCite,
}: {
  findings: ReportFinding[];
  emptyText: string;
  onCite?: (page: number) => void;
}) {
  const { t } = useTranslation();
  const labels = {
    metric: t("report.col.metric"),
    dataset: t("report.col.dataset"),
    value: t("report.col.value"),
    baseline: t("report.col.baseline"),
    delta: t("report.col.delta"),
    page: t("report.col.page"),
  };

  if (findings.length === 0) {
    return <p className="text-[13px] italic text-muted-foreground">{emptyText}</p>;
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full min-w-[38rem] border-collapse text-[13px]">
        <thead>
          <tr className="bg-muted/60 text-left text-xs text-muted-foreground">
            <th className="px-3 py-2 font-medium">{labels.metric}</th>
            <th className="px-3 py-2 font-medium">{labels.dataset}</th>
            <th className="px-3 py-2 text-right font-medium">{labels.value}</th>
            <th className="px-3 py-2 text-right font-medium">{labels.baseline}</th>
            <th className="px-3 py-2 text-right font-medium">{labels.delta}</th>
            <th className="px-3 py-2 font-medium">{labels.page}</th>
          </tr>
        </thead>
        <tbody>
          {findings.map((f, i) => (
            <tr key={i} className="border-t border-border/70 align-top">
              <td className="px-3 py-2 font-medium">{f.metric}</td>
              <td className="px-3 py-2 text-foreground/80">{f.dataset || "—"}</td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">{f.value}</td>
              <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                {f.baseline || "—"}
              </td>
              <td
                className={`px-3 py-2 text-right font-mono tabular-nums ${
                  f.delta.startsWith("+") ? "text-success" : f.delta.startsWith("-") ? "text-destructive" : ""
                }`}
              >
                {f.delta || "—"}
              </td>
              <td className="px-3 py-2">
                {f.page > 0 ? (
                  <PageBadge page={f.page} onCite={onCite} variant="cell" />
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {findings.some((f) => f.note) && (
        <ul className="space-y-0.5 border-t border-border/70 bg-muted/30 px-3 py-2 text-[12px] text-muted-foreground">
          {findings
            .filter((f) => f.note)
            .map((f, i) => (
              <li key={i}>
                <span className="font-medium text-foreground/70">{f.metric}</span>: {f.note}
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}
