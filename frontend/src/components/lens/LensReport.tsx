"use client";

/**
 * Logic Lens as cards.
 *
 * The Markdown report is the deep read; this is its index. Each card restates
 * one thing the report established — a pipeline stage, a formula and what it
 * computes, a measured result — and carries the page the report cited, so a
 * reader can scan the method, jump to the PDF, and only then decide to read the
 * prose. Anything the digest could not fill simply does not render: an empty
 * card would imply the paper lacks something the report may well have covered.
 */

import { useMemo } from "react";
import { IconCheck } from "@/components/icons";
import {
  BlockMath,
  ClaimList,
  FindingsTable,
  InlineMath,
  MeterCard,
  PageBadge,
  RichText,
  Section,
  SectionNav,
  StatTiles,
} from "@/components/report/primitives";
import { useTranslation } from "@/lib/i18n";
import type {
  LensAlgorithm,
  LensData,
  LensDataset,
  LensFigure,
  LensFormula,
  LensReproducibility,
  LensStage,
} from "@/lib/lens";

type SectionId = "overview" | "method" | "experiments" | "assessment";

/** A labelled sub-heading inside a section. */
function SubHead({ children }: { children: React.ReactNode }) {
  return <p className="mb-1.5 text-xs font-medium text-muted-foreground">{children}</p>;
}

/** A short prose block with its own label — problem, gap, and the like. */
function LabeledBlock({ label, text }: { label: string; text: string }) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <p className="mt-1 text-[13.5px] leading-relaxed text-foreground/90">
        <RichText text={text} />
      </p>
    </div>
  );
}

/** The paper's own architecture diagram, as extracted from the PDF. */
function FigureCard({ figure, onCite }: { figure: LensFigure; onCite?: (page: number) => void }) {
  return (
    <figure className="rounded-xl border border-border bg-card p-3">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={figure.url}
        alt={figure.caption}
        loading="lazy"
        className="mx-auto max-h-[24rem] max-w-full rounded-lg border border-border bg-white object-contain"
      />
      {(figure.caption || figure.page > 0) && (
        <figcaption className="mt-2 text-[12px] leading-relaxed text-muted-foreground">
          {figure.caption}
          <PageBadge page={figure.page} onCite={onCite} />
        </figcaption>
      )}
    </figure>
  );
}

/** The end-to-end pipeline, in data-flow order. */
function PipelineList({ stages, onCite }: { stages: LensStage[]; onCite?: (page: number) => void }) {
  return (
    <ol className="space-y-2">
      {stages.map((stage, i) => (
        <li key={i} className="flex gap-3">
          <span className="relative flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-border bg-muted font-mono text-[11px] text-muted-foreground">
            {i + 1}
            {i < stages.length - 1 && (
              <span className="absolute left-1/2 top-full h-[calc(100%+0.25rem)] w-px -translate-x-1/2 bg-border" />
            )}
          </span>
          <div className="min-w-0 pb-1">
            <p className="text-[13.5px] font-medium leading-snug">
              <RichText text={stage.name} />
              <PageBadge page={stage.page} onCite={onCite} />
            </p>
            {stage.role && (
              <p className="mt-0.5 text-[13px] leading-relaxed text-foreground/75">
                <RichText text={stage.role} />
              </p>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

/** A key equation: typeset, explained, with its symbols named. */
function FormulaCard({
  formula,
  fallbackName,
  onCite,
}: {
  formula: LensFormula;
  fallbackName: string;
  onCite?: (page: number) => void;
}) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3.5">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <h3 className="text-[13.5px] font-semibold">{formula.name || fallbackName}</h3>
        <PageBadge page={formula.page} onCite={onCite} />
      </div>
      <div className="mt-2">
        <BlockMath latex={formula.latex} />
      </div>
      {formula.role && (
        <p className="mt-2 text-[13px] leading-relaxed text-foreground/80">
          <RichText text={formula.role} />
        </p>
      )}
      {formula.symbols.length > 0 && (
        <dl className="mt-3 grid gap-x-5 gap-y-1 border-t border-border/60 pt-2.5 sm:grid-cols-2">
          {formula.symbols.map((s, i) => (
            <div key={i} className="flex gap-2 text-[12.5px]">
              <dt className="shrink-0 text-primary">
                <InlineMath latex={s.symbol} />
              </dt>
              <dd className="min-w-0 text-muted-foreground">
                <RichText text={s.meaning} />
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

/** The paper's procedure, step by step, with the report's annotations. */
function AlgorithmCard({
  algorithm,
  fallbackName,
  complexityLabel,
  onCite,
}: {
  algorithm: LensAlgorithm;
  fallbackName: string;
  complexityLabel: string;
  onCite?: (page: number) => void;
}) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3.5">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <h3 className="text-[13.5px] font-semibold">{algorithm.name || fallbackName}</h3>
        <PageBadge page={algorithm.page} onCite={onCite} />
        {algorithm.complexity && (
          <span className="ml-auto rounded-md border border-border bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
            {complexityLabel} {algorithm.complexity}
          </span>
        )}
      </div>
      <ol className="mt-3 space-y-2">
        {algorithm.steps.map((step, i) => (
          <li key={i} className="flex gap-2.5">
            <span className="mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-[11px] text-primary">
              {i + 1}
            </span>
            <div className="min-w-0">
              <p className="text-[13px] leading-relaxed">
                <RichText text={step.step} />
              </p>
              {step.note && (
                <p className="mt-0.5 text-[12px] leading-relaxed text-muted-foreground">
                  <RichText text={step.note} />
                </p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Datasets with the metrics computed on them and what those metrics mean. */
function DatasetTable({
  datasets,
  onCite,
}: {
  datasets: LensDataset[];
  onCite?: (page: number) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full min-w-[32rem] border-collapse text-[13px]">
        <thead>
          <tr className="bg-muted/60 text-left text-xs text-muted-foreground">
            <th className="px-3 py-2 font-medium">{t("lens.col.dataset")}</th>
            <th className="px-3 py-2 font-medium">{t("lens.col.metrics")}</th>
            <th className="px-3 py-2 font-medium">{t("lens.col.measures")}</th>
            <th className="px-3 py-2 font-medium">{t("report.col.page")}</th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((d, i) => (
            <tr key={i} className="border-t border-border/70 align-top">
              <td className="px-3 py-2 font-medium">{d.name}</td>
              <td className="px-3 py-2 text-foreground/80">{d.metrics || "—"}</td>
              <td className="px-3 py-2 text-foreground/75">
                {d.measures ? <RichText text={d.measures} /> : "—"}
              </td>
              <td className="px-3 py-2">
                {d.page > 0 ? (
                  <PageBadge page={d.page} onCite={onCite} variant="cell" />
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** What a reader would have to reconstruct themselves, and what they wouldn't. */
function ReproCard({ repro }: { repro: LensReproducibility }) {
  const { t } = useTranslation();
  return (
    <div className="grid items-start gap-2.5 sm:grid-cols-[8.5rem_1fr]">
      <MeterCard label={t("lens.repro.score")} value={repro.score} />
      <div className="space-y-2 rounded-lg border border-border bg-card px-3 py-2.5">
        {repro.available.length > 0 && (
          <div>
            <SubHead>{t("lens.repro.available")}</SubHead>
            <div className="flex flex-wrap gap-1.5">
              {repro.available.map((item, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 rounded-md border border-success/30 bg-success/10 px-2 py-0.5 text-[12px] text-success"
                >
                  <IconCheck className="text-[11px]" />
                  {item}
                </span>
              ))}
            </div>
          </div>
        )}
        {repro.missing.length > 0 && (
          <div>
            <SubHead>{t("lens.repro.missing")}</SubHead>
            <div className="flex flex-wrap gap-1.5">
              {repro.missing.map((item, i) => (
                <span
                  key={i}
                  className="inline-flex items-center rounded-md border border-destructive/25 bg-destructive/5 px-2 py-0.5 text-[12px] text-destructive"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function LensReport({
  data,
  onCitationClick,
}: {
  data: LensData;
  onCitationClick?: (page: number) => void;
}) {
  const { t } = useTranslation();
  const d = data.digest;

  const hasOverview = Boolean(d.problem || d.gap || d.contributions.length);
  const hasMethod = Boolean(
    data.key_figures.length || d.pipeline.length || d.formulas.length || d.algorithm,
  );
  const hasExperiments = Boolean(
    d.datasets.length || d.setup.length || d.findings.length || d.takeaways.length,
  );
  const hasAssessment = Boolean(
    d.why_it_works.length ||
      d.limitations.length ||
      d.open_questions.length ||
      d.reproducibility.score > 0 ||
      d.reproducibility.available.length ||
      d.reproducibility.missing.length,
  );

  const sections = useMemo(() => {
    const list: { id: SectionId; label: string }[] = [];
    if (hasOverview) list.push({ id: "overview", label: t("lens.sec.overview") });
    if (hasMethod) list.push({ id: "method", label: t("lens.sec.method") });
    if (hasExperiments) list.push({ id: "experiments", label: t("lens.sec.experiments") });
    if (hasAssessment) list.push({ id: "assessment", label: t("lens.sec.assessment") });
    return list;
  }, [hasOverview, hasMethod, hasExperiments, hasAssessment, t]);

  return (
    <div className="pb-10">
      <SectionNav sections={sections} idPrefix="lens" />

      {/* Headline — the central insight, which is what the deep read is for. */}
      <div className="mt-4 rounded-2xl border border-l-[3px] border-border border-l-primary/70 bg-card p-5">
        <p className="text-xs font-medium text-muted-foreground">{t("lens.core_idea")}</p>
        {d.core_idea ? (
          <p className="mt-1.5 text-[14.5px] leading-relaxed text-foreground/90">
            <RichText text={d.core_idea} />
          </p>
        ) : (
          <p className="mt-1.5 text-[14.5px] leading-relaxed text-foreground/90">{data.title}</p>
        )}
        {data.citation_audit && data.citation_audit.claims_total > 0 && (
          <div className="mt-3.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <IconCheck className="text-[11px]" />
              {t("report.audit.coverage", {
                pct: String(Math.round(data.citation_audit.coverage * 100)),
              })}
            </span>
          </div>
        )}
      </div>

      <div className="mt-3">
        <StatTiles
          stats={[
            { label: t("lens.stat.formulas"), value: String(d.formulas.length) },
            { label: t("lens.stat.stages"), value: String(d.pipeline.length) },
            { label: t("lens.stat.datasets"), value: String(d.datasets.length) },
            { label: t("lens.stat.findings"), value: String(d.findings.length) },
          ]}
        />
      </div>

      {hasOverview && (
        <Section id="lens-sec-overview" title={t("lens.sec.overview")}>
          {(d.problem || d.gap) && (
            <div className="grid gap-2.5 sm:grid-cols-2">
              {d.problem && <LabeledBlock label={t("lens.problem")} text={d.problem} />}
              {d.gap && <LabeledBlock label={t("lens.gap")} text={d.gap} />}
            </div>
          )}
          {d.contributions.length > 0 && (
            <div className={d.problem || d.gap ? "mt-4" : ""}>
              <SubHead>{t("lens.contributions")}</SubHead>
              <ClaimList claims={d.contributions} onCite={onCitationClick} />
            </div>
          )}
        </Section>
      )}

      {hasMethod && (
        <Section id="lens-sec-method" title={t("lens.sec.method")}>
          {data.key_figures.length > 0 && (
            <div className="mb-4 space-y-3">
              {data.key_figures.map((fig, i) => (
                <FigureCard key={i} figure={fig} onCite={onCitationClick} />
              ))}
            </div>
          )}

          {d.pipeline.length > 0 && (
            <div className="mb-4">
              <SubHead>{t("lens.pipeline")}</SubHead>
              <PipelineList stages={d.pipeline} onCite={onCitationClick} />
            </div>
          )}

          {d.formulas.length > 0 && (
            <div className="mb-4">
              <SubHead>{t("lens.formulas")}</SubHead>
              <div className="space-y-2.5">
                {d.formulas.map((f, i) => (
                  <FormulaCard
                    key={i}
                    formula={f}
                    fallbackName={t("lens.formula_n", { n: String(i + 1) })}
                    onCite={onCitationClick}
                  />
                ))}
              </div>
            </div>
          )}

          {d.algorithm && (
            <div>
              <SubHead>{t("lens.algorithm")}</SubHead>
              <AlgorithmCard
                algorithm={d.algorithm}
                fallbackName={t("lens.algorithm")}
                complexityLabel={t("lens.complexity")}
                onCite={onCitationClick}
              />
            </div>
          )}
        </Section>
      )}

      {hasExperiments && (
        <Section id="lens-sec-experiments" title={t("lens.sec.experiments")}>
          {d.datasets.length > 0 && (
            <div className="mb-4">
              <SubHead>{t("lens.datasets")}</SubHead>
              <DatasetTable datasets={d.datasets} onCite={onCitationClick} />
            </div>
          )}

          {d.setup.length > 0 && (
            <div className="mb-4">
              <SubHead>{t("lens.setup")}</SubHead>
              <ClaimList claims={d.setup} onCite={onCitationClick} />
            </div>
          )}

          <div className="mb-4">
            <SubHead>{t("lens.findings")}</SubHead>
            <FindingsTable
              findings={d.findings}
              emptyText={t("lens.no_findings")}
              onCite={onCitationClick}
            />
          </div>

          {d.takeaways.length > 0 && (
            <div>
              <SubHead>{t("lens.takeaways")}</SubHead>
              <ClaimList claims={d.takeaways} onCite={onCitationClick} />
            </div>
          )}
        </Section>
      )}

      {hasAssessment && (
        <Section id="lens-sec-assessment" title={t("lens.sec.assessment")}>
          {d.why_it_works.length > 0 && (
            <div className="mb-4">
              <SubHead>{t("lens.why")}</SubHead>
              <ClaimList claims={d.why_it_works} onCite={onCitationClick} />
            </div>
          )}

          {d.limitations.length > 0 && (
            <div className="mb-4">
              <SubHead>{t("lens.limitations")}</SubHead>
              <ClaimList claims={d.limitations} onCite={onCitationClick} />
            </div>
          )}

          {(d.reproducibility.score > 0 ||
            d.reproducibility.available.length > 0 ||
            d.reproducibility.missing.length > 0) && (
            <div className="mb-4">
              <SubHead>{t("lens.repro")}</SubHead>
              <ReproCard repro={d.reproducibility} />
            </div>
          )}

          {d.open_questions.length > 0 && (
            <div>
              <SubHead>{t("lens.open_questions")}</SubHead>
              <ClaimList claims={d.open_questions} onCite={onCitationClick} />
            </div>
          )}
        </Section>
      )}
    </div>
  );
}
