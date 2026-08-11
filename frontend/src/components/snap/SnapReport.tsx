"use client";

import { BADGE_BASE, CCF_COLORS, SCI_COLORS } from "@/components/RankBadges";
import { IconCheck, IconExternal } from "@/components/icons";
import { ClaimList, FindingsTable, MeterCard, Section } from "@/components/report/primitives";
import { useTranslation } from "@/lib/i18n";
import {
  TIER_ACCENT,
  formatCount,
  isUnknown,
  paperExternalUrl,
  repoLabel,
  type SnapCodeRepo,
  type SnapData,
  type SnapSignals,
  type VerdictTier,
} from "@/lib/snap";

/** One row of the evidence table: a value plus where it came from. */
function EvidenceRow({
  label,
  children,
  source,
  warn = false,
}: {
  label: string;
  children: React.ReactNode;
  source?: string;
  warn?: boolean;
}) {
  return (
    <div className="grid grid-cols-[7.5rem_1fr] gap-x-3 gap-y-0.5 border-b border-border/60 py-2 last:border-0 sm:grid-cols-[9rem_1fr_8rem]">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={`min-w-0 text-[13px] ${warn ? "font-semibold text-destructive" : "text-foreground/90"}`}>
        {children}
      </span>
      {source && (
        <span className="col-start-2 text-[11px] text-muted-foreground sm:col-start-3 sm:text-right">
          {source}
        </span>
      )}
    </div>
  );
}

function RepoLink({ repo }: { repo: SnapCodeRepo }) {
  return (
    <a
      href={repo.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group inline-flex flex-wrap items-baseline gap-x-1.5 break-all decoration-primary/40 underline-offset-2 hover:text-primary hover:underline"
    >
      <span className="font-mono text-[12px]">{repoLabel(repo)}</span>
      {repo.probe_ok && repo.stars > 0 && (
        <span className="shrink-0 tabular-nums text-[11px] text-muted-foreground">
          {formatCount(repo.stars)}★
        </span>
      )}
      {repo.probe_ok && repo.last_push && (
        <span className="shrink-0 font-mono text-[11px] text-muted-foreground">{repo.last_push}</span>
      )}
      <IconExternal className="shrink-0 align-[-0.1em] text-[10px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-70" />
    </a>
  );
}

/** Venue / citations / code / access / retraction, each with its source. */
function EvidenceTable({ signals }: { signals: SnapSignals }) {
  const { t } = useTranslation();
  const p = signals.provenance;
  const official = signals.repos.filter((r) => r.is_official);
  const thirdParty = signals.repos.filter((r) => !r.is_official);

  return (
    <div className="rounded-xl border border-border bg-card px-4 py-1">
      <EvidenceRow
        label={t("snap.sig.venue")}
        source={p.sci_rank || p.ccf_rank || p.venue || "—"}
      >
        <span className="flex flex-wrap items-center gap-1.5">
          {signals.venue_normalized || signals.venue ? (
            <span title={signals.venue}>{signals.venue_normalized || signals.venue}</span>
          ) : (
            <span className="text-muted-foreground">{t("snap.unknown")}</span>
          )}
          {signals.year > 0 && <span className="tabular-nums text-muted-foreground">{signals.year}</span>}
          {signals.sci_rank && SCI_COLORS[signals.sci_rank] && (
            <span className={`${BADGE_BASE} ${SCI_COLORS[signals.sci_rank]}`}>SCI {signals.sci_rank}</span>
          )}
          {signals.ccf_rank && CCF_COLORS[signals.ccf_rank] && (
            <span className={`${BADGE_BASE} ${CCF_COLORS[signals.ccf_rank]}`}>CCF {signals.ccf_rank}</span>
          )}
          {signals.is_preprint && (
            <span className={`${BADGE_BASE} border-border bg-muted text-muted-foreground`}>
              {t("snap.sig.preprint")}
            </span>
          )}
          {signals.is_survey && (
            <span className={`${BADGE_BASE} border-border bg-muted text-muted-foreground`}>
              {t("snap.sig.survey")}
            </span>
          )}
        </span>
      </EvidenceRow>

      <EvidenceRow label={t("snap.sig.citations")} source={p.citations || "—"}>
        {isUnknown(signals, "citations") ? (
          <span className="text-muted-foreground">{t("snap.unknown")}</span>
        ) : (
          <span className="tabular-nums">
            {formatCount(signals.cited_by_count)}
            {signals.citations_per_year > 0 && (
              <span className="ml-1.5 text-muted-foreground">
                ({signals.citations_per_year}
                {t("snap.per_year")})
              </span>
            )}
          </span>
        )}
      </EvidenceRow>

      {signals.influential_citation_count > 0 && (
        <EvidenceRow label={t("snap.sig.influential")} source={p.influential_citations || "—"}>
          <span className="tabular-nums">{formatCount(signals.influential_citation_count)}</span>
        </EvidenceRow>
      )}

      {signals.intents.sampled > 0 && (
        <EvidenceRow label={t("snap.sig.intents")} source={p.intents || "—"}>
          <span className="flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
            <span>
              {t("snap.intent.methodology")}{" "}
              <span className="font-mono tabular-nums">{signals.intents.methodology}</span>
            </span>
            <span>
              {t("snap.intent.result")}{" "}
              <span className="font-mono tabular-nums">{signals.intents.result}</span>
            </span>
            <span className="text-muted-foreground">
              {t("snap.intent.background")}{" "}
              <span className="font-mono tabular-nums">{signals.intents.background}</span>
            </span>
            <span className="text-[11px] text-muted-foreground">
              {t("snap.intent.sampled", { n: String(signals.intents.sampled) })}
            </span>
          </span>
        </EvidenceRow>
      )}

      <EvidenceRow
        label={official.length ? t("snap.sig.code") : t("snap.sig.code_third_party")}
        source={p.code || "—"}
      >
        {official.length > 0 ? (
          <span className="flex flex-col gap-0.5">
            {official.map((r) => (
              <RepoLink key={r.url} repo={r} />
            ))}
          </span>
        ) : thirdParty.length > 0 ? (
          <span className="flex flex-col gap-0.5">
            {thirdParty.slice(0, 2).map((r) => (
              <RepoLink key={r.url} repo={r} />
            ))}
          </span>
        ) : (
          <span className="text-muted-foreground">{t("snap.none")}</span>
        )}
      </EvidenceRow>

      <EvidenceRow label={t("snap.sig.oa")} source={p.open_access || "—"}>
        {isUnknown(signals, "open_access") ? (
          <span className="text-muted-foreground">{t("snap.unknown")}</span>
        ) : signals.is_open_access ? (
          signals.oa_url ? (
            <a
              href={signals.oa_url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary hover:underline"
            >
              {t("snap.yes")}
            </a>
          ) : (
            t("snap.yes")
          )
        ) : (
          t("snap.no")
        )}
      </EvidenceRow>

      {signals.is_retracted ? (
        <EvidenceRow label={t("snap.sig.retraction")} source={p.retraction || "—"} warn>
          {t("snap.sig.retracted")}
        </EvidenceRow>
      ) : (
        !isUnknown(signals, "retraction") && (
          <EvidenceRow label={t("snap.sig.retraction")} source="OpenAlex">
            {t("snap.sig.not_retracted")}
          </EvidenceRow>
        )
      )}
    </div>
  );
}

/** The five content-review dimensions as small 0-3 meters. */
function ReviewScores({ data }: { data: SnapData }) {
  const { t } = useTranslation();
  const r = data.content_review;
  if (!r.available) return null;

  const dims: { label: string; value: number; invert?: boolean }[] = [
    { label: t("snap.dim.contribution"), value: r.contribution_strength },
    { label: t("snap.dim.evidence"), value: r.evidence_strength },
    { label: t("snap.dim.novelty"), value: r.novelty },
    { label: t("snap.dim.reproducibility"), value: r.reproducibility },
    { label: t("snap.dim.limitations"), value: r.limitation_severity, invert: true },
  ];

  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-5">
      {dims.map((d) => (
        <MeterCard key={d.label} label={d.label} value={d.value} invert={d.invert} />
      ))}
    </div>
  );
}

const TIER_LABEL_KEY: Record<VerdictTier, string> = {
  must_read: "snap.tier.must_read",
  selective: "snap.tier.selective",
  skip: "snap.tier.skip",
};

export default function SnapReport({
  data,
  onCitationClick,
}: {
  data: SnapData;
  onCitationClick?: (page: number) => void;
}) {
  const { t } = useTranslation();
  const { report, verdict, content_review: review, signals } = data;
  const accent = TIER_ACCENT[verdict.tier] || TIER_ACCENT.selective;
  const headlineSection = review.must_read_sections[0]?.where || "";

  return (
    <div className="pb-10">
      {/* Verdict headline — the answer the reader came for, with its basis. */}
      <div className={`rounded-2xl border border-l-[3px] border-border bg-card p-5 ${accent.rail}`}>
        <div className="flex flex-wrap items-center gap-2.5">
          <span className={`inline-flex items-center rounded-lg border px-2.5 py-1 text-[13px] font-semibold ${accent.chip}`}>
            {t(TIER_LABEL_KEY[verdict.tier])}
          </span>
          {verdict.tier === "selective" && headlineSection && (
            <span className="text-[13px] text-foreground/80">{headlineSection}</span>
          )}
          <a
            href={paperExternalUrl(signals, data.title)}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary hover:underline"
          >
            {t("snap.open_source")}
            <IconExternal className="text-[11px]" />
          </a>
        </div>

        {signals.is_retracted && (
          <p className="mt-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-[13px] font-semibold text-destructive">
            {t("snap.sig.retracted")}
          </p>
        )}

        {report.one_liner && (
          <p className="mt-3 text-[14.5px] leading-relaxed text-foreground/90">{report.one_liner}</p>
        )}

        {/* Score provenance: which of the two readings drove the tier. */}
        <div className="mt-3.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-muted-foreground">
          <span>
            {t("snap.score.content")} <span className="font-mono tabular-nums">{verdict.content_score.toFixed(2)}</span>
          </span>
          <span>
            {t("snap.score.external")} <span className="font-mono tabular-nums">{verdict.external_score.toFixed(2)}</span>
          </span>
          <span>
            {t("snap.score.combined")} <span className="font-mono tabular-nums">{verdict.combined_score.toFixed(2)}</span>
          </span>
          {data.citation_audit && data.citation_audit.claims_total > 0 && (
            <span className="inline-flex items-center gap-1">
              <IconCheck className="text-[11px]" />
              {t("report.audit.coverage", {
                pct: String(Math.round(data.citation_audit.coverage * 100)),
              })}
            </span>
          )}
        </div>

        {!verdict.content_available && (
          <p className="mt-2.5 text-[12px] italic text-muted-foreground">{t("snap.no_review")}</p>
        )}
        {verdict.overrides.includes("strong_content_thin_record") && (
          <p className="mt-2.5 text-[12px] italic text-muted-foreground">{t("snap.override.thin_record")}</p>
        )}
      </div>

      {review.question_relevance >= 0 && review.question_note && (
        <div className="mt-3 rounded-xl border border-primary/25 bg-accent/40 px-4 py-3">
          <p className="text-xs font-medium text-muted-foreground">{t("snap.for_question")}</p>
          <p className="mt-1 text-[13.5px] leading-relaxed">{review.question_note}</p>
        </div>
      )}

      <Section title={t("snap.sec.evidence")}>
        <EvidenceTable signals={signals} />
        {!signals.resolved && (
          <p className="mt-2 text-[12px] italic text-muted-foreground">{t("snap.unresolved")}</p>
        )}
      </Section>

      {review.available && (
        <Section title={t("snap.sec.review")}>
          <ReviewScores data={data} />
          {review.reasons.length > 0 && (
            <div className="mt-3">
              <ClaimList
                claims={review.reasons.map((r) => ({ text: r, page: 0 }))}
                onCite={onCitationClick}
              />
            </div>
          )}
          {review.red_flags.length > 0 && (
            <div className="mt-3 rounded-xl border border-destructive/25 bg-destructive/5 px-4 py-3">
              <p className="text-xs font-medium text-destructive">{t("snap.red_flags")}</p>
              <ul className="mt-1.5 space-y-1 text-[13px] leading-relaxed">
                {review.red_flags.map((flag, i) => (
                  <li key={i}>• {flag}</li>
                ))}
              </ul>
            </div>
          )}
        </Section>
      )}

      {review.must_read_sections.length > 0 && (
        <Section title={t("snap.sec.worth_time")}>
          <ul className="space-y-2">
            {review.must_read_sections.map((s, i) => (
              <li key={i} className="rounded-xl border border-border bg-card px-3.5 py-2.5">
                <p className="text-[13.5px] font-medium">{s.where}</p>
                {s.why && <p className="mt-0.5 text-[13px] text-foreground/75">{s.why}</p>}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {report.contributions.length > 0 && (
        <Section title={t("snap.sec.contributions")}>
          <ClaimList claims={report.contributions} onCite={onCitationClick} />
        </Section>
      )}

      <Section title={t("snap.sec.findings")}>
        <FindingsTable
          findings={report.findings}
          emptyText={t("snap.no_findings")}
          onCite={onCitationClick}
        />
      </Section>

      {(report.suitable_for || report.limitations.length > 0) && (
        <Section title={t("snap.sec.applicability")}>
          {report.suitable_for && (
            <p className="text-[13.5px] leading-relaxed text-foreground/90">{report.suitable_for}</p>
          )}
          {report.limitations.length > 0 && (
            <div className="mt-3">
              <p className="mb-1.5 text-xs font-medium text-muted-foreground">{t("snap.limitations")}</p>
              <ClaimList claims={report.limitations} onCite={onCitationClick} />
            </div>
          )}
        </Section>
      )}

      {review.target_readers && (
        <Section title={t("snap.sec.decide")}>
          <dl className="space-y-2.5">
            <div>
              <dt className="text-xs font-medium text-muted-foreground">{t("snap.readers")}</dt>
              <dd className="mt-0.5 text-[13.5px] leading-relaxed">{review.target_readers}</dd>
            </div>
          </dl>
        </Section>
      )}
    </div>
  );
}
