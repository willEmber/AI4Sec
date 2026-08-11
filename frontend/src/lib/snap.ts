/**
 * Insight Snap structured payload.
 *
 * The snap subgraph emits a markdown report (used for export) plus a JSON twin
 * in `run_outputs.json_data`. The JSON carries the report as fields — the
 * one-liner, contributions, quantified findings, limitations — alongside the
 * triage verdict and the external evidence that backs it, so the UI can render
 * a scorecard and a findings table instead of a wall of prose.
 *
 * A run whose JSON pass degraded (`report.degraded`) has no fields to render;
 * `parseSnapData` returns null for those and the caller falls back to markdown.
 */

export type VerdictTier = "must_read" | "selective" | "skip";

export interface SnapFinding {
  metric: string;
  dataset: string;
  value: string;
  baseline: string;
  delta: string;
  page: number;
  note: string;
}

export interface SnapClaim {
  text: string;
  page: number;
}

export interface SnapReportData {
  one_liner: string;
  contributions: SnapClaim[];
  findings: SnapFinding[];
  suitable_for: string;
  limitations: SnapClaim[];
  degraded: boolean;
}

export interface SnapCodeRepo {
  url: string;
  host: string;
  slug: string;
  is_official: boolean;
  evidence_page: number;
  evidence_quote: string;
  stars: number;
  last_push: string;
  archived: boolean;
  probe_ok: boolean;
}

export interface SnapCitationIntents {
  background: number;
  methodology: number;
  result: number;
  influential: number;
  /** How many citation edges were inspected — these counts are a sample. */
  sampled: number;
}

export interface SnapSignals {
  doi: string;
  arxiv_id: string;
  openalex_id: string;
  s2_paper_id: string;
  resolved: boolean;
  venue: string;
  venue_normalized: string;
  year: number;
  sci_rank: string;
  ccf_rank: string;
  is_preprint: boolean;
  is_survey: boolean;
  cited_by_count: number;
  citations_per_year: number;
  influential_citation_count: number;
  intents: SnapCitationIntents;
  repos: SnapCodeRepo[];
  is_open_access: boolean;
  oa_url: string;
  is_retracted: boolean;
  venue_score: number;
  citation_score: number;
  code_score: number;
  external_score: number;
  provenance: Record<string, string>;
  unavailable: string[];
}

export interface SnapContentReview {
  contribution_strength: number;
  evidence_strength: number;
  novelty: number;
  reproducibility: number;
  limitation_severity: number;
  must_read_sections: { where: string; why: string }[];
  target_readers: string;
  reasons: string[];
  red_flags: string[];
  question_relevance: number;
  question_note: string;
  available: boolean;
}

export interface SnapVerdict {
  tier: VerdictTier;
  combined_score: number;
  content_score: number;
  external_score: number;
  drivers: string[];
  overrides: string[];
  content_available: boolean;
}

export interface SnapCitationAudit {
  claims_total: number;
  claims_uncited: number;
  coverage: number;
  uncited_samples: string[];
}

export interface SnapContextStats {
  chars: number;
  budget: number;
  slots: Record<string, number>;
  tables: number;
  figures: number;
  equations: number;
  truncated: string[];
}

export interface SnapData {
  mode: string;
  paper_id: string;
  title: string;
  report: SnapReportData;
  signals: SnapSignals;
  content_review: SnapContentReview;
  verdict: SnapVerdict;
  citation_audit: SnapCitationAudit | null;
  context_stats: SnapContextStats | null;
}

const EMPTY_SIGNALS: SnapSignals = {
  doi: "", arxiv_id: "", openalex_id: "", s2_paper_id: "", resolved: false,
  venue: "", venue_normalized: "", year: 0, sci_rank: "", ccf_rank: "",
  is_preprint: false, is_survey: false, cited_by_count: 0, citations_per_year: 0,
  influential_citation_count: 0,
  intents: { background: 0, methodology: 0, result: 0, influential: 0, sampled: 0 },
  repos: [], is_open_access: false, oa_url: "",
  is_retracted: false, venue_score: 0, citation_score: 0, code_score: 0,
  external_score: 0, provenance: {}, unavailable: [],
};

const EMPTY_REVIEW: SnapContentReview = {
  contribution_strength: 0, evidence_strength: 0, novelty: 0, reproducibility: 0,
  limitation_severity: 0, must_read_sections: [], target_readers: "",
  reasons: [], red_flags: [], question_relevance: -1, question_note: "", available: false,
};

/**
 * Parse `json_data` into a snap payload.
 *
 * Returns null for other modes, malformed JSON, reports produced before the
 * structured payload existed, and degraded runs — callers fall back to the
 * markdown renderer in all of those cases.
 */
export function parseSnapData(jsonData: string | undefined | null): SnapData | null {
  if (!jsonData) return null;
  try {
    const parsed = JSON.parse(jsonData) as Partial<SnapData>;
    if (parsed?.mode !== "snap") return null;
    if (!parsed.report || !parsed.verdict) return null;
    if (parsed.report.degraded) return null;

    return {
      mode: "snap",
      paper_id: parsed.paper_id || "",
      title: parsed.title || "",
      report: {
        one_liner: parsed.report.one_liner || "",
        contributions: parsed.report.contributions || [],
        findings: parsed.report.findings || [],
        suitable_for: parsed.report.suitable_for || "",
        limitations: parsed.report.limitations || [],
        degraded: false,
      },
      signals: { ...EMPTY_SIGNALS, ...(parsed.signals || {}) },
      content_review: { ...EMPTY_REVIEW, ...(parsed.content_review || {}) },
      verdict: parsed.verdict,
      citation_audit: parsed.citation_audit || null,
      context_stats: parsed.context_stats || null,
    };
  } catch {
    return null;
  }
}

/** Per-tier accent, shared by the scorecard and the library matrix. */
export const TIER_ACCENT: Record<VerdictTier, { chip: string; rail: string; dot: string }> = {
  must_read: {
    chip: "border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    rail: "border-l-emerald-500/70",
    dot: "bg-emerald-500",
  },
  selective: {
    chip: "border-amber-500/35 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    rail: "border-l-amber-500/70",
    dot: "bg-amber-500",
  },
  skip: {
    chip: "border-slate-500/35 bg-slate-500/10 text-slate-600 dark:text-slate-300",
    rail: "border-l-slate-500/70",
    dot: "bg-slate-500",
  },
};

/** Compact citation counts: 99111 → 99k. */
export function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  if (n >= 1_000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

/** True when a lookup failed, so the UI must say "unknown" rather than show 0. */
export function isUnknown(signals: SnapSignals, key: string): boolean {
  return signals.unavailable.includes(key);
}

/** Best external link for the paper: DOI → arXiv → OpenAlex → OA copy. */
export function paperExternalUrl(signals: SnapSignals, title: string): string {
  if (signals.doi) return `https://doi.org/${signals.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, "")}`;
  if (signals.arxiv_id) return `https://arxiv.org/abs/${signals.arxiv_id.replace(/^arxiv:/i, "")}`;
  if (signals.openalex_id) return `https://openalex.org/${signals.openalex_id}`;
  if (signals.oa_url) return signals.oa_url;
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(title)}`;
}

/** Repo label without the scheme, e.g. `github.com/acme/foonet`. */
export function repoLabel(repo: SnapCodeRepo): string {
  return repo.url.replace(/^https?:\/\//, "");
}
