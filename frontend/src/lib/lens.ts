/**
 * Logic Lens structured payload.
 *
 * Lens is the one mode whose report is prose by design — the argument is the
 * product. So its JSON twin is not an alternative report: it is a digest the
 * backend extracts *from* the finished Markdown (see
 * `backend/app/services/lens_digest.py`), which is what lets the reader toggle
 * between cards and prose and find the same claims, with the same pages, on
 * both sides.
 *
 * `parseLensData` returns null whenever there is nothing to render as cards —
 * runs produced before the digest existed, a disabled or failed digest pass,
 * malformed JSON — and the caller falls back to the Markdown renderer.
 */

export interface LensClaim {
  text: string;
  page: number;
}

export interface LensSymbol {
  symbol: string;
  meaning: string;
}

export interface LensFormula {
  name: string;
  /** LaTeX body, no delimiters — the card typesets it in display mode. */
  latex: string;
  page: number;
  role: string;
  symbols: LensSymbol[];
}

export interface LensStage {
  name: string;
  role: string;
  page: number;
}

export interface LensStep {
  step: string;
  note: string;
}

export interface LensAlgorithm {
  name: string;
  page: number;
  complexity: string;
  steps: LensStep[];
}

export interface LensDataset {
  name: string;
  metrics: string;
  measures: string;
  page: number;
}

export interface LensFinding {
  metric: string;
  dataset: string;
  value: string;
  baseline: string;
  delta: string;
  page: number;
  note: string;
}

export interface LensReproducibility {
  /** 0-3: nothing usable → enough to reproduce the core result. */
  score: number;
  available: string[];
  missing: string[];
}

export interface LensDigest {
  core_idea: string;
  problem: string;
  gap: string;
  contributions: LensClaim[];
  pipeline: LensStage[];
  formulas: LensFormula[];
  algorithm: LensAlgorithm | null;
  datasets: LensDataset[];
  setup: LensClaim[];
  findings: LensFinding[];
  takeaways: LensClaim[];
  why_it_works: LensClaim[];
  limitations: LensClaim[];
  reproducibility: LensReproducibility;
  open_questions: LensClaim[];
  available: boolean;
}

/** An architecture figure, extracted from the PDF rather than from the model. */
export interface LensFigure {
  page: number;
  caption: string;
  url: string;
}

export interface LensCitationAudit {
  claims_total: number;
  claims_uncited: number;
  coverage: number;
  uncited_samples: string[];
}

export interface LensData {
  mode: string;
  paper_id: string;
  title: string;
  num_equations: number;
  num_algorithms: number;
  num_tables: number;
  num_figures: number;
  citation_audit: LensCitationAudit | null;
  digest: LensDigest;
  key_figures: LensFigure[];
}

const EMPTY_DIGEST: LensDigest = {
  core_idea: "", problem: "", gap: "", contributions: [], pipeline: [], formulas: [],
  algorithm: null, datasets: [], setup: [], findings: [], takeaways: [],
  why_it_works: [], limitations: [],
  reproducibility: { score: 0, available: [], missing: [] },
  open_questions: [], available: false,
};

/**
 * Which view a report opens on. The Lens digest is an index of the prose, not a
 * substitute for it — measured at under half the report's text — so a Lens run
 * opens on the prose; the other modes' cards are the report itself.
 */
export function defaultLensView(lensData: LensData | null): "structured" | "markdown" {
  return lensData ? "markdown" : "structured";
}

/** The report's prose, cut at its four top-level headings. */
export interface LensProse {
  overview: string;
  method: string;
  experiments: string;
  assessment: string;
}

/**
 * Split a Lens report into the four parts its prompt fixes, so the card view
 * can carry each part's full text under the cards that index it.
 *
 * Returns null unless the report has exactly four `## ` headings: the parts are
 * matched by position (the headings are translated, and the model rewords
 * them), so any other count means the mapping would be a guess. The caller then
 * shows the report whole.
 */
export function splitLensReport(markdown: string | undefined | null): LensProse | null {
  if (!markdown) return null;
  const preamble: string[] = [];
  const parts: string[][] = [];
  let fenced = false;
  for (const line of markdown.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    if (!fenced && /^##\s+\S/.test(line)) {
      parts.push([]);
      continue;
    }
    (parts.length ? parts[parts.length - 1] : preamble).push(line);
  }
  if (parts.length !== 4) return null;

  // Text before the first heading is usually just the title; anything else is
  // content and stays with the first part rather than being dropped.
  const lead = preamble.filter((line) => !/^#\s/.test(line)).join("\n").trim();
  const [overview, method, experiments, assessment] = parts.map((p) => p.join("\n").trim());
  return {
    overview: lead ? `${lead}\n\n${overview}` : overview,
    method,
    experiments,
    assessment,
  };
}

export function parseLensData(jsonData: string | undefined | null): LensData | null {
  if (!jsonData) return null;
  try {
    const parsed = JSON.parse(jsonData) as Partial<LensData>;
    if (parsed?.mode !== "lens") return null;
    if (!parsed.digest?.available) return null;

    return {
      mode: "lens",
      paper_id: parsed.paper_id || "",
      title: parsed.title || "",
      num_equations: parsed.num_equations || 0,
      num_algorithms: parsed.num_algorithms || 0,
      num_tables: parsed.num_tables || 0,
      num_figures: parsed.num_figures || 0,
      citation_audit: parsed.citation_audit || null,
      digest: { ...EMPTY_DIGEST, ...parsed.digest },
      key_figures: (parsed.key_figures || []).filter((f) => f.url),
    };
  } catch {
    return null;
  }
}
