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
