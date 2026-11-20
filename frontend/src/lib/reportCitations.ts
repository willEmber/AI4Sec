import type { Evidence } from "./agent";

/**
 * A citation of a report written as an answer, frozen when the report was
 * saved (`json_data.citations`, backend `services/report_citations.py`).
 *
 * A report is shared by its link, but evidence is only served to its owner, so
 * a reader of a shared report renders citations from this table and never asks
 * the evidence API. Older reports cite pages (`[p.X]`) and have no table.
 */
export interface FrozenCitation {
  n: number;
  source_level: Evidence["source_level"];
  paper_id: string;
  literature_id?: string;
  title: string;
  page_label: number | null;
  section?: string;
  quote: string;
  source_url: string;
  provider: string;
}

export type CitationTable = Record<string, FrozenCitation>;

/** The table in a run's `json_data`; empty for a report without one. */
export function parseCitationTable(jsonData: string | null | undefined): CitationTable {
  if (!jsonData) return {};
  try {
    const data = JSON.parse(jsonData);
    const table = data && typeof data === "object" ? data.citations : null;
    if (!table || typeof table !== "object" || Array.isArray(table)) return {};
    const out: CitationTable = {};
    for (const [id, entry] of Object.entries(table as Record<string, unknown>)) {
      if (entry && typeof entry === "object" && typeof (entry as FrozenCitation).n === "number") {
        out[id] = entry as FrozenCitation;
      }
    }
    return out;
  } catch {
    return {};
  }
}

/** A frozen citation in the shape the citation popover renders. */
export function frozenEvidence(evidenceId: string, c: FrozenCitation): Evidence {
  return {
    evidence_id: evidenceId,
    session_id: "",
    literature_id: c.literature_id ?? "",
    paper_id: c.paper_id,
    parse_version: "",
    source_level: c.source_level,
    locator: {
      section_id: "",
      section_path: c.section ?? "",
      page_index: c.page_label == null ? null : c.page_label - 1,
      page_end_index: null,
      block_start: null,
      block_end: null,
      node_id: "",
      bbox: [],
      page_label: c.page_label,
      page_end_label: null,
    },
    quote: c.quote,
    content_hash: "",
    source_url: c.source_url,
    provider: c.provider,
    retrieved_at: "",
    literature: c.title
      ? {
          literature_id: c.literature_id ?? "",
          title: c.title,
          year: null,
          year_known: false,
          venue: "",
          doi: "",
          arxiv_id: "",
        }
      : undefined,
  };
}
