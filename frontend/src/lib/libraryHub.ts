import type { AgentMode, SessionPaper } from "./agent";
import type { RecentRunResponse } from "./types";

export interface LibraryHubItem {
  /** Stable key: `paper_id` when a PDF exists, otherwise `lit:<literature_id>`. */
  key: string;
  paperId: string;
  literatureId: string;
  title: string;
  year: number;
  yearKnown: boolean;
  venue: string;
  doi: string;
  arxivId: string;
  availability: SessionPaper["availability"] | "pdf_ready";
  /** Latest conversation holding this paper or its reports, when known. */
  sessionId: string;
  updatedAt: string;
  runs: RecentRunResponse[];
}

function normTime(ts: string | null | undefined): string {
  return (ts || "").trim();
}

/**
 * Merge papers attached across conversations with mode runs (both conversational
 * and standalone) into one deduplicated list ordered by most recent activity.
 */
export function mergeLibraryItems(
  papers: SessionPaper[],
  runs: RecentRunResponse[],
): LibraryHubItem[] {
  const byPaperId = new Map<string, LibraryHubItem>();
  const metadataOnly: LibraryHubItem[] = [];

  for (const paper of papers) {
    const item: LibraryHubItem = {
      key: paper.paper_id || `lit:${paper.literature_id}`,
      paperId: paper.paper_id || "",
      literatureId: paper.literature_id || "",
      title: paper.title || "",
      year: paper.year || 0,
      yearKnown: Boolean(paper.year_known),
      venue: paper.venue || "",
      doi: paper.doi || "",
      arxivId: paper.arxiv_id || "",
      availability: paper.availability,
      sessionId: paper.session_id || "",
      updatedAt: normTime(paper.updated_at),
      runs: [],
    };
    if (paper.paper_id) {
      const existing = byPaperId.get(paper.paper_id);
      if (!existing) {
        byPaperId.set(paper.paper_id, item);
      } else {
        if (!existing.title && item.title) existing.title = item.title;
        if (item.availability === "parsed") existing.availability = "parsed";
        if (item.updatedAt > existing.updatedAt) existing.updatedAt = item.updatedAt;
      }
    } else {
      metadataOnly.push(item);
    }
  }

  for (const run of runs) {
    if (!run.paper_id) continue;
    const runTime = normTime(run.finished_at || run.started_at);
    let item = byPaperId.get(run.paper_id);
    if (!item) {
      item = {
        key: run.paper_id,
        paperId: run.paper_id,
        literatureId: "",
        title: run.paper_title || "",
        year: 0,
        yearKnown: false,
        venue: "",
        doi: "",
        arxivId: "",
        availability: run.status === "done" ? "parsed" : "pdf_ready",
        sessionId: run.agent_session_id || "",
        updatedAt: runTime,
        runs: [],
      };
      byPaperId.set(run.paper_id, item);
    } else {
      if (!item.title && run.paper_title) item.title = run.paper_title;
      if (!item.sessionId && run.agent_session_id) item.sessionId = run.agent_session_id;
      if (run.status === "done") item.availability = "parsed";
      if (runTime > item.updatedAt) item.updatedAt = runTime;
    }
    item.runs.push(run);
  }

  const all = [...byPaperId.values(), ...metadataOnly];
  for (const item of all) {
    item.runs.sort((a, b) =>
      normTime(b.finished_at || b.started_at).localeCompare(normTime(a.finished_at || a.started_at)),
    );
  }
  all.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  return all;
}

/**
 * Whether the composer may submit a turn.
 *
 * Normal chat (`auto`) needs non-empty text; a whole-paper report mode
 * (`snap` / `lens` / `sphere`) can also start with empty text once at least
 * one readable PDF is attached, matching the optional-focus placeholder.
 */
export function canSubmitTurn(opts: {
  draft: string;
  mode: AgentMode;
  hasReadablePaper: boolean;
  sending?: boolean;
  streaming?: boolean;
  attaching?: boolean;
  tooLong?: boolean;
}): boolean {
  if (opts.sending || opts.streaming || opts.attaching || opts.tooLong) return false;
  if (opts.draft.trim().length > 0) return true;
  return opts.mode !== "auto" && opts.hasReadablePaper;
}

/**
 * Resolve the user message sent for a turn: the typed text when present, or
 * the mode's default report request when running a one-click mode report.
 */
export function resolveTurnPrompt(
  draft: string,
  mode: AgentMode,
  defaultPrompt: (mode: Exclude<AgentMode, "auto">) => string,
): string {
  const trimmed = draft.trim();
  if (trimmed) return trimmed;
  if (mode !== "auto") return defaultPrompt(mode);
  return "";
}
