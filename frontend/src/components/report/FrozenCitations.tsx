"use client";

import { useCallback, type ReactNode } from "react";
import { CitationBadge } from "@/components/agent/ChatMessage";
import { frozenEvidence, type CitationTable } from "@/lib/reportCitations";

/**
 * `renderEvidence` for a report written as an answer: each `[ev_…]` becomes
 * the number and passage frozen in the report's citation table, so whoever
 * holds the link sees what the author saw. `undefined` for a report without a
 * table — its `[p.X]` citations render as page badges as before.
 */
export function useFrozenCitations(
  table: CitationTable,
  onJumpToPage?: (paperId: string, page: number) => void,
  /** Only this paper's passages offer a jump (a page showing one PDF). */
  jumpablePaper?: string,
): ((evidenceId: string) => ReactNode) | undefined {
  const render = useCallback(
    (evidenceId: string) => {
      const entry = table[evidenceId];
      // The snapshot removes ids it could not resolve; anything else is noise.
      if (!entry) return null;
      return (
        <CitationBadge
          evidenceId={evidenceId}
          number={entry.n}
          preset={frozenEvidence(evidenceId, entry)}
          onJumpToPage={
            !jumpablePaper || entry.paper_id === jumpablePaper ? onJumpToPage : undefined
          }
        />
      );
    },
    [table, onJumpToPage, jumpablePaper],
  );
  return Object.keys(table).length > 0 ? render : undefined;
}
