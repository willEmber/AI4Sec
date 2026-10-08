import { getEvidence } from "./agent";
import type { Evidence } from "./agent";

// Evidence is an immutable snapshot, so one fetch per id serves the citation
// popover, the list under an answer and the session's source tab alike.
const cache = new Map<string, Promise<Evidence>>();

export function loadEvidence(evidenceId: string): Promise<Evidence> {
  let pending = cache.get(evidenceId);
  if (!pending) {
    pending = getEvidence(evidenceId);
    cache.set(evidenceId, pending);
    // A failure is not remembered: the next click may succeed.
    pending.catch(() => cache.delete(evidenceId));
  }
  return pending;
}

/**
 * Several at once, a few requests at a time. An id that cannot be resolved
 * comes back as `null` rather than failing the rest.
 */
export async function loadEvidenceMany(
  evidenceIds: string[],
  concurrency = 4,
): Promise<Map<string, Evidence | null>> {
  const results = new Map<string, Evidence | null>();
  const queue = [...new Set(evidenceIds)];
  const worker = async () => {
    for (let id = queue.shift(); id !== undefined; id = queue.shift()) {
      try {
        results.set(id, await loadEvidence(id));
      } catch {
        results.set(id, null);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, queue.length) }, worker));
  return results;
}
