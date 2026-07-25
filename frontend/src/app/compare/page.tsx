"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { getRunOutput, listRecentRuns } from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import SnapMatrix, { type MatrixEntry } from "@/components/snap/SnapMatrix";
import { parseSnapData } from "@/lib/snap";
import type { RecentRunResponse } from "@/lib/types";

/**
 * Triage workbench: pick several finished Insight Snap runs and compare them
 * side by side.
 *
 * Runs are loaded once and parsed client-side; a run whose structured payload is
 * missing (an older run, or one whose JSON pass degraded) is listed as
 * unavailable rather than silently dropped, so it is clear why it cannot be
 * compared.
 */
export default function ComparePage() {
  const { t } = useTranslation();
  const [runs, setRuns] = useState<RecentRunResponse[]>([]);
  const [entries, setEntries] = useState<Map<string, MatrixEntry>>(new Map());
  const [unavailable, setUnavailable] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const recent = await listRecentRuns(40).catch(() => [] as RecentRunResponse[]);
      const snapRuns = recent.filter((r) => r.mode === "snap" && r.status === "done");
      if (cancelled) return;
      setRuns(snapRuns);

      const loaded = new Map<string, MatrixEntry>();
      const missing = new Set<string>();
      await Promise.all(
        snapRuns.map(async (run) => {
          const output = await getRunOutput(run.run_id).catch(() => null);
          const data = output ? parseSnapData(output.json_data) : null;
          if (data) {
            loaded.set(run.run_id, { runId: run.run_id, paperId: run.paper_id, data });
          } else {
            missing.add(run.run_id);
          }
        }),
      );
      if (cancelled) return;

      setEntries(loaded);
      setUnavailable(missing);
      // Preselect the first few comparable runs so the matrix is not empty.
      setSelected(new Set([...loaded.keys()].slice(0, 5)));
      setLoading(false);
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const toggle = useCallback((runId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  }, []);

  const visible = useMemo(
    () => [...selected].map((id) => entries.get(id)).filter((e): e is MatrixEntry => !!e),
    [selected, entries],
  );

  return (
    <div className="mx-auto max-w-6xl px-5 py-8 sm:px-8">
      <h1 className="font-display text-2xl font-semibold tracking-tight">{t("matrix.title")}</h1>
      <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        {t("matrix.subtitle")}
      </p>

      {loading ? (
        <div className="mt-10 flex items-center gap-3 text-sm text-muted-foreground">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          {t("matrix.loading")}
        </div>
      ) : entries.size === 0 ? (
        <div className="mt-8 rounded-2xl border border-border bg-card p-8 text-center">
          <p className="text-sm text-muted-foreground">{t("matrix.empty")}</p>
          <a
            href="/upload"
            className="mt-4 inline-flex rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
          >
            {t("nav.upload")}
          </a>
        </div>
      ) : (
        <>
          <div className="mt-6">
            <p className="mb-2 text-xs font-medium text-muted-foreground">
              {t("matrix.pick", { n: String(selected.size), total: String(entries.size) })}
            </p>
            <div className="flex flex-wrap gap-2">
              {runs.map((run) => {
                const entry = entries.get(run.run_id);
                const isMissing = unavailable.has(run.run_id);
                const isOn = selected.has(run.run_id);
                return (
                  <button
                    key={run.run_id}
                    type="button"
                    disabled={isMissing}
                    onClick={() => toggle(run.run_id)}
                    aria-pressed={isOn}
                    title={isMissing ? t("matrix.unavailable") : run.paper_title}
                    className={`max-w-[20rem] truncate rounded-lg border px-2.5 py-1.5 text-left text-xs transition-colors ${
                      isMissing
                        ? "cursor-not-allowed border-dashed border-border text-muted-foreground/60"
                        : isOn
                          ? "border-primary/45 bg-accent text-accent-foreground"
                          : "border-border text-foreground/80 hover:border-primary/30 hover:bg-muted"
                    }`}
                  >
                    {entry?.data.title || run.paper_title || run.paper_id}
                  </button>
                );
              })}
            </div>
            {unavailable.size > 0 && (
              <p className="mt-2 text-[11.5px] italic text-muted-foreground">
                {t("matrix.unavailable_n", { n: String(unavailable.size) })}
              </p>
            )}
          </div>

          <div className="mt-7">
            {visible.length > 0 ? (
              <SnapMatrix entries={visible} />
            ) : (
              <p className="rounded-xl border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
                {t("matrix.select_prompt")}
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
