"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { createSession, listSessions } from "@/lib/agent";
import type { AgentSession } from "@/lib/agent";
import { listRecentRuns } from "@/lib/api";
import type { RecentRunResponse } from "@/lib/types";
import { useTranslation } from "@/lib/i18n";
import { IconArrowRight, IconSparkles, IconUpload } from "@/components/icons";

/**
 * Entry to the reading agent.
 *
 * `?paper=<id>` opens a session on that paper straight away — the path taken
 * from a finished run. Without it, the reader picks from the papers already in
 * the workspace, or resumes an earlier conversation.
 *
 * Reading the query string opts the page out of prerendering, so it needs a
 * Suspense boundary of its own.
 */
export default function ChatEntryPage() {
  return (
    <Suspense fallback={<main className="mx-auto max-w-3xl px-6 py-12" />}>
      <ChatEntry />
    </Suspense>
  );
}

function ChatEntry() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const presetPaperId = searchParams.get("paper") || "";
  const { t, locale } = useTranslation();

  const [sessions, setSessions] = useState<AgentSession[] | null>(null);
  const [runs, setRuns] = useState<RecentRunResponse[] | null>(null);
  const [creating, setCreating] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    listSessions()
      .then((data) => setSessions(data.sessions))
      .catch(() => setSessions([]));
    listRecentRuns(40, false)
      .then(setRuns)
      .catch(() => setRuns([]));
  }, []);

  // One row per paper: the same PDF is usually analysed several times, and the
  // agent reads the paper, not the run.
  const papers = useMemo(() => {
    const seen = new Map<string, RecentRunResponse>();
    for (const run of runs ?? []) {
      if (run.paper_id && !seen.has(run.paper_id)) seen.set(run.paper_id, run);
    }
    return [...seen.values()];
  }, [runs]);

  async function open(paperId: string, title: string) {
    if (creating) return;
    setCreating(paperId);
    setError("");
    try {
      const session = await createSession({
        title,
        language: locale === "zh" ? "zh" : "en",
        paper_ids: paperId ? [paperId] : [],
      });
      router.push(`/chat/${session.session_id}`);
    } catch (err) {
      setError(String(err));
      setCreating("");
    }
  }

  // Arriving with ?paper=… means the choice is already made.
  useEffect(() => {
    if (!presetPaperId || creating) return;
    void open(presetPaperId, "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetPaperId]);

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8">
        <p className="mb-2 inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-accent-foreground">
          <IconSparkles className="h-4 w-4" />
          {t("chat.entry.eyebrow")}
        </p>
        <h1 className="font-display text-3xl font-bold text-foreground">
          {t("chat.entry.title")}
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          {t("chat.entry.subtitle")}
        </p>
      </div>

      {error && (
        <p className="mb-6 rounded-lg border border-border bg-muted px-3 py-2 text-xs text-destructive">
          {error}
        </p>
      )}

      <section className="mb-10">
        <h2 className="mb-3 text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {t("chat.entry.pick_paper")}
        </h2>

        {papers.length === 0 && (
          <div className="rounded-xl border border-dashed border-border px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">{t("chat.entry.no_papers")}</p>
            <Link
              href="/upload"
              className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
            >
              <IconUpload className="h-4 w-4" />
              {t("home.cta")}
            </Link>
          </div>
        )}

        <div className="space-y-2">
          {papers.map((paper) => (
            <button
              key={paper.paper_id}
              type="button"
              disabled={Boolean(creating)}
              onClick={() => void open(paper.paper_id, paper.paper_title)}
              className="lift flex w-full items-center justify-between gap-4 rounded-xl border border-border bg-card px-4 py-3 text-left disabled:opacity-60"
            >
              <span className="min-w-0">
                <span className="line-clamp-2 block text-sm font-medium text-foreground">
                  {paper.paper_title || paper.paper_id.slice(0, 16)}
                </span>
              </span>
              <span className="shrink-0 text-xs text-muted-foreground">
                {creating === paper.paper_id ? (
                  t("chat.entry.opening")
                ) : (
                  <IconArrowRight className="h-4 w-4" />
                )}
              </span>
            </button>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
          {t("chat.sessions.heading")}
        </h2>
        {sessions !== null && sessions.length === 0 && (
          <p className="text-xs text-muted-foreground">{t("chat.sessions.empty")}</p>
        )}
        <div className="space-y-1">
          {(sessions ?? []).map((session) => (
            <Link
              key={session.session_id}
              href={`/chat/${session.session_id}`}
              className="block truncate rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              {session.title || t("chat.sessions.untitled")}
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
