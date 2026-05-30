"use client";

import { useCallback, useEffect, useState } from "react";
import {
  listLibraryDocuments,
  getLibraryMarkdown,
  searchLibrary,
  askLibrary,
} from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import SplitPane from "@/components/SplitPane";
import type {
  LibraryDocument,
  LibrarySearchRecord,
  LibraryAskResponse,
  SearchMethod,
} from "@/lib/types";

const METHODS: SearchMethod[] = ["full_text_search", "semantic_search", "hybrid_search"];

function fmtScore(score: number | null | undefined): string | null {
  return typeof score === "number" && score > 0 ? score.toFixed(3) : null;
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export default function LibraryPage() {
  const { t, locale } = useTranslation();

  const [tab, setTab] = useState<"search" | "ask">("search");
  const [method, setMethod] = useState<SearchMethod>("full_text_search");
  const [error, setError] = useState<string>("");

  // Search
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<LibrarySearchRecord[]>([]);
  const [searched, setSearched] = useState(false);
  const [searching, setSearching] = useState(false);

  // Ask
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<LibraryAskResponse | null>(null);
  const [asking, setAsking] = useState(false);

  // Browse documents
  const [docs, setDocs] = useState<LibraryDocument[]>([]);
  const [docsPage, setDocsPage] = useState(0);
  const [docsHasMore, setDocsHasMore] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);

  // Right-pane document preview
  const [selectedName, setSelectedName] = useState("");
  const [docContent, setDocContent] = useState("");
  const [docLoading, setDocLoading] = useState(false);

  const loadDocs = useCallback(
    async (page: number) => {
      setDocsLoading(true);
      try {
        const res = await listLibraryDocuments({ page, limit: 20 });
        setDocs((prev) => (page === 1 ? res.data : [...prev, ...res.data]));
        setDocsHasMore(Boolean(res.has_more));
        setDocsPage(page);
        setError("");
      } catch (err) {
        setError(errMessage(err));
      } finally {
        setDocsLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    loadDocs(1);
  }, [loadDocs]);

  const openDoc = useCallback(async (documentId: string, name: string) => {
    if (!documentId) return;
    setDocLoading(true);
    setSelectedName(name);
    try {
      const res = await getLibraryMarkdown(documentId);
      setDocContent(res.content || "");
      if (!name && res.document_name) setSelectedName(res.document_name);
      setError("");
    } catch (err) {
      setDocContent("");
      setError(errMessage(err));
    } finally {
      setDocLoading(false);
    }
  }, []);

  // Deep link: /library?doc=<document_id> (e.g. from a Research Sphere report)
  // opens that document in the preview pane. Read from the URL on the client to
  // avoid the Suspense boundary that useSearchParams would require at build.
  useEffect(() => {
    const doc = new URLSearchParams(window.location.search).get("doc");
    if (doc) openDoc(doc, "");
  }, [openDoc]);

  const runSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) {
      setError(t("library.empty_query"));
      return;
    }
    setSearching(true);
    setError("");
    try {
      const res = await searchLibrary({ query: q, search_method: method, top_k: 20 });
      setResults(res.records || []);
      setSearched(true);
    } catch (err) {
      setError(errMessage(err));
    } finally {
      setSearching(false);
    }
  }, [query, method, t]);

  const runAsk = useCallback(async () => {
    const q = question.trim();
    if (!q) {
      setError(t("library.empty_question"));
      return;
    }
    setAsking(true);
    setError("");
    try {
      const res = await askLibrary({
        question: q,
        search_method: method,
        language: locale,
        top_k: 10,
      });
      setAnswer(res);
    } catch (err) {
      setError(errMessage(err));
    } finally {
      setAsking(false);
    }
  }, [question, method, locale, t]);

  const onLibraryCitationClick = useCallback(
    (idx: number) => {
      const src = answer?.sources.find((s) => s.idx === idx);
      if (src) openDoc(src.document_id, src.document_name);
    },
    [answer, openDoc],
  );

  const slowHint = method !== "full_text_search";

  const methodSelect = (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">{t("library.method_label")}</span>
      <select
        value={method}
        onChange={(e) => setMethod(e.target.value as SearchMethod)}
        className="rounded-lg border border-border bg-card px-2 py-1.5 text-sm"
      >
        {METHODS.map((m) => (
          <option key={m} value={m}>
            {t(`library.method.${m}`)}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col">
      {/* Header */}
      <div className="shrink-0 border-b border-border bg-card/60 px-5 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="min-w-0">
            <h1 className="font-display text-lg font-semibold tracking-tight">
              {t("library.title")}
            </h1>
            <p className="truncate text-xs text-muted-foreground">{t("library.subtitle")}</p>
          </div>
          <div className="flex shrink-0 items-center gap-1 rounded-xl border border-border bg-background p-1">
            {(["search", "ask"] as const).map((key) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  tab === key
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {t(`library.tab.${key}`)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        <SplitPane
          defaultLeftWidth={48}
          left={
            <div className="flex h-full flex-col">
              {/* Controls */}
              <div className="shrink-0 space-y-2 border-b border-border px-5 py-3">
                {tab === "search" ? (
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      runSearch();
                    }}
                    className="flex items-center gap-2"
                  >
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder={t("library.search_placeholder")}
                      className="min-w-0 flex-1 rounded-lg border border-border bg-card px-3 py-2 text-sm outline-none focus:border-primary/50"
                    />
                    <button
                      type="submit"
                      disabled={searching}
                      className="shrink-0 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-50"
                    >
                      {searching ? t("library.searching") : t("library.run_search")}
                    </button>
                  </form>
                ) : (
                  <div className="space-y-2">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyDown={(e) => {
                        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                          e.preventDefault();
                          runAsk();
                        }
                      }}
                      placeholder={t("library.ask_placeholder")}
                      rows={3}
                      className="w-full resize-y rounded-lg border border-border bg-card px-3 py-2 text-sm outline-none focus:border-primary/50"
                    />
                    <div className="flex justify-end">
                      <button
                        onClick={runAsk}
                        disabled={asking}
                        className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-50"
                      >
                        {asking ? t("library.thinking") : t("library.run_ask")}
                      </button>
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between gap-3">
                  {methodSelect}
                  {slowHint && (
                    <span className="text-right text-xs text-muted-foreground">
                      {t("library.method_slow_hint")}
                    </span>
                  )}
                </div>

                {error && <p className="text-xs text-destructive">{error}</p>}
              </div>

              {/* Results / answer / browse */}
              <div className="flex-1 overflow-auto px-5 py-3">
                {tab === "search" ? (
                  searching ? (
                    <CenterSpinner label={t("library.searching")} />
                  ) : searched ? (
                    results.length === 0 ? (
                      <p className="py-10 text-center text-sm text-muted-foreground">
                        {t("library.no_results")}
                      </p>
                    ) : (
                      <div className="space-y-2">
                        <p className="px-1 text-xs text-muted-foreground">
                          {t("library.results_count", { count: results.length })}
                        </p>
                        {results.map((r, i) => (
                          <ResultCard
                            key={`${r.segment_id}-${i}`}
                            name={r.document_name}
                            score={fmtScore(r.score)}
                            snippet={r.content}
                            onClick={() => openDoc(r.document_id, r.document_name)}
                          />
                        ))}
                      </div>
                    )
                  ) : (
                    <div className="space-y-2">
                      <p className="px-1 text-xs font-medium text-muted-foreground">
                        {t("library.documents")}
                      </p>
                      {docs.map((d) => (
                        <ResultCard
                          key={d.id}
                          name={d.name}
                          score={d.word_count ? `${d.word_count.toLocaleString()} w` : null}
                          onClick={() => openDoc(d.id, d.name)}
                        />
                      ))}
                      {docsHasMore && (
                        <button
                          onClick={() => loadDocs(docsPage + 1)}
                          disabled={docsLoading}
                          className="w-full rounded-lg border border-border py-2 text-sm text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50"
                        >
                          {docsLoading ? t("library.searching") : t("library.load_more")}
                        </button>
                      )}
                    </div>
                  )
                ) : asking ? (
                  <CenterSpinner label={t("library.thinking")} />
                ) : answer ? (
                  <div className="space-y-5">
                    <MarkdownRenderer
                      content={answer.markdown}
                      onLibraryCitationClick={onLibraryCitationClick}
                    />
                    {answer.sources.length > 0 && (
                      <div className="border-t border-border pt-3">
                        <p className="mb-2 text-xs font-medium text-muted-foreground">
                          {t("library.sources")}
                        </p>
                        <div className="space-y-1.5">
                          {answer.sources.map((s) => (
                            <button
                              key={s.idx}
                              onClick={() => openDoc(s.document_id, s.document_name)}
                              className="flex w-full items-start gap-2 rounded-lg border border-border bg-card px-3 py-2 text-left text-sm transition-colors hover:border-primary/40"
                            >
                              <span className="mt-0.5 shrink-0 font-mono text-xs text-primary">
                                L{s.idx}
                              </span>
                              <span className="min-w-0 flex-1 truncate">{s.document_name}</span>
                              {fmtScore(s.score) && (
                                <span className="shrink-0 text-xs text-muted-foreground">
                                  {fmtScore(s.score)}
                                </span>
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="py-10 text-center text-sm text-muted-foreground">
                    {t("library.select_hint")}
                  </p>
                )}
              </div>
            </div>
          }
          right={
            <div className="h-full">
              {docLoading ? (
                <CenterSpinner label={t("library.doc_loading")} />
              ) : docContent ? (
                <div className="px-6 py-6">
                  {selectedName && (
                    <p className="mb-4 break-words border-b border-border pb-3 text-sm font-medium text-muted-foreground">
                      {selectedName}
                    </p>
                  )}
                  <div className="mx-auto max-w-3xl">
                    <MarkdownRenderer content={docContent} />
                  </div>
                </div>
              ) : (
                <p className="flex h-full items-center justify-center px-6 text-center text-sm text-muted-foreground">
                  {t("library.select_hint")}
                </p>
              )}
            </div>
          }
        />
      </div>
    </div>
  );
}

function CenterSpinner({ label }: { label: string }) {
  return (
    <div className="flex h-full items-start justify-center py-12">
      <div className="text-center">
        <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-[3px] border-primary border-t-transparent" />
        <p className="text-sm text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}

function ResultCard({
  name,
  score,
  snippet,
  onClick,
}: {
  name: string;
  score: string | null;
  snippet?: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="block w-full rounded-xl border border-border bg-card px-3.5 py-2.5 text-left transition-colors hover:border-primary/40"
    >
      <div className="flex items-start gap-2">
        <span className="min-w-0 flex-1 break-words text-sm font-medium">{name}</span>
        {score && (
          <span className="shrink-0 rounded-md bg-accent px-1.5 py-0.5 font-mono text-xs text-primary">
            {score}
          </span>
        )}
      </div>
      {snippet && (
        <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
          {snippet.trim().slice(0, 240)}
        </p>
      )}
    </button>
  );
}
