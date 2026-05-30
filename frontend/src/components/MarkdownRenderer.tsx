"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import { useTranslation } from "@/lib/i18n";

// Extend the default GitHub sanitize schema:
// - Allow className on div/span (needed for math wrappers from remark-math)
// - Allow dataPage on span (keeps backwards compatibility with legacy citation badges)
// - Allow className on code (needed for syntax-highlighted code blocks)
const sanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    div: [...(defaultSchema.attributes?.div || []), "className"],
    span: [...(defaultSchema.attributes?.span || []), "className", "dataPage", "data-page"],
    code: [...(defaultSchema.attributes?.code || []), "className"],
    // Allow embedded figure images (e.g. the architecture diagram in Logic Lens).
    img: [...(defaultSchema.attributes?.img || []), "src", "alt", "title", "loading"],
  },
  // Only same-origin /api image URLs are emitted by the backend; permit relative paths.
  protocols: {
    ...defaultSchema.protocols,
    src: [...(defaultSchema.protocols?.src || []), "http", "https"],
  },
};

const LEGACY_CITATION_SPAN_RE =
  /<span\s+class=["']cite-badge["']\s+data-page=["'](\d+)["']>\[p\.\1\]<\/span>/gi;
const PLAIN_CITATION_RE = /\[p\.(\d+)\]/gi;
const CITATION_HREF_RE = /^#cite-page-(\d+)$/;

// Library (knowledge-base) citations: corpus answers carry document-level
// `[L1]`, `[L2]`… markers (no page index) that link to a source document.
const PLAIN_LIB_CITATION_RE = /\[L(\d+)\]/g;
const LIB_CITATION_HREF_RE = /^#cite-lib-(\d+)$/;

// remark-math only treats `$$...$$` as a display (block) equation when it is
// separated from the surrounding text by blank lines. LLM output routinely puts
// each equation on its own line but with no blank line around it, so the whole
// run becomes one paragraph and every `$$...$$` is parsed as *inline* math,
// rendered squished against the prose. Re-wrap each block (outside fenced code)
// with blank lines so it renders as a centered display equation instead.
const CODE_FENCE_RE = /```[\s\S]*?```/g;
const DISPLAY_MATH_RE = /\$\$([\s\S]+?)\$\$/g;

export function normalizeDisplayMath(content: string): string {
  const codeBlocks: string[] = [];

  // Park fenced code blocks so any `$$` inside them is left untouched.
  const guarded = content.replace(CODE_FENCE_RE, (block) => {
    codeBlocks.push(block);
    return `@@mathcode${codeBlocks.length - 1}@@`;
  });

  const wrapped = guarded
    .replace(DISPLAY_MATH_RE, (_match, inner: string) => `\n\n$$\n${inner.trim()}\n$$\n\n`)
    .replace(/\n{3,}/g, "\n\n");

  return wrapped.replace(/@@mathcode(\d+)@@/g, (_match, i: string) => codeBlocks[Number(i)]);
}

export function prepareCitationMarkdown(content: string): string {
  return content
    .replace(LEGACY_CITATION_SPAN_RE, "[p.$1]")
    .replace(PLAIN_CITATION_RE, (_match, page: string) => `[[p.${page}]](#cite-page-${page})`)
    .replace(PLAIN_LIB_CITATION_RE, (_match, idx: string) => `[[L${idx}]](#cite-lib-${idx})`);
}

interface MarkdownRendererProps {
  content: string;
  onCitationClick?: (page: number) => void;
  onLibraryCitationClick?: (idx: number) => void;
}

export default function MarkdownRenderer({
  content,
  onCitationClick,
  onLibraryCitationClick,
}: MarkdownRendererProps) {
  const { t } = useTranslation();

  // Normalize display equations to block form first, then keep citations as
  // Markdown links so raw HTML never leaks into rendered answers.
  const processed = prepareCitationMarkdown(normalizeDisplayMath(content));

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema], rehypeKatex]}
        components={{
          a: ({ node, children, ...props }) => {
            const href = typeof props.href === "string" ? props.href : "";
            const citationMatch = CITATION_HREF_RE.exec(href);

            if (citationMatch) {
              const page = parseInt(citationMatch[1], 10);
              return (
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    onCitationClick?.(page);
                  }}
                  className="mx-0.5 inline-flex cursor-pointer items-center rounded-md border border-primary/30 bg-accent px-1.5 py-0.5 align-baseline font-mono text-xs font-medium text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
                  title={t("pdf.jump_to_page", { page: String(page) })}
                >
                  p.{page}
                </button>
              );
            }

            const libMatch = LIB_CITATION_HREF_RE.exec(href);
            if (libMatch) {
              const idx = parseInt(libMatch[1], 10);
              return (
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    onLibraryCitationClick?.(idx);
                  }}
                  className="mx-0.5 inline-flex cursor-pointer items-center rounded-md border border-primary/30 bg-accent px-1.5 py-0.5 align-baseline font-mono text-xs font-medium text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
                  title={t("library.open_doc")}
                >
                  L{idx}
                </button>
              );
            }

            return <a {...props}>{children}</a>;
          },
          span: ({ node, children, ...props }) => {
            const className = (props as Record<string, unknown>).className as string | undefined;
            const dataPage = (
              (props as Record<string, unknown>)["data-page"] ||
              (props as Record<string, unknown>).dataPage
            ) as string | undefined;

            if (className === "cite-badge" && dataPage) {
              const page = parseInt(dataPage, 10);
              return (
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    onCitationClick?.(page);
                  }}
                  className="mx-0.5 inline-flex cursor-pointer items-center rounded-md border border-primary/30 bg-accent px-1.5 py-0.5 align-baseline font-mono text-xs font-medium text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
                  title={t("pdf.jump_to_page", { page: String(page) })}
                >
                  p.{page}
                </button>
              );
            }

            return <span {...props}>{children}</span>;
          },
          img: ({ node, ...props }) => {
            const src = typeof props.src === "string" ? props.src : "";
            const alt = typeof props.alt === "string" ? props.alt : "";
            if (!src) return null;
            // Rendered inside a <p>, so use inline-level wrappers (no <figure>/<div>).
            return (
              <span className="my-4 flex flex-col items-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={src}
                  alt={alt}
                  loading="lazy"
                  className="max-h-[28rem] max-w-full rounded-lg border border-border bg-white object-contain shadow-sm"
                />
                {alt ? (
                  <span className="mt-1.5 px-4 text-center text-xs text-muted-foreground">
                    {alt}
                  </span>
                ) : null}
              </span>
            );
          },
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}
