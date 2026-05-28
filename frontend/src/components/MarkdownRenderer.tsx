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
  },
};

const LEGACY_CITATION_SPAN_RE =
  /<span\s+class=["']cite-badge["']\s+data-page=["'](\d+)["']>\[p\.\1\]<\/span>/gi;
const PLAIN_CITATION_RE = /\[p\.(\d+)\]/gi;
const CITATION_HREF_RE = /^#cite-page-(\d+)$/;

export function prepareCitationMarkdown(content: string): string {
  return content
    .replace(LEGACY_CITATION_SPAN_RE, "[p.$1]")
    .replace(PLAIN_CITATION_RE, (_match, page: string) => `[[p.${page}]](#cite-page-${page})`);
}

interface MarkdownRendererProps {
  content: string;
  onCitationClick?: (page: number) => void;
}

export default function MarkdownRenderer({ content, onCitationClick }: MarkdownRendererProps) {
  const { t } = useTranslation();

  // Keep citations as Markdown links so raw HTML never leaks into rendered answers.
  const processed = prepareCitationMarkdown(content);

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
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}
