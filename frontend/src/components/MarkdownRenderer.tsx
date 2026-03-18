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
// - Allow dataPage on span (needed for citation badges)
// - Allow className on code (needed for syntax-highlighted code blocks)
const sanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    div: [...(defaultSchema.attributes?.div || []), "className"],
    span: [...(defaultSchema.attributes?.span || []), "className", "dataPage"],
    code: [...(defaultSchema.attributes?.code || []), "className"],
  },
};

interface MarkdownRendererProps {
  content: string;
  onCitationClick?: (page: number) => void;
}

export default function MarkdownRenderer({ content, onCitationClick }: MarkdownRendererProps) {
  const { t } = useTranslation();

  // Pre-process content to convert [p.X] citations into clickable span badges
  const processed = content.replace(
    /\[p\.(\d+)\]/g,
    '<span class="cite-badge" data-page="$1">[p.$1]</span>'
  );

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema], rehypeKatex]}
        components={{
          span: ({ node, children, ...props }) => {
            const className = (props as Record<string, unknown>).className as string | undefined;
            const dataPage = (props as Record<string, unknown>)["data-page"] as string | undefined;

            if (className === "cite-badge" && dataPage) {
              const page = parseInt(dataPage, 10);
              return (
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    onCitationClick?.(page);
                  }}
                  className="inline-flex items-center px-1.5 py-0.5 text-xs font-mono bg-blue-50 text-blue-700 rounded border border-blue-200 hover:bg-blue-600 hover:text-white transition-colors cursor-pointer mx-0.5 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800"
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
