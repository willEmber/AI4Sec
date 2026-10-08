"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { useTranslation } from "@/lib/i18n";
import {
  IconChevronLeft,
  IconChevronRight,
  IconMinus,
  IconPlus,
} from "@/components/icons";

// Load worker from same origin (copied to public/ by postinstall script)
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

interface PdfViewerProps {
  url: string;
  targetPage?: number;
  /**
   * Changes on every jump request, even one to the page already shown.
   *
   * Without it, "go to page 3 of this other paper" is invisible when page 3 of
   * the previous paper is already open: the page number did not change, so
   * nothing would re-run and the reader would be left looking at the wrong
   * document (acceptance case A07).
   */
  jumpToken?: number;
  /** Rendered at the start of the toolbar, e.g. a paper switcher. */
  toolbarStart?: ReactNode;
  /** Rendered at the end of the toolbar, e.g. a collapse button. */
  toolbarEnd?: ReactNode;
}

const TOOLBAR_BTN =
  "inline-flex h-8 w-8 items-center justify-center rounded-lg border border-border text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:bg-transparent";

export default function PdfViewer({
  url,
  targetPage,
  jumpToken,
  toolbarStart,
  toolbarEnd,
}: PdfViewerProps) {
  const { t } = useTranslation();
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [scale, setScale] = useState(1.0);
  const containerRef = useRef<HTMLDivElement>(null);
  // 100% means "as wide as the pane", so a page fits however the pane was
  // resized; zoom scales from there.
  const [fitWidth, setFitWidth] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      // contentRect excludes the padding; floored so sub-pixel jitter does not re-render the page.
      setFitWidth(Math.max(200, Math.floor(entry.contentRect.width)));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  }, []);

  // A different document is a different document: forget the old page count so
  // the jump below waits for the new one to load before trying to scroll.
  useEffect(() => {
    setNumPages(0);
    setCurrentPage(1);
  }, [url]);

  // Jump when the page changes, and when the same page is requested again for
  // a different paper — hence jumpToken.
  useEffect(() => {
    if (targetPage && targetPage >= 1 && targetPage <= numPages) {
      setCurrentPage(targetPage);
      const pageEl = document.getElementById(`pdf-page-${targetPage}`);
      pageEl?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [targetPage, numPages, jumpToken]);

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-2 border-b border-border bg-card/60 px-3 py-2 text-sm">
        {toolbarStart}
        <button
          onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
          disabled={currentPage <= 1}
          className={TOOLBAR_BTN}
          title={t("pdf.prev")}
        >
          <IconChevronLeft />
        </button>
        <span className="tabular-nums text-muted-foreground">
          <span className="font-medium text-foreground">{currentPage}</span> / {numPages || "–"}
        </span>
        <button
          onClick={() => setCurrentPage(Math.min(numPages, currentPage + 1))}
          disabled={currentPage >= numPages}
          className={TOOLBAR_BTN}
          title={t("pdf.next")}
        >
          <IconChevronRight />
        </button>
        <div className="flex-1" />
        <button
          onClick={() => setScale(Math.max(0.5, scale - 0.1))}
          className={TOOLBAR_BTN}
        >
          <IconMinus />
        </button>
        <span className="w-11 text-center tabular-nums text-muted-foreground">
          {Math.round(scale * 100)}%
        </span>
        <button
          onClick={() => setScale(Math.min(3.0, scale + 0.1))}
          className={TOOLBAR_BTN}
        >
          <IconPlus />
        </button>
        {toolbarEnd}
      </div>

      {/* PDF content */}
      <div ref={containerRef} className="relative flex-1 overflow-auto bg-muted p-5">
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess} loading={
          <div className="flex h-48 items-center justify-center text-muted-foreground">
            {t("pdf.loading")}
          </div>
        }>
          <div id={`pdf-page-${currentPage}`}>
            <Page
              pageNumber={currentPage}
              width={fitWidth || undefined}
              scale={scale}
              className="mx-auto overflow-hidden rounded-lg shadow-[0_4px_24px_-8px_rgba(20,20,19,0.25)]"
              renderTextLayer={true}
              renderAnnotationLayer={true}
            />
          </div>
        </Document>
      </div>
    </div>
  );
}
