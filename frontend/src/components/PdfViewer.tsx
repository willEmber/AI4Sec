"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
}

const TOOLBAR_BTN =
  "inline-flex h-8 w-8 items-center justify-center rounded-lg border border-border text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:bg-transparent";

export default function PdfViewer({ url, targetPage }: PdfViewerProps) {
  const { t } = useTranslation();
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [scale, setScale] = useState(1.0);
  const containerRef = useRef<HTMLDivElement>(null);

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  }, []);

  // Jump to target page when it changes
  useEffect(() => {
    if (targetPage && targetPage >= 1 && targetPage <= numPages) {
      setCurrentPage(targetPage);
      // Scroll to page
      const pageEl = document.getElementById(`pdf-page-${targetPage}`);
      pageEl?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [targetPage, numPages]);

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-2 border-b border-border bg-card/60 px-3 py-2 text-sm">
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
      </div>

      {/* PDF content */}
      <div ref={containerRef} className="flex-1 overflow-auto bg-muted p-5">
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess} loading={
          <div className="flex h-48 items-center justify-center text-muted-foreground">
            {t("pdf.loading")}
          </div>
        }>
          <div id={`pdf-page-${currentPage}`}>
            <Page
              pageNumber={currentPage}
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
