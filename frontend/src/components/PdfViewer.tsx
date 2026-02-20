"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";

// Load worker from same origin (copied to public/ by postinstall script)
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

interface PdfViewerProps {
  url: string;
  targetPage?: number;
}

export default function PdfViewer({ url, targetPage }: PdfViewerProps) {
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
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-3 py-2 border-b border-[var(--border)] text-sm shrink-0">
        <button
          onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
          disabled={currentPage <= 1}
          className="px-2 py-1 rounded border border-[var(--border)] disabled:opacity-30"
        >
          Prev
        </button>
        <span>
          {currentPage} / {numPages}
        </span>
        <button
          onClick={() => setCurrentPage(Math.min(numPages, currentPage + 1))}
          disabled={currentPage >= numPages}
          className="px-2 py-1 rounded border border-[var(--border)] disabled:opacity-30"
        >
          Next
        </button>
        <div className="flex-1" />
        <button
          onClick={() => setScale(Math.max(0.5, scale - 0.1))}
          className="px-2 py-1 rounded border border-[var(--border)]"
        >
          -
        </button>
        <span>{Math.round(scale * 100)}%</span>
        <button
          onClick={() => setScale(Math.min(3.0, scale + 0.1))}
          className="px-2 py-1 rounded border border-[var(--border)]"
        >
          +
        </button>
      </div>

      {/* PDF content */}
      <div ref={containerRef} className="flex-1 overflow-auto bg-[var(--muted)] p-4">
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess} loading={
          <div className="flex items-center justify-center h-48 text-[var(--muted-foreground)]">
            Loading PDF...
          </div>
        }>
          <div id={`pdf-page-${currentPage}`}>
            <Page
              pageNumber={currentPage}
              scale={scale}
              className="mx-auto shadow-lg"
              renderTextLayer={true}
              renderAnnotationLayer={true}
            />
          </div>
        </Document>
      </div>
    </div>
  );
}
