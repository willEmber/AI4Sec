"use client";

import dynamic from "next/dynamic";

// pdf.js is the largest dependency in the app and only matters once a PDF is
// on screen, so it is kept out of the pages' first load and fetched when the
// viewer mounts.
const LazyPdfViewer = dynamic(() => import("@/components/PdfViewer"), {
  ssr: false,
  loading: () => <div className="h-full w-full animate-pulse bg-muted/40" />,
});

export default LazyPdfViewer;
