"use client";

import { useCallback, useRef, useState } from "react";

interface SplitPaneProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultLeftWidth?: number; // percentage, default 55
}

export default function SplitPane({ left, right, defaultLeftWidth = 55 }: SplitPaneProps) {
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth);
  const dragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const onMouseDown = useCallback(() => {
    dragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setLeftWidth(Math.max(20, Math.min(80, pct)));
  }, []);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  return (
    <div
      ref={containerRef}
      className="flex h-full"
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      <div style={{ width: `${leftWidth}%` }} className="overflow-auto">
        {left}
      </div>
      <div
        className="group relative w-px shrink-0 cursor-col-resize bg-border"
        onMouseDown={onMouseDown}
      >
        {/* Wider invisible hit area for easier grabbing */}
        <div className="absolute inset-y-0 -left-2 -right-2 z-10" />
        {/* Visible grip */}
        <div className="absolute left-1/2 top-1/2 h-9 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-border transition-colors group-hover:bg-primary" />
      </div>
      <div style={{ width: `${100 - leftWidth}%` }} className="overflow-auto">
        {right}
      </div>
    </div>
  );
}
