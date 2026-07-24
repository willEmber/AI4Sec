"use client";

import { useCallback, useRef, useState } from "react";
import { IconPanelRightClose, IconPanelRightOpen } from "@/components/icons";

interface SplitPaneProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultLeftWidth?: number; // percentage, default 55
  /** Controlled collapse state of the right pane (requires onToggleCollapse). */
  collapsed?: boolean;
  /** When provided, a collapse/expand toggle for the right pane is rendered. */
  onToggleCollapse?: () => void;
  collapseTitle?: string;
  expandTitle?: string;
}

export default function SplitPane({
  left,
  right,
  defaultLeftWidth = 55,
  collapsed = false,
  onToggleCollapse,
  collapseTitle,
  expandTitle,
}: SplitPaneProps) {
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
      <div
        style={collapsed ? undefined : { width: `${leftWidth}%` }}
        className={collapsed ? "min-w-0 flex-1 overflow-auto" : "overflow-auto"}
      >
        {left}
      </div>
      {collapsed ? (
        onToggleCollapse && (
          /* Slim expand handle pinned to the right edge */
          <button
            onClick={onToggleCollapse}
            className="flex w-7 shrink-0 items-center justify-center border-l border-border bg-card/60 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title={expandTitle}
          >
            <IconPanelRightOpen />
          </button>
        )
      ) : (
        <div
          className="group relative w-px shrink-0 cursor-col-resize bg-border"
          onMouseDown={onMouseDown}
        >
          {/* Wider invisible hit area for easier grabbing */}
          <div className="absolute inset-y-0 -left-2 -right-2 z-10" />
          {/* Visible grip */}
          <div className="absolute left-1/2 top-1/2 h-9 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-border transition-colors group-hover:bg-primary" />
          {onToggleCollapse && (
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={onToggleCollapse}
              className="absolute left-1/2 top-3 z-20 flex h-6 w-6 -translate-x-1/2 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-sm transition-colors hover:bg-muted hover:text-foreground"
              title={collapseTitle}
            >
              <IconPanelRightClose />
            </button>
          )}
        </div>
      )}
      {/* Keep the right pane mounted while collapsed so expensive content
          (e.g. the PDF document) does not reload on expand. */}
      <div
        style={collapsed ? undefined : { width: `${100 - leftWidth}%` }}
        className={collapsed ? "hidden" : "overflow-auto"}
      >
        {right}
      </div>
    </div>
  );
}
