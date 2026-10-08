"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
  /**
   * Show the collapse button on the divider. Off when the right pane carries
   * its own close control, so there are not two buttons side by side.
   */
  dividerToggle?: boolean;
  /** Remember the divider position in localStorage under this key. */
  storageKey?: string;
}

const MIN = 25;
const MAX = 80;

export default function SplitPane({
  left,
  right,
  defaultLeftWidth = 55,
  collapsed = false,
  onToggleCollapse,
  collapseTitle,
  expandTitle,
  dividerToggle = true,
  storageKey,
}: SplitPaneProps) {
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth);
  const widthRef = useRef(leftWidth);
  widthRef.current = leftWidth;
  const containerRef = useRef<HTMLDivElement>(null);

  // Read after mount, so the server render and the first client render agree.
  useEffect(() => {
    if (!storageKey) return;
    try {
      const stored = Number(window.localStorage.getItem(storageKey));
      if (stored >= MIN && stored <= MAX) setLeftWidth(stored);
    } catch {
      // Blocked storage: keep the default.
    }
  }, [storageKey]);

  const remember = useCallback(() => {
    if (!storageKey) return;
    try {
      window.localStorage.setItem(storageKey, String(Math.round(widthRef.current)));
    } catch {
      // Not remembering a layout choice is not worth surfacing.
    }
  }, [storageKey]);

  // Pointer events with capture: one path for mouse, pen and touch, and the
  // drag keeps tracking over content that swallows events (a PDF canvas).
  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    document.body.style.userSelect = "none";
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!e.currentTarget.hasPointerCapture(e.pointerId) || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setLeftWidth(Math.max(MIN, Math.min(MAX, pct)));
  }, []);

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
      document.body.style.userSelect = "";
      remember();
    },
    [remember],
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      setLeftWidth((w) => {
        widthRef.current = Math.max(MIN, Math.min(MAX, w + (e.key === "ArrowLeft" ? -2 : 2)));
        return widthRef.current;
      });
      remember();
    },
    [remember],
  );

  return (
    <div ref={containerRef} className="flex h-full">
      <div
        style={collapsed ? undefined : { width: `${leftWidth}%` }}
        className={collapsed ? "relative min-w-0 flex-1 overflow-auto" : "relative overflow-auto"}
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
          role="separator"
          aria-orientation="vertical"
          aria-valuenow={Math.round(leftWidth)}
          aria-valuemin={MIN}
          aria-valuemax={MAX}
          tabIndex={0}
          className="group relative w-px shrink-0 cursor-col-resize touch-none bg-border"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onKeyDown={onKeyDown}
        >
          {/* Wider invisible hit area for easier grabbing */}
          <div className="absolute inset-y-0 -left-2 -right-2 z-10" />
          {/* Visible grip */}
          <div className="absolute left-1/2 top-1/2 h-9 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-border transition-colors group-hover:bg-primary group-focus-visible:bg-primary" />
          {onToggleCollapse && dividerToggle && (
            <button
              onPointerDown={(e) => e.stopPropagation()}
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
        className={collapsed ? "hidden" : "relative overflow-auto"}
      >
        {right}
      </div>
    </div>
  );
}
