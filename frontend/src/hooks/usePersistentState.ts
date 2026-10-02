"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * State remembered in localStorage — for layout preferences such as which
 * panels are collapsed, never for anything that must survive reliably.
 *
 * The stored value is read after mount, not during the first render, so the
 * server render and the first client render agree. `initial` may be a function
 * for a default that depends on the window (e.g. collapsed on narrow screens);
 * it runs only when nothing is stored yet.
 */
export function usePersistentState<T>(
  key: string,
  initial: T | (() => T),
  fallback: T,
): [T, (value: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(fallback);

  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(key);
    } catch {
      // Blocked storage (private mode, sandboxed frame): use the default.
    }
    if (stored !== null) {
      try {
        setValue(JSON.parse(stored) as T);
        return;
      } catch {
        // A malformed entry is treated as missing.
      }
    }
    setValue(typeof initial === "function" ? (initial as () => T)() : initial);
    // `initial` is a default, read once per key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const update = useCallback(
    (next: T | ((prev: T) => T)) => {
      setValue((prev) => {
        const resolved =
          typeof next === "function" ? (next as (prev: T) => T)(prev) : next;
        try {
          window.localStorage.setItem(key, JSON.stringify(resolved));
        } catch {
          // Not remembering a layout choice is not worth surfacing.
        }
        return resolved;
      });
    },
    [key],
  );

  return [value, update];
}
