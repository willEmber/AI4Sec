"use client";

import { useEffect, useState } from "react";

/**
 * Whether a media query matches — `null` until the first client render has
 * happened, so the server render and hydration agree and a caller can hold
 * off on a layout that depends on the answer.
 */
export function useMediaQuery(query: string): boolean | null {
  const [matches, setMatches] = useState<boolean | null>(null);

  useEffect(() => {
    const list = window.matchMedia(query);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, [query]);

  return matches;
}
