"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * Legacy compare route (`/compare`): consolidated into the unified Library
 * Matrix Compare view (`/library?view=compare`).
 */
export default function CompareRedirectPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/library?view=compare");
  }, [router]);

  return null;
}
