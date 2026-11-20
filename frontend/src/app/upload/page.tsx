"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * Legacy classic-report route (`/upload`): redirected into the unified `/chat`
 * composer, where attaching a PDF and picking Snap / Lens / Sphere produces
 * the same report inside a follow-up-ready conversation.
 */
export default function UploadRedirectPage() {
  const router = useRouter();

  useEffect(() => {
    const mode = new URLSearchParams(window.location.search).get("mode");
    router.replace(mode ? `/chat?mode=${encodeURIComponent(mode)}` : "/chat");
  }, [router]);

  return null;
}
