"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useTranslation } from "@/lib/i18n";
import ProjectList from "@/components/project/ProjectList";

export default function ProjectsPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-3xl px-6 py-10" />}>
      <ProjectsContent />
    </Suspense>
  );
}

function ProjectsContent() {
  const { t } = useTranslation();
  const searchParams = useSearchParams();
  const initialCreateOpen = searchParams.get("new") === "1";

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <header className="mb-7">
        <h1 className="font-display text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">
          {t("project.list.title")}
        </h1>
        <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
          {t("project.list.subtitle")}
        </p>
      </header>
      <ProjectList initialCreateOpen={initialCreateOpen} />
    </div>
  );
}
