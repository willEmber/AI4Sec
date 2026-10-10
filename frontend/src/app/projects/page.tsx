"use client";

import { useTranslation } from "@/lib/i18n";
import { IconDocument } from "@/components/icons";
import ProjectList from "@/components/project/ProjectList";

/** Research projects: groups of conversations about one question. */
export default function ProjectsPage() {
  const { t } = useTranslation();

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8">
        <p className="mb-2 inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-accent-foreground">
          <IconDocument className="h-4 w-4" />
          {t("project.list.eyebrow")}
        </p>
        <h1 className="font-display text-3xl font-bold text-foreground">{t("project.list.title")}</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{t("project.list.subtitle")}</p>
      </div>
      <ProjectList />
    </main>
  );
}
