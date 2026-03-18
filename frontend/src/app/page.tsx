"use client";

import Link from "next/link";
import { useTranslation } from "@/lib/i18n";

export default function Home() {
  const { t } = useTranslation();

  return (
    <div className="max-w-3xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-bold mb-4">{t("home.title")}</h1>
      <p className="text-[var(--muted-foreground)] mb-8 text-lg">
        {t("home.subtitle")}
      </p>

      <div className="grid gap-4 sm:grid-cols-3 mb-12">
        <div className="border border-[var(--border)] rounded-lg p-5">
          <h3 className="font-semibold mb-2">{t("home.mode.snap.title")}</h3>
          <p className="text-sm text-[var(--muted-foreground)]">
            {t("home.mode.snap.desc")}
          </p>
        </div>
        <div className="border border-[var(--border)] rounded-lg p-5">
          <h3 className="font-semibold mb-2">{t("home.mode.lens.title")}</h3>
          <p className="text-sm text-[var(--muted-foreground)]">
            {t("home.mode.lens.desc")}
          </p>
        </div>
        <div className="border border-[var(--border)] rounded-lg p-5">
          <h3 className="font-semibold mb-2">{t("home.mode.sphere.title")}</h3>
          <p className="text-sm text-[var(--muted-foreground)]">
            {t("home.mode.sphere.desc")}
          </p>
        </div>
      </div>

      <Link
        href="/upload"
        className="inline-block bg-[var(--primary)] text-[var(--primary-foreground)] px-6 py-3 rounded-lg font-medium hover:opacity-90"
      >
        {t("home.cta")}
      </Link>
    </div>
  );
}
