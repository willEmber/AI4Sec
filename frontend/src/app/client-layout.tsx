"use client";

import { LanguageProvider, LanguageToggle, useTranslation } from "@/lib/i18n";
import type { ReactNode } from "react";

function NavBar() {
  const { t } = useTranslation();

  return (
    <nav className="border-b border-[var(--border)] px-6 py-3 flex items-center gap-6">
      <a href="/" className="font-bold text-lg">{t("nav.brand")}</a>
      <a href="/upload" className="text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
        {t("nav.upload")}
      </a>
      <div className="flex-1" />
      <LanguageToggle />
    </nav>
  );
}

export default function ClientLayout({ children }: { children: ReactNode }) {
  return (
    <LanguageProvider>
      <NavBar />
      <main>{children}</main>
    </LanguageProvider>
  );
}
