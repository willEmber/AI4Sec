"use client";

import { useEffect, type ReactNode } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AuthMenu } from "@/components/AuthMenu";
import ChatShell from "@/components/agent/ChatShell";
import { recordVisit } from "@/lib/api";
import {
  LanguageProvider,
  LanguageToggle,
  ThemeToggle,
  useTranslation,
  type Locale,
  type ThemeChoice,
} from "@/lib/i18n";

function LandingHeader() {
  const { t } = useTranslation();

  return (
    <header className="sticky top-0 z-40 h-14 border-b border-border/70 bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex h-full max-w-5xl items-center gap-3 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
          <Image
            src="/logo.png"
            alt="Scholar"
            width={28}
            height={28}
            className="h-7 w-7 object-contain"
            priority
          />
          <span className="text-[15px]">{t("nav.brand")}</span>
        </Link>
        <div className="flex-1" />
        <ThemeToggle />
        <LanguageToggle />
        <AuthMenu />
      </div>
    </header>
  );
}

export default function ClientLayout({
  initialLocale,
  hasStoredLocale,
  initialTheme,
  children,
}: {
  initialLocale: Locale;
  hasStoredLocale: boolean;
  initialTheme?: ThemeChoice;
  children: ReactNode;
}) {
  const pathname = usePathname();
  const isLanding = pathname === "/";

  useEffect(() => {
    // Delay slightly so instant redirect routes (`/`, `/upload`, `/compare`)
    // do not fire and immediately abort an in-flight visit request.
    const timer = setTimeout(() => {
      recordVisit(pathname).catch(() => {});
    }, 250);
    return () => clearTimeout(timer);
  }, [pathname]);

  return (
    <LanguageProvider
      initialLocale={initialLocale}
      hasStoredLocale={hasStoredLocale}
      initialTheme={initialTheme}
    >
      {isLanding ? (
        <>
          <LandingHeader />
          <main>{children}</main>
        </>
      ) : (
        <ChatShell>{children}</ChatShell>
      )}
    </LanguageProvider>
  );
}
