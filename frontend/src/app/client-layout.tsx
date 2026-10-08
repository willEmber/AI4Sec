"use client";

import { AuthMenu } from "@/components/AuthMenu";
import { LanguageProvider, LanguageToggle, useTranslation, type Locale } from "@/lib/i18n";
import { recordVisit } from "@/lib/api";
import { useEffect, type ReactNode } from "react";
import Image from "next/image";
import { usePathname } from "next/navigation";

function NavLink({ href, label }: { href: string; label: string }) {
  const pathname = usePathname();
  const active = pathname === href || (href !== "/" && pathname.startsWith(href));
  return (
    <a
      href={href}
      className={`text-sm transition-colors ${
        active
          ? "text-foreground font-medium"
          : "text-muted-foreground hover:text-foreground"
      }`}
    >
      {label}
    </a>
  );
}

function NavBar() {
  const { t } = useTranslation();

  return (
    <nav className="sticky top-0 z-40 h-14 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="flex h-full items-center gap-6 px-4 sm:px-6">
        <a href="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
          <Image
            src="/scholar.png"
            alt="Scholar"
            width={28}
            height={28}
            className="h-7 w-7 rounded-lg object-contain"
            priority
          />
          <span className="text-[15px]">{t("nav.brand")}</span>
        </a>
        <div className="mx-1 hidden h-5 w-px bg-border sm:block" />
        <NavLink href="/chat" label={t("nav.chat")} />
        <NavLink href="/projects" label={t("nav.projects")} />
        <NavLink href="/upload" label={t("nav.upload")} />
        <NavLink href="/compare" label={t("nav.compare")} />
        <NavLink href="/library" label={t("nav.library")} />
        <div className="flex-1" />
        <LanguageToggle />
        <AuthMenu />
      </div>
    </nav>
  );
}

export default function ClientLayout({
  initialLocale,
  hasStoredLocale,
  children,
}: {
  initialLocale: Locale;
  hasStoredLocale: boolean;
  children: ReactNode;
}) {
  const pathname = usePathname();
  const inChat = pathname === "/chat" || pathname.startsWith("/chat/");

  useEffect(() => {
    // Traffic collection must never affect page rendering or navigation.
    recordVisit(pathname).catch(() => {});
  }, [pathname]);

  return (
    <LanguageProvider initialLocale={initialLocale} hasStoredLocale={hasStoredLocale}>
      {/* A conversation is a full-height workspace with its own sidebar, which
          carries what the top bar carries elsewhere. */}
      {!inChat && <NavBar />}
      <main>{children}</main>
    </LanguageProvider>
  );
}
