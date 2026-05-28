"use client";

import { LanguageProvider, LanguageToggle, useTranslation } from "@/lib/i18n";
import type { ReactNode } from "react";
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
        <NavLink href="/upload" label={t("nav.upload")} />
        <div className="flex-1" />
        <LanguageToggle />
      </div>
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
