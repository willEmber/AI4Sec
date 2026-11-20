import "./globals.css";
import type { Metadata } from "next";
import { cookies } from "next/headers";
import ClientLayout from "./client-layout";
import {
  DEFAULT_LOCALE,
  DEFAULT_THEME,
  LOCALE_COOKIE,
  THEME_COOKIE,
  parseLocale,
  parseTheme,
} from "@/lib/locale";

export const metadata: Metadata = {
  title: "Scholar — AI Paper Reading",
  description: "Upload a paper, choose a reading mode, and get structured AI analysis with citations linking back to the source PDF.",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Read the language and theme on the server so the first HTML matches;
  // resolving them after mount flashed defaults on every load.
  const jar = await cookies();
  const stored = parseLocale(jar.get(LOCALE_COOKIE)?.value);
  const locale = stored ?? DEFAULT_LOCALE;
  const theme = parseTheme(jar.get(THEME_COOKIE)?.value) ?? DEFAULT_THEME;

  return (
    <html lang={locale === "zh" ? "zh-CN" : "en"} data-theme={theme}>
      <body>
        <ClientLayout initialLocale={locale} hasStoredLocale={stored !== null} initialTheme={theme}>
          {children}
        </ClientLayout>
      </body>
    </html>
  );
}
