import "./globals.css";
import type { Metadata } from "next";
import { cookies } from "next/headers";
import ClientLayout from "./client-layout";
import { DEFAULT_LOCALE, LOCALE_COOKIE, parseLocale } from "@/lib/locale";

export const metadata: Metadata = {
  title: "Scholar — AI Paper Reading",
  description: "Upload a paper, choose a reading mode, and get structured AI analysis with citations linking back to the source PDF.",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Read the language on the server so the first HTML is already in it;
  // resolving it after mount showed every page in the default language first.
  const stored = parseLocale((await cookies()).get(LOCALE_COOKIE)?.value);
  const locale = stored ?? DEFAULT_LOCALE;

  return (
    <html lang={locale === "zh" ? "zh-CN" : "en"}>
      <body>
        <ClientLayout initialLocale={locale} hasStoredLocale={stored !== null}>
          {children}
        </ClientLayout>
      </body>
    </html>
  );
}
