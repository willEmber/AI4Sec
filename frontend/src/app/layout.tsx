import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Scholar Platform",
  description: "Academic paper reading and analysis platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <nav className="border-b border-[var(--border)] px-6 py-3 flex items-center gap-6">
          <a href="/" className="font-bold text-lg">Scholar</a>
          <a href="/upload" className="text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
            Upload & Analyze
          </a>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
