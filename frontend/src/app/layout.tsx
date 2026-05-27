import "./globals.css";
import type { Metadata } from "next";
import ClientLayout from "./client-layout";

export const metadata: Metadata = {
  title: "Scholar — AI Paper Reading",
  description: "Upload a paper, choose a reading mode, and get structured AI analysis with citations linking back to the source PDF.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
