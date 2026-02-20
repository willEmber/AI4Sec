import Link from "next/link";

export default function Home() {
  return (
    <div className="max-w-3xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-bold mb-4">Scholar Platform</h1>
      <p className="text-[var(--muted-foreground)] mb-8 text-lg">
        Upload an academic paper and get structured AI-powered analysis with evidence citations
        linking back to PDF pages.
      </p>

      <div className="grid gap-4 sm:grid-cols-3 mb-12">
        <div className="border border-[var(--border)] rounded-lg p-5">
          <h3 className="font-semibold mb-2">Insight Snap</h3>
          <p className="text-sm text-[var(--muted-foreground)]">
            30-second triage. Core contributions, key findings, and whether it&apos;s worth reading.
          </p>
        </div>
        <div className="border border-[var(--border)] rounded-lg p-5">
          <h3 className="font-semibold mb-2">Logic Lens</h3>
          <p className="text-sm text-[var(--muted-foreground)]">
            Deep analysis of formulas, algorithms, and experiments with reproduction checklists.
          </p>
        </div>
        <div className="border border-[var(--border)] rounded-lg p-5">
          <h3 className="font-semibold mb-2">Research Sphere</h3>
          <p className="text-sm text-[var(--muted-foreground)]">
            Reference network analysis with multi-paper comparison and research gap identification.
          </p>
        </div>
      </div>

      <Link
        href="/upload"
        className="inline-block bg-[var(--primary)] text-[var(--primary-foreground)] px-6 py-3 rounded-lg font-medium hover:opacity-90"
      >
        Upload Paper
      </Link>
    </div>
  );
}
