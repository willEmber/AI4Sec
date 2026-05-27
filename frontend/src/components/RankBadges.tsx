interface RankBadgesProps {
  venue?: string;
  year?: number;
  sciRank?: string;
  ccfRank?: string;
}

const SCI_COLORS: Record<string, string> = {
  Q1: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  Q2: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  Q3: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
  Q4: "bg-gray-50 text-gray-600 border-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-600",
};

const CCF_COLORS: Record<string, string> = {
  A: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  B: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  C: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
};

const BADGE_BASE = "inline-flex items-center px-1.5 py-0.5 text-xs font-semibold rounded-md border";
const VENUE_STYLE = "bg-muted text-foreground/70 border-border";
const YEAR_STYLE = "bg-muted text-muted-foreground border-border";

export default function RankBadges({ venue, year, sciRank, ccfRank }: RankBadgesProps) {
  const badges: { label: string; className: string }[] = [];

  // Always show venue
  if (venue) {
    badges.push({
      label: venue.length > 40 ? venue.slice(0, 37) + "..." : venue,
      className: VENUE_STYLE,
    });
  }

  // Always show year
  if (year && year > 0) {
    badges.push({ label: String(year), className: YEAR_STYLE });
  }

  // SCI/CCF rank badges
  if (sciRank && SCI_COLORS[sciRank]) {
    badges.push({ label: `SCI ${sciRank}`, className: SCI_COLORS[sciRank] });
  }

  if (ccfRank && CCF_COLORS[ccfRank]) {
    badges.push({ label: `CCF ${ccfRank}`, className: CCF_COLORS[ccfRank] });
  }

  if (badges.length === 0) return null;

  return (
    <span className="inline-flex items-center gap-1">
      {badges.map((b, i) => (
        <span key={i} className={`${BADGE_BASE} ${b.className}`}>
          {b.label}
        </span>
      ))}
    </span>
  );
}
