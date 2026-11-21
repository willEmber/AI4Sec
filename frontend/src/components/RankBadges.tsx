interface RankBadgesProps {
  venue?: string;
  year?: number;
  sciRank?: string;
  ccfRank?: string;
  venueRanks?: Record<string, string>;
  subtrack?: string;
  casWarning?: string;
}

export const SCI_COLORS: Record<string, string> = {
  Q1: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  Q2: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  Q3: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
  Q4: "bg-gray-50 text-gray-600 border-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-600",
};

export const CCF_COLORS: Record<string, string> = {
  A: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  B: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  C: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
};

export const CORE_COLORS: Record<string, string> = {
  "A*": "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  A: "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-300 dark:border-emerald-800",
  B: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  C: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
};

const CAS_COLORS: Record<string, string> = {
  "1": "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  "2": "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  "3": "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
  "4": "bg-gray-50 text-gray-600 border-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-600",
};

export const WARNING_CHIP_STYLE =
  "bg-destructive/10 text-destructive border-destructive/35 font-semibold";
const SUBTRACK_STYLE =
  "bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-950 dark:text-sky-300 dark:border-sky-800";

export const BADGE_BASE =
  "inline-flex items-center px-1.5 py-0.5 text-xs font-semibold rounded-md border";
const VENUE_STYLE = "bg-muted text-foreground/70 border-border";
const YEAR_STYLE = "bg-muted text-muted-foreground border-border";

function headToken(label: string): string {
  return label.split("(", 1)[0].trim();
}

export interface RankChipDescriptor {
  key: string;
  label: string;
  title?: string;
  className: string;
}

/** Build ordered multi-system rank, subtrack and early-warning chips. */
export function buildRankChips({
  sciRank,
  ccfRank,
  venueRanks,
  subtrack,
  casWarning,
  warningPrefix = "CAS Warning",
}: {
  sciRank?: string;
  ccfRank?: string;
  venueRanks?: Record<string, string>;
  subtrack?: string;
  casWarning?: string;
  warningPrefix?: string;
}): RankChipDescriptor[] {
  const chips: RankChipDescriptor[] = [];
  const sub = (subtrack || "main").toLowerCase();

  if (sub === "workshop" || sub === "companion") {
    chips.push({
      key: "subtrack",
      label: sub === "companion" ? "Companion" : "Workshop",
      className: VENUE_STYLE,
    });
  } else {
    if (sub === "findings") {
      chips.push({
        key: "findings",
        label: "Findings",
        className: SUBTRACK_STYLE,
      });
    }

    const ccfFull = venueRanks?.CCF || "";
    const ccfKey = (ccfFull ? headToken(ccfFull) : ccfRank || "").toUpperCase();
    if (ccfKey && CCF_COLORS[ccfKey]) {
      chips.push({
        key: "CCF",
        label: `CCF ${ccfKey}`,
        title: ccfFull ? `CCF ${ccfFull}` : undefined,
        className: CCF_COLORS[ccfKey],
      });
    }

    const jcrFull = venueRanks?.JCR || "";
    const jcrKey = (jcrFull ? headToken(jcrFull) : sciRank || "").toUpperCase();
    if (jcrKey && SCI_COLORS[jcrKey]) {
      chips.push({
        key: "JCR",
        label: jcrFull ? `JCR ${jcrKey}` : `SCI ${jcrKey}`,
        title: jcrFull ? `JCR ${jcrFull}` : undefined,
        className: SCI_COLORS[jcrKey],
      });
    }

    const coreFull = venueRanks?.CORE || "";
    const coreKey = headToken(coreFull).toUpperCase();
    if (coreKey && CORE_COLORS[coreKey]) {
      chips.push({
        key: "CORE",
        label: `CORE ${coreKey}`,
        title: `CORE ${coreFull}`,
        className: CORE_COLORS[coreKey],
      });
    }

    const casFull = venueRanks?.CAS || "";
    if (casFull) {
      const shortCas = headToken(casFull);
      const divMatch = shortCas.match(/([1-4])/);
      const casStyle = (divMatch && CAS_COLORS[divMatch[1]]) || VENUE_STYLE;
      chips.push({
        key: "CAS",
        label: `CAS ${shortCas}`,
        title: `CAS ${casFull}`,
        className: casStyle,
      });
    }

    for (const [sysKey, prefix] of [
      ["CCF_CN", "CCF-CN"],
      ["TH_CPL", "TH-CPL"],
      ["CAAI", "CAAI"],
    ] as const) {
      const full = venueRanks?.[sysKey] || "";
      if (full) {
        const short = headToken(full);
        chips.push({
          key: sysKey,
          label: `${prefix} ${short}`,
          title: `${prefix} ${full}`,
          className: CCF_COLORS[short.toUpperCase()] || CORE_COLORS.A,
        });
      }
    }
  }

  if (casWarning) {
    chips.push({
      key: "cas_warning",
      label: `${warningPrefix}: ${casWarning}`,
      className: WARNING_CHIP_STYLE,
    });
  }
  return chips;
}

export function MultiSystemRankChips(props: {
  sciRank?: string;
  ccfRank?: string;
  venueRanks?: Record<string, string>;
  subtrack?: string;
  casWarning?: string;
  warningPrefix?: string;
}) {
  const chips = buildRankChips(props);
  if (chips.length === 0) return null;
  return (
    <>
      {chips.map((c) => (
        <span key={c.key} title={c.title} className={`${BADGE_BASE} ${c.className}`}>
          {c.label}
        </span>
      ))}
    </>
  );
}

export default function RankBadges({
  venue,
  year,
  sciRank,
  ccfRank,
  venueRanks,
  subtrack,
  casWarning,
}: RankBadgesProps) {
  const chips = buildRankChips({ sciRank, ccfRank, venueRanks, subtrack, casWarning });
  const hasVenue = Boolean(venue);
  const hasYear = Boolean(year && year > 0);

  if (!hasVenue && !hasYear && chips.length === 0) return null;

  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {hasVenue && venue && (
        <span className={`${BADGE_BASE} ${VENUE_STYLE}`}>
          {venue.length > 40 ? venue.slice(0, 37) + "..." : venue}
        </span>
      )}
      {hasYear && (
        <span className={`${BADGE_BASE} ${YEAR_STYLE}`}>{String(year)}</span>
      )}
      {chips.map((b) => (
        <span key={b.key} title={b.title} className={`${BADGE_BASE} ${b.className}`}>
          {b.label}
        </span>
      ))}
    </span>
  );
}
