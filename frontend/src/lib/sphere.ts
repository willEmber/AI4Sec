/**
 * Research Sphere structured payload.
 *
 * The sphere subgraph emits both a markdown report (used for export) and a
 * JSON twin of the same content in `run_outputs.json_data`. The JSON carries
 * the report as fields — nodes, partitions, themes, timeline, comparison rows,
 * hubs, gaps and reading paths — so the UI can render cards, badges and
 * cross-links instead of a flat wall of markdown.
 */

export interface SphereNodeData {
  node_id: string;
  title: string;
  year: number;
  venue: string;
  authors: string;
  doi: string;
  arxiv_id: string;
  openalex_id: string;
  cited_by_count: number;
  sci_rank: string;
  ccf_rank: string;
  tier: number;
  quality_score: number;
  relevance: number;
  relation_type: string;
  relation_reason: string;
  reason: string;
  cluster_id: number;
  influential: boolean;
  library_document_id: string;
  method_summary: string;
  contribution: string;
  task: string;
  dataset: string;
}

export interface SpherePartition {
  key: string;
  node_ids: string[];
}

export interface SphereTheme {
  name: string;
  definition: string;
  representative_node_ids: string[];
}

export interface SphereTimelineEntry {
  year: number;
  node_ids: string[];
  summary: string;
}

export interface SphereHub {
  node_id: string;
  title: string;
  pagerank: number;
  cited_by_count: number;
  reason: string;
}

export interface SphereComparisonRow {
  node_id: string;
  title: string;
  problem: string;
  assumption: string;
  method: string;
  dataset: string;
  metric: string;
  strength: string;
  weakness: string;
}

export interface SphereGap {
  title: string;
  description: string;
  evidence_node_ids: string[];
}

export interface SphereReadingPath {
  name: string;
  description: string;
  node_ids: string[];
}

export interface SphereLibraryMatch {
  document_id: string;
  title: string;
  score: number;
  snippet: string;
}

export interface SphereData {
  mode: string;
  paper_id: string;
  title: string;
  num_nodes: number;
  num_edges: number;
  num_core: number;
  num_layer2: number;
  center_node_id: string;
  nodes: Record<string, SphereNodeData>;
  partitions: SpherePartition[];
  library_matches: SphereLibraryMatch[];
  sphere_output: {
    sphere_overview: string;
    partitions: SpherePartition[];
    themes: SphereTheme[];
    timeline: SphereTimelineEntry[];
    key_hubs: SphereHub[];
    comparison_table: SphereComparisonRow[];
    gaps_and_ideas: SphereGap[];
    reading_paths: SphereReadingPath[];
  };
}

/**
 * Parse `json_data` into a sphere payload.
 *
 * Returns null for any other mode, for malformed JSON, and for reports
 * produced before the node dictionary was added — callers fall back to the
 * markdown renderer in those cases.
 */
export function parseSphereData(jsonData: string | undefined | null): SphereData | null {
  if (!jsonData) return null;
  try {
    const parsed = JSON.parse(jsonData) as Partial<SphereData>;
    if (parsed?.mode !== "sphere") return null;
    if (!parsed.nodes || typeof parsed.nodes !== "object") return null;
    if (!parsed.sphere_output) return null;
    return {
      mode: "sphere",
      paper_id: parsed.paper_id || "",
      title: parsed.title || "",
      num_nodes: parsed.num_nodes || 0,
      num_edges: parsed.num_edges || 0,
      num_core: parsed.num_core || 0,
      num_layer2: parsed.num_layer2 || 0,
      center_node_id: parsed.center_node_id || "",
      nodes: parsed.nodes,
      partitions: parsed.partitions || [],
      library_matches: parsed.library_matches || [],
      sphere_output: {
        sphere_overview: parsed.sphere_output.sphere_overview || "",
        partitions: parsed.sphere_output.partitions || [],
        themes: parsed.sphere_output.themes || [],
        timeline: parsed.sphere_output.timeline || [],
        key_hubs: parsed.sphere_output.key_hubs || [],
        comparison_table: parsed.sphere_output.comparison_table || [],
        gaps_and_ideas: parsed.sphere_output.gaps_and_ideas || [],
        reading_paths: parsed.sphere_output.reading_paths || [],
      },
    };
  } catch {
    return null;
  }
}

/** Best external link for a node: DOI → arXiv → OpenAlex → Scholar search. */
export function nodeExternalUrl(node: SphereNodeData): string {
  if (node.doi) {
    const doi = node.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, "");
    return `https://doi.org/${doi}`;
  }
  if (node.arxiv_id) {
    return `https://arxiv.org/abs/${node.arxiv_id.replace(/^arxiv:/i, "")}`;
  }
  if (node.openalex_id) {
    const id = node.openalex_id.replace(/^https?:\/\/openalex\.org\//i, "");
    return `https://openalex.org/${id}`;
  }
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(node.title)}`;
}

/** Partition display order — foundational work first, frontier last. */
export const PARTITION_ORDER = [
  "foundation",
  "method",
  "follow_up",
  "application",
  "frontier",
  "survey",
  "library",
] as const;

/** Per-partition accent, applied to chips and paper-card rails. */
export const PARTITION_ACCENT: Record<string, { dot: string; chip: string; rail: string }> = {
  foundation: {
    dot: "bg-amber-500",
    chip: "border-amber-500/35 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    rail: "border-l-amber-500/60",
  },
  method: {
    dot: "bg-rose-500",
    chip: "border-rose-500/35 bg-rose-500/10 text-rose-700 dark:text-rose-300",
    rail: "border-l-rose-500/60",
  },
  follow_up: {
    dot: "bg-emerald-500",
    chip: "border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    rail: "border-l-emerald-500/60",
  },
  application: {
    dot: "bg-sky-500",
    chip: "border-sky-500/35 bg-sky-500/10 text-sky-700 dark:text-sky-300",
    rail: "border-l-sky-500/60",
  },
  frontier: {
    dot: "bg-violet-500",
    chip: "border-violet-500/35 bg-violet-500/10 text-violet-700 dark:text-violet-300",
    rail: "border-l-violet-500/60",
  },
  survey: {
    dot: "bg-teal-500",
    chip: "border-teal-500/35 bg-teal-500/10 text-teal-700 dark:text-teal-300",
    rail: "border-l-teal-500/60",
  },
  library: {
    dot: "bg-orange-500",
    chip: "border-orange-500/35 bg-orange-500/10 text-orange-700 dark:text-orange-300",
    rail: "border-l-orange-500/60",
  },
};

export const DEFAULT_ACCENT = {
  dot: "bg-muted-foreground",
  chip: "border-border bg-muted text-muted-foreground",
  rail: "border-l-border",
};

/** Theme-cluster palette, indexed by cluster order. */
export const CLUSTER_ACCENT = [
  "bg-rose-500",
  "bg-sky-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-violet-500",
  "bg-teal-500",
];

/** Compact citation counts: 99111 → 99.1k. */
export function formatCitations(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  if (n >= 1_000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

/** DOM id used to cross-link a node from anywhere in the report. */
export function nodeAnchorId(nodeId: string): string {
  return `sphere-node-${nodeId}`;
}

/**
 * Scroll a node's card into view and flash it.
 * Returns false when the node has no card in the report (e.g. it only appears
 * in the timeline), letting callers fall back to the external link.
 */
export function focusNode(nodeId: string): boolean {
  const el = document.getElementById(nodeAnchorId(nodeId));
  if (!el) return false;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("sphere-flash");
  window.setTimeout(() => el.classList.remove("sphere-flash"), 1600);
  return true;
}
