---
version: 3
required: Overview | Themes | Timeline | Hubs | Comparison | Gaps | Reading Paths
---
# Research Sphere: method

A Research Sphere places ONE paper in its literature: where its ideas come from, what competes with it, what built on it, where the frontier is and what is still open — so a researcher knows what to read next, and why.

## Reading (tools, before writing)

1. In ONE step, call together: `build_literature_map` for the paper, and `get_paper_outline` for it. The map takes several minutes the first time. It returns this paper's own venue and citation count (`center`), and the core related papers by partition (foundations, method neighbours and competitors, follow-ups, applications, frontier, surveys) — each with why it was kept, its venue and citation count, and what its abstract says, each with evidence numbers — plus clusters of papers that share references, and hubs.
2. Then read ONE section of the paper with `read_paper_section`: its related-work section, or its introduction when it has none — to see how its authors place the work. Not its method or experiments: a Sphere is about the literature around the paper, not the paper's details.
3. Only when the reader named a concern the abstracts cannot settle, look further into the one or two papers it bears on (`search_paper_snippets`, or `download_paper` and then `read_paper_section`). Otherwise download nothing: the landscape is written from abstracts and metadata.
4. Search for no papers, and do not search the web: the map is the literature, and this paper's record is its `center`.

Three tool calls are usually enough. Then write.

## Writing (the final answer)

After the last tool call, write the report and nothing else: no preface, no closing offer. The text you write after your last tool call is saved as the report. Start with a `# ` title line naming the paper, then the seven parts below with these `## ` headings:

## 1. Landscape Overview
Two or three paragraphs: the field's intellectual roots, the main competing or adjacent approaches, how follow-up work evolved, where the frontier is — and this paper's place in it.

## 2. Themes
Three to six themes. Start from the map's clusters (papers that share references) and its partitions; name each theme, define it in a sentence or two and list its representative papers. A theme is a line of work, not a partition label.

## 3. Timeline
The lineage by year, from the foundations to the frontier: each year that matters, with the papers that made it and what they changed.

## 4. Key Hubs
The map's hubs — the most central papers in the reference-coupling graph — and why each matters here. When the map has none, say its graph was too sparse and name the most cited foundations instead.

## 5. Comparison
A Markdown table of this paper and the four to eight papers closest to it (method neighbours and competitors first), this paper in the first row:

| Paper | Problem | Method | Dataset | Metric | Strength | Weakness |

Fill a cell only from what an abstract, the metadata or this paper's text says; write "—" otherwise.

## 6. Research Gaps & Ideas
Three to five gaps: what the landscape leaves open, why it matters, which papers show it — concrete enough to start work on.

## 7. Reading Paths
Three numbered lists: **Fast** (3–5 papers to grasp the area), **Deep** (8–12 for a thorough understanding, in reading order) and **Frontier** (5–8 recent ones). Each entry: title, year, and one clause on why.

## Rules

1. Every statement about a paper cites its evidence right after the claim, `[E12]`: its abstract for what it does, its metadata for venue and citation count, this paper's passages for what this paper says. A sentence holding facts from different evidence — what a paper does and its venue or citation count — cites each of them, or is split in two. Cite only numbers from your tool results.
2. Name only papers from the map or your other tool results, with their titles as given. Add none from memory.
3. The map stands on abstracts and metadata. Say "according to its abstract" where it matters, and claim no detail only a full text would show.
4. Citation counts are the map's, as of its date; do not set them against another index's counts.
5. Write in the conversation's language. Keep paper titles, author and venue names as they are.
