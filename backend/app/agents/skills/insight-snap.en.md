---
version: 2
required: One-line verdict | Contributions | Key results | Who should read it | Limitations | External signals & blind review | Reading advice
---
# Insight Snap: method

An Insight Snap lets a researcher decide in a minute whether a paper deserves a full read: what it claims, how well its results support that, what the outside record says, and a verdict.

## Reading (tools, before writing)

1. In ONE step, call `get_triage_signals` and `review_paper_blind` together (pass the reader's concern as `focus` when they named one). They run side by side.
2. In the same step or the next, read the evidence yourself: `search_paper_content` or `read_paper_section` for the abstract and introduction (the claims), the main results table or section (the numbers), and the conclusion or limitations. Read only what the seven sections need.
3. No web search and no other papers: a Snap is about this paper and its record.

Four to seven tool calls are usually enough. Then write.

## Writing (the final answer)

After the last tool call, write the report and nothing else: no preface, no closing offer. The text after your last tool call is saved as the report. Start with a `# ` title line naming the paper, then these seven `## ` sections, in this order:

## One-line verdict
One or two sentences: what the paper does and whether it is worth reading, with the verdict tier `review_paper_blind` gave.

## Contributions
Two to four bullets: what is new, each with the problem it addresses.

## Key results
The headline numbers against the strongest baseline, as a short table or bullets, each figure cited. Say under which setting they hold.

## Who should read it
One short paragraph: which readers, for what purpose.

## Limitations
Two to four bullets: assumptions, missing comparisons, threats to the conclusions — from the paper, and the blind review's red flags (attributed).

## External signals & blind review
- The signals as facts, each cited with its evidence number: venue and its separate system tiers (CCF / CORE / JCR / CAS — distinguish main track, `Findings`, `Workshop` and preprints), citations (name the index — counts differ between indexes), open access, retraction or CAS early-warning red flags, code. Say what is unknown rather than guessing.
- The blind review's scores in one line (contribution / evidence / novelty / reproducibility out of 3, limitation severity) and its main reasons, attributed to the review.

## Reading advice
The verdict tier exactly as given (must read / selective / skip) and why, in two or three sentences; name the sections worth reading first (the review's `must_read_sections` where they agree with what you read). When the reader named a concern, say how well the paper answers it.

## Rules

1. Every factual statement about the paper cites the passage it came from, `[E12]`; every signal cites its fact's evidence number. A sentence holding facts from different evidence — the paper's claim, its venue, a citation count — cites each of them. The blind review is an opinion: attribute it, do not cite it as the paper, and do not copy its page marks as citations.
2. Do not change the verdict tier. When your reading disagrees with it, say so beside it.
3. Short: a Snap is read in a minute. About 400-700 words.
4. Write in the conversation's language. Keep paper titles, venue, metric and dataset names as they are.
