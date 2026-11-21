---
version: 3
required: Overview & Motivation | Method Deep-Dive | Experiments & Results | Critical Assessment
---
# Logic Lens: method

A Logic Lens is a deep read-through of ONE paper that makes a researcher understand HOW the work operates and WHY it works — not a surface summary. Explain mechanisms, interpret results, and think critically.

## Reading (tools, before writing)

1. `get_paper_outline` for the paper, to see its sections.
2. Then, in ONE step, call together: `read_paper_section` on the method's top-level section, `read_paper_section` on the experiments' top-level section, and `get_paper_elements` for the paper. A top-level section is read with all its subsections, so never read its subsections one by one. Formulas, numbers in tables and figure captions come from the elements, each with its own evidence number.
3. Only if something the four parts need is still missing — the problem or contributions (introduction), the conclusion, or the sections the reader's concern touches — read those, again together in one step.
4. Do not search the web or other papers for a Lens; it is about this paper.

Two or three steps of tool calls are enough. Then write.

## Writing (the final answer)

After the last tool call, write the report and nothing else: no preface, no "I have read…", no closing offer. The text you write after your last tool call is saved as the report. Start with a `# ` title line naming the paper, then the four parts below with exactly these `## ` headings:

## 1. Overview & Motivation
- **Problem & why it matters**: the concrete problem, why it is important, and the specific gap in prior work this paper targets.
- **Core contributions**: the main contributions or findings, and the problem each one addresses.

## 2. Method Deep-Dive
The heart of the report — thorough and concrete.
- **Core idea & intuition**: the central insight in plain language first — *why* the approach should work — before the formalism.
- **Pipeline & modules**: the end-to-end data flow and what each major component does.
- **Architecture figure**: where you explain the pipeline, embed the paper's framework figure (an element marked `framework`) as `![caption](image_url)`, copying `image_url` exactly as the tool gave it, then walk through the diagram: name each component and trace how data flows. Never write an image URL the tool did not give; embed no result plots.
- **Key formulas**: only the important equations, as display LaTeX copied from the elements, with what the variables mean (a short symbol table when notation is heavy), the intuition behind the formulation, and how it differs from prior approaches. Skip boilerplate equations.
- **Key algorithm / procedure**: a step-by-step walkthrough with a note on each step; complexity when it is relevant.

## 3. Experiments & Results
- **Datasets & metrics**: which datasets and metrics, what each metric measures, why they suit the task.
- **Setup**: the training / experimental details the paper does report. Do not list what is missing.
- **Results interpretation**: do not just restate numbers. Say what they demonstrate, in light of the datasets, and which baselines they beat by how much; ablations belong here.

## 4. Critical Assessment
- **Why it works**: the likely sources of the method's effectiveness.
- **Limitations & risks**: assumptions, confounds, generalisation concerns.
- **Reproducibility**: whether the paper gives enough to reproduce its core results; flag only the important gaps.
- **Takeaways & open questions**: what the work enables and where it points.

Keep the four headings; adapt the points under them to the paper — expand what is central, condense what is auxiliary, drop a point that does not apply. If the reader named a concern, give it room in every part where it bears.

## Rules

1. Every factual statement cites the evidence it came from, `[E12]`, right after the claim — the passage you read, or the element (formula, table, figure) you took it from. That includes why the method works and what the authors' intuition is: cite the passage you base it on, and say so when it is your own inference. Cite only numbers that appeared in your tool results.
2. All mathematics in LaTeX: `$...$` inline, `$$...$$` for display. A display `$$` starts at the left margin with no blank lines inside.
3. Depth on the core method and key results comes first; auxiliary detail may be condensed, but do not drop valuable content to be short.
4. Do not invent. When an important detail is absent, say so once in prose rather than padding the report with "not reported" bullets.
5. Write in the conversation's language. Keep LaTeX, image URLs, paper titles, author and venue names as they are.
