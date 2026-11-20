const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const ts = require("typescript");

function load(relative) {
  const file = path.resolve(__dirname, relative);
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const compiled = new Module(file, module);
  compiled._compile(code, file);
  return compiled.exports;
}

const { mergeLibraryItems, canSubmitTurn, resolveTurnPrompt } = load("../src/lib/libraryHub.ts");
const { parseTheme } = load("../src/lib/locale.ts");

test("mergeLibraryItems deduplicates papers, attaches runs, and preserves standalone runs", () => {
  const papers = [
    {
      session_id: "s1",
      literature_id: "lit_a",
      paper_id: "p_a",
      availability: "pdf_ready",
      added_by: "user",
      note: "",
      title: "Paper A",
      year: 2025,
      year_known: true,
      venue: "NeurIPS",
      doi: "",
      arxiv_id: "",
      updated_at: "2026-10-01 10:00:00",
    },
    {
      session_id: "s2",
      literature_id: "lit_c",
      paper_id: "",
      availability: "candidate",
      added_by: "agent",
      note: "",
      title: "Metadata Only Work",
      year: 2024,
      year_known: true,
      venue: "ICLR",
      doi: "",
      arxiv_id: "",
      updated_at: "2026-10-02 09:00:00",
    },
  ];

  const runs = [
    {
      run_id: "r1",
      paper_id: "p_a",
      paper_title: "Paper A",
      mode: "snap",
      status: "done",
      started_at: "2026-10-03 08:00:00",
      finished_at: "2026-10-03 08:02:00",
      current_step: "",
      user_question: "",
      agent_session_id: "s1",
    },
    {
      run_id: "r2",
      paper_id: "p_legacy",
      paper_title: "Legacy Upload Paper",
      mode: "lens",
      status: "done",
      started_at: "2026-10-01 12:00:00",
      finished_at: "2026-10-01 12:05:00",
      current_step: "",
      user_question: "",
      agent_session_id: "",
    },
  ];

  const merged = mergeLibraryItems(papers, runs);
  assert.equal(merged.length, 3);
  // Paper A got bumped to top by its 2026-10-03 run and promoted to parsed.
  assert.equal(merged[0].paperId, "p_a");
  assert.equal(merged[0].availability, "parsed");
  assert.equal(merged[0].runs.length, 1);
  assert.equal(merged[1].literatureId, "lit_c");
  assert.equal(merged[2].paperId, "p_legacy");
  assert.equal(merged[2].title, "Legacy Upload Paper");
});

test("canSubmitTurn requires text in auto mode, but allows one-click report modes with a readable paper", () => {
  assert.equal(canSubmitTurn({ draft: "", mode: "auto", hasReadablePaper: true }), false);
  assert.equal(canSubmitTurn({ draft: "What is the method?", mode: "auto", hasReadablePaper: false }), true);
  assert.equal(canSubmitTurn({ draft: "", mode: "snap", hasReadablePaper: false }), false);
  assert.equal(canSubmitTurn({ draft: "", mode: "snap", hasReadablePaper: true }), true);
  assert.equal(canSubmitTurn({ draft: "", mode: "lens", hasReadablePaper: true, sending: true }), false);
  assert.equal(canSubmitTurn({ draft: "focus", mode: "lens", hasReadablePaper: true, tooLong: true }), false);
});

test("resolveTurnPrompt keeps user text or synthesizes the selected report mode prompt", () => {
  const defaults = (mode) => `default:${mode}`;
  assert.equal(resolveTurnPrompt("  focus on ablation  ", "lens", defaults), "focus on ablation");
  assert.equal(resolveTurnPrompt("   ", "snap", defaults), "default:snap");
  assert.equal(resolveTurnPrompt("   ", "auto", defaults), "");
});

test("parseTheme accepts only valid theme tokens", () => {
  assert.equal(parseTheme("light"), "light");
  assert.equal(parseTheme("dark"), "dark");
  assert.equal(parseTheme("system"), "system");
  assert.equal(parseTheme("sepia"), null);
  assert.equal(parseTheme(undefined), null);
});
