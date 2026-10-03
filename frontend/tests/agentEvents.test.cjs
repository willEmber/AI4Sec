const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const ts = require("typescript");

// Compile the repository's reducer, whose imports are types only. No browser,
// network or additional test dependencies are needed.
const file = path.resolve(__dirname, "../src/lib/agentEvents.ts");
const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const compiled = new Module(file, module);
compiled._compile(code, file);
const { applyToolEvent, foldToolActivity } = compiled.exports;
const event = (seq, type, payload) => ({ seq, type, payload, run_id: "run", timestamp: "2026-10-04 03:00:00" });

test("queue and page progress survive live events and persisted replay", () => {
  const events = [
    event(1, "tool.started", { tool: "ensure_paper_parsed", call_id: "call" }),
    event(2, "tool.progress", { tool: "ensure_paper_parsed", step: "mineru_parse", status: "running", phase: "pending", elapsed_s: 60 }),
    event(3, "tool.progress", { tool: "ensure_paper_parsed", step: "mineru_parse", status: "running", phase: "running", elapsed_s: 80, extracted_pages: 3, total_pages: 8 }),
    event(4, "tool.progress", { tool: "ensure_paper_parsed", step: "mineru_parse", status: "done", elapsed_s: 100 }),
    event(5, "tool.completed", { tool: "ensure_paper_parsed", status: "ok" }),
  ];
  const live = events.reduce(applyToolEvent, []);
  assert.deepEqual(foldToolActivity(events), live);
  assert.equal(live[0].steps.length, 1);
  assert.equal(live[0].steps[0].status, "done");
  assert.equal(live[0].steps[0].extracted_pages, 3);
  assert.equal(live[0].steps[0].total_pages, 8);
});

test("a queue timeout keeps the step waiting and the tool partial", () => {
  const tools = foldToolActivity([
    event(1, "tool.started", { tool: "run_insight_snap" }),
    event(2, "tool.progress", { tool: "run_insight_snap", step: "mineru_parse", status: "running", phase: "pending", elapsed_s: 300 }),
    event(3, "tool.progress", { tool: "run_insight_snap", step: "mineru_parse", status: "waiting" }),
    event(4, "tool.completed", { tool: "run_insight_snap", status: "partial" }),
  ]);
  assert.equal(tools[0].status, "partial");
  assert.equal(tools[0].steps[0].status, "waiting");
  assert.equal(tools[0].steps[0].phase, "pending");
});
