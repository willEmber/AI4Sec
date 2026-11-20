const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const ts = require("typescript");

// The module imports only types, which transpiling erases.
const file = path.resolve(__dirname, "../src/lib/reportCitations.ts");
const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const compiled = new Module(file, module);
compiled._compile(code, file);
const { frozenEvidence, parseCitationTable } = compiled.exports;

const A = "ev_" + "a".repeat(20);

test("a report written as an answer carries its citation table", () => {
  const json = JSON.stringify({
    mode: "lens",
    citations: {
      [A]: { n: 1, source_level: "fulltext", paper_id: "p1", title: "T", page_label: 5,
        quote: "q", source_url: "", provider: "mineru" },
      bogus: { quote: "no number" },
    },
  });
  const table = parseCitationTable(json);
  assert.deepEqual(Object.keys(table), [A]);
  assert.equal(table[A].n, 1);
});

test("older reports and broken payloads have no table", () => {
  assert.deepEqual(parseCitationTable(JSON.stringify({ mode: "lens", digest: {} })), {});
  assert.deepEqual(parseCitationTable(JSON.stringify({ citations: [1, 2] })), {});
  assert.deepEqual(parseCitationTable("not json"), {});
  assert.deepEqual(parseCitationTable(""), {});
});

test("a frozen citation renders as the evidence it froze", () => {
  const evidence = frozenEvidence(A, {
    n: 2, source_level: "fulltext", paper_id: "p1", title: "Paper", page_label: 5,
    section: "3 Method", quote: "Routing is sparse.", source_url: "", provider: "mineru",
  });
  assert.equal(evidence.evidence_id, A);
  assert.equal(evidence.locator.page_label, 5);
  assert.equal(evidence.locator.page_index, 4);
  assert.equal(evidence.locator.section_path, "3 Method");
  assert.equal(evidence.literature.title, "Paper");
  assert.equal(evidence.parse_status, undefined);

  const web = frozenEvidence(A, {
    n: 3, source_level: "external_web", paper_id: "", title: "", page_label: null,
    quote: "x", source_url: "https://example.org", provider: "tavily",
  });
  assert.equal(web.literature, undefined);
  assert.equal(web.locator.page_index, null);
});
