const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const ts = require("typescript");

// Compile the repository's module, which has no imports. No browser, network or
// additional test dependencies are needed.
const file = path.resolve(__dirname, "../src/lib/lens.ts");
const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const compiled = new Module(file, module);
compiled._compile(code, file);
const { defaultLensView, parseLensData, splitLensReport } = compiled.exports;

const REPORT = [
  "# Paper title",
  "",
  "## 1. 概览与动机",
  "问题 [p.1]",
  "",
  "## 2. 方法深读",
  "### 2.1 流程",
  "$$",
  "C' = W_s f_s(S)",
  "$$",
  "```",
  "## not a heading",
  "```",
  "",
  "## 3. 实验与结果",
  "Table 4 消融 [p.14]",
  "",
  "## 4. 批判性评估",
  "局限 [p.15]",
].join("\n");

test("a report splits at its four top-level headings, in order", () => {
  const prose = splitLensReport(REPORT);
  assert.equal(prose.overview, "问题 [p.1]");
  assert.match(prose.method, /^### 2\.1 流程/);
  assert.match(prose.method, /## not a heading/);
  assert.equal(prose.experiments, "Table 4 消融 [p.14]");
  assert.equal(prose.assessment, "局限 [p.15]");
});

test("nothing the report says is lost by splitting it", () => {
  const prose = splitLensReport(`开场白 [p.1]\n\n${REPORT}`);
  assert.match(prose.overview, /^开场白 \[p\.1\]/);
  const joined = Object.values(prose).join("\n");
  for (const line of REPORT.split("\n")) {
    if (line && !/^##? /.test(line)) assert.ok(joined.includes(line), line);
  }
});

test("any other heading count is not guessed at", () => {
  assert.equal(splitLensReport("## a\nx\n## b\ny"), null);
  assert.equal(splitLensReport(`${REPORT}\n\n## 5. 附录\nz`), null);
  assert.equal(splitLensReport(""), null);
});

test("a Lens run opens on its prose, other modes on their cards", () => {
  const lens = parseLensData(JSON.stringify({ mode: "lens", digest: { available: true } }));
  assert.equal(defaultLensView(lens), "markdown");
  assert.equal(defaultLensView(null), "structured");
});
