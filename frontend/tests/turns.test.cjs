const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const ts = require("typescript");

// Same approach as agentEvents.test.cjs: the module's imports are types only.
const file = path.resolve(__dirname, "../src/lib/turns.ts");
const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const compiled = new Module(file, module);
compiled._compile(code, file);
const { groupTurns, dateBucket, citationNumbers } = compiled.exports;

const msg = (role, run_id, content = "") => ({ role, run_id, content, message_id: `${role}-${run_id}-${content}` });

test("a turn is a question and the replies up to the next question", () => {
  const turns = groupTurns([
    msg("user", "r1", "q1"),
    msg("assistant", "r1", "a1"),
    msg("user", "r2", "q2"),
    msg("user", "", "q3"),
  ]);
  assert.deepEqual(turns.map((t) => [t.index, t.runId, t.question.content, t.replies.length]), [
    [0, "r1", "q1", 1],
    [1, "r2", "q2", 0],
    [2, "", "q3", 0],
  ]);
});

test("an answer with no question before it is kept", () => {
  const turns = groupTurns([msg("assistant", "r0", "hello"), msg("user", "r1", "q")]);
  assert.equal(turns.length, 2);
  assert.equal(turns[0].question, null);
  assert.equal(turns[0].runId, "r0");
});

test("date buckets follow the local day, not a 24-hour window", () => {
  const now = new Date(2026, 9, 8, 0, 30).getTime();
  assert.equal(dateBucket(new Date(2026, 9, 8, 0, 5).getTime(), now), "today");
  assert.equal(dateBucket(new Date(2026, 9, 7, 23, 50).getTime(), now), "week");
  assert.equal(dateBucket(new Date(2026, 9, 2, 12, 0).getTime(), now), "week");
  assert.equal(dateBucket(new Date(2026, 9, 1, 23, 0).getTime(), now), "earlier");
  assert.equal(dateBucket(NaN, now), "earlier");
});

test("a source cited twice keeps its first number", () => {
  const a = "ev_" + "a".repeat(20);
  const b = "ev_" + "b".repeat(20);
  const numbers = citationNumbers(`x [${a}] y [${b}] z [${a}]`);
  assert.deepEqual([...numbers], [[a, 1], [b, 2]]);
});
