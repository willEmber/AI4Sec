const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const ts = require("typescript");

// Compile the pure helpers directly; neither imports anything at runtime.
function load(relative) {
  const file = path.resolve(__dirname, relative);
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const compiled = new Module(file, module);
  compiled._compile(code, file);
  return compiled.exports;
}

const { answerOutcome, errorTextKey } = load("../src/lib/outcome.ts");
const { isLocalImageSrc } = load("../src/lib/safeUrl.ts");
const { applyTurnEvent, EMPTY_TURN } = load("../src/lib/agentEvents.ts");

test("a finished answer has nothing under it", () => {
  assert.equal(answerOutcome({ completion: "complete", stop_reason: "" }), null);
  // Rows written before the columns existed.
  assert.equal(answerOutcome({}), null);
});

test("an answer written because a budget ran out says so and offers nothing to press", () => {
  const outcome = answerOutcome({ completion: "complete", stop_reason: "max_tool_calls" });
  assert.equal(outcome.key, "chat.answer.wrapped_up");
  assert.equal(outcome.reasonKey, "chat.limit.max_tool_calls");
  assert.equal(outcome.action, null);
});

test("a failed turn's text is marked with the failure and can be retried", () => {
  const outcome = answerOutcome({ completion: "failed", stop_reason: "upstream_error" });
  assert.deepEqual(
    [outcome.tone, outcome.key, outcome.reasonKey, outcome.action],
    ["error", "chat.answer.failed", "chat.error.upstream_error", "retry"],
  );
});

test("a cut answer continues, a stopped one retries", () => {
  assert.equal(answerOutcome({ completion: "truncated", stop_reason: "max_output_tokens" }).action, "continue");
  const stopped = answerOutcome({ completion: "stopped", stop_reason: "cancelled" });
  assert.deepEqual([stopped.key, stopped.action], ["chat.answer.stopped", "retry"]);
  const cutOff = answerOutcome({ completion: "stopped", stop_reason: "max_tokens" });
  assert.deepEqual([cutOff.key, cutOff.reasonKey], ["chat.answer.cut_off", "chat.limit.max_tokens"]);
});

test("an unknown code is never shown as if it were known", () => {
  assert.equal(errorTextKey("something_new"), "chat.error.unknown");
  assert.equal(errorTextKey(""), "chat.error.unknown");
  assert.equal(errorTextKey("rate_limited"), "chat.error.rate_limited");
});

test("only the application's own images load", () => {
  const origin = "https://scholar.example";
  assert.equal(isLocalImageSrc("/api/papers/abc/images/fig1.png", origin), true);
  assert.equal(isLocalImageSrc("https://scholar.example/api/papers/abc/images/fig1.png", origin), true);
  assert.equal(isLocalImageSrc("data:image/png;base64,AAAA", origin), true);

  assert.equal(isLocalImageSrc("https://evil.example/?q=secret", origin), false);
  assert.equal(isLocalImageSrc("//evil.example/x.png", origin), false);
  assert.equal(isLocalImageSrc("/\\evil.example/x.png", origin), false);
  assert.equal(isLocalImageSrc("data:image/svg+xml;base64,AAAA", origin), false);
  assert.equal(isLocalImageSrc("data:text/html,<script>", origin), false);
  assert.equal(isLocalImageSrc("javascript:alert(1)", origin), false);
  assert.equal(isLocalImageSrc("", origin), false);
  // Without an origin to compare with, an absolute URL is not ours.
  assert.equal(isLocalImageSrc("https://scholar.example/x.png"), false);
});

test("a move to the fallback model is part of the turn's timeline", () => {
  const event = { seq: 7, type: "model.fallback", payload: { from: "a", to: "b", code: "timeout" }, run_id: "r" };
  const next = applyTurnEvent(EMPTY_TURN, event);
  assert.deepEqual(next.timeline, [{ kind: "fallback", key: "fallback-7", model: "b" }]);
});
