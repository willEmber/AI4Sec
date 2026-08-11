from __future__ import annotations

import json
import unittest

from app.services.lens_digest import (
    clean_latex,
    generate_lens_digest,
    parse_lens_digest,
    system_prompt,
)

_FULL = {
    "core_idea": "Attention alone, with no recurrence, suffices for translation.",
    "problem": "Recurrent models cannot be parallelized across positions.",
    "gap": "Prior attention work still kept an RNN backbone.",
    "contributions": [
        {"text": "An encoder-decoder built only from attention", "page": 2},
        {"text": "Multi-head attention", "page": 4},
    ],
    "pipeline": [
        {"name": "Input embedding", "role": "tokens to vectors", "page": 3},
        {"name": "Encoder stack", "role": "6 identical layers", "page": 3},
    ],
    "formulas": [
        {
            "name": "Scaled dot-product attention",
            "latex": r"\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V",
            "page": 4,
            "role": "Scales the logits so gradients stay usable at large $d_k$.",
            "symbols": [{"symbol": "d_k", "meaning": "key dimension"}],
        }
    ],
    "algorithm": {
        "name": "Training",
        "page": 7,
        "complexity": "O(n^2 d)",
        "steps": [{"step": "Warm up the learning rate", "note": "4000 steps"}],
    },
    "datasets": [
        {"name": "WMT14 EN-DE", "metrics": "BLEU", "measures": "n-gram overlap", "page": 8}
    ],
    "setup": [{"text": "Adam, 100k steps, 8x P100", "page": 7}],
    "findings": [
        {
            "metric": "BLEU", "dataset": "WMT14 EN-DE", "value": "28.4",
            "baseline": "26.3", "delta": "+2.1", "page": 8, "note": "single run",
        }
    ],
    "takeaways": [{"text": "Parallelism buys accuracy at lower cost", "page": 8}],
    "why_it_works": [{"text": "Every position attends to every other in one step", "page": 6}],
    "limitations": [{"text": "Quadratic in sequence length", "page": 6}],
    "reproducibility": {"score": 2, "available": ["hyperparameters"], "missing": ["seeds"]},
    "open_questions": [{"text": "Can attention handle images?", "page": 10}],
}


class TestDigestParsing(unittest.TestCase):
    def test_full_payload_round_trips(self) -> None:
        digest = parse_lens_digest(json.dumps(_FULL))
        self.assertTrue(digest.available)
        self.assertEqual(len(digest.contributions), 2)
        self.assertEqual(digest.pipeline[1].name, "Encoder stack")
        self.assertEqual(digest.formulas[0].symbols[0].symbol, "d_k")
        self.assertIsNotNone(digest.algorithm)
        assert digest.algorithm is not None
        self.assertEqual(digest.algorithm.steps[0].note, "4000 steps")
        self.assertEqual(digest.findings[0].delta, "+2.1")
        self.assertEqual(digest.reproducibility.score, 2)

    def test_fenced_and_prose_wrapped_json_are_recovered(self) -> None:
        fenced = parse_lens_digest("```json\n" + json.dumps(_FULL) + "\n```")
        wrapped = parse_lens_digest("Here it is:\n" + json.dumps(_FULL) + "\nDone.")
        self.assertTrue(fenced.available)
        self.assertTrue(wrapped.available)

    def test_invalid_json_is_unavailable(self) -> None:
        self.assertFalse(parse_lens_digest("not json at all").available)
        self.assertFalse(parse_lens_digest("").available)

    def test_a_digest_with_nothing_to_show_is_unavailable(self) -> None:
        """One paragraph is not a card view — the run keeps its markdown."""
        digest = parse_lens_digest(json.dumps({"core_idea": "A paper about attention."}))
        self.assertFalse(digest.available)

    def test_unquantified_findings_are_dropped(self) -> None:
        digest = parse_lens_digest(json.dumps({
            **_FULL,
            "findings": [
                {"metric": "quality", "dataset": "various", "note": "much better"},
                {"metric": "BLEU", "dataset": "WMT14", "value": "28.4", "page": 8},
            ],
        }))
        self.assertEqual(len(digest.findings), 1)
        self.assertEqual(digest.findings[0].value, "28.4")

    def test_pages_are_never_negative_or_fabricated_types(self) -> None:
        digest = parse_lens_digest(json.dumps({
            **_FULL,
            "contributions": [
                {"text": "no page given", "page": "unknown"},
                {"text": "negative page", "page": -3},
            ],
        }))
        self.assertEqual([c.page for c in digest.contributions], [0, 0])

    def test_reproducibility_score_is_clamped(self) -> None:
        digest = parse_lens_digest(json.dumps({**_FULL, "reproducibility": {"score": 9}}))
        self.assertEqual(digest.reproducibility.score, 3)

    def test_algorithm_without_steps_is_dropped(self) -> None:
        digest = parse_lens_digest(json.dumps({**_FULL, "algorithm": {"name": "Training"}}))
        self.assertIsNone(digest.algorithm)


class TestLatexCleaning(unittest.TestCase):
    def test_delimiters_and_environments_are_stripped(self) -> None:
        """The card typesets the body itself; a leftover wrapper renders as text."""
        self.assertEqual(clean_latex("$$E=mc^2$$"), "E=mc^2")
        self.assertEqual(clean_latex("$E=mc^2$"), "E=mc^2")
        self.assertEqual(
            clean_latex(r"\begin{equation}E=mc^2\end{equation}"), "E=mc^2"
        )
        self.assertEqual(
            clean_latex(r"$$\begin{equation}E=mc^2\end{equation}$$"), "E=mc^2"
        )

    def test_a_clean_body_is_left_alone(self) -> None:
        latex = r"\frac{QK^{\top}}{\sqrt{d_k}}"
        self.assertEqual(clean_latex(latex), latex)


class _FakeLLM:
    """Records calls and replays canned responses, one per attempt."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"messages": messages, **kwargs})
        reply = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


class TestDigestGeneration(unittest.IsolatedAsyncioTestCase):
    async def test_first_attempt_success_makes_one_call(self) -> None:
        llm = _FakeLLM([json.dumps(_FULL)])
        digest = await generate_lens_digest(llm, markdown="# Report\n\nSome analysis.")
        self.assertTrue(digest.available)
        self.assertEqual(len(llm.calls), 1)

    async def test_unparseable_json_is_repaired_on_the_second_attempt(self) -> None:
        llm = _FakeLLM(["{oops", json.dumps(_FULL)])
        digest = await generate_lens_digest(llm, markdown="# Report")
        self.assertTrue(digest.available)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("could not be parsed", llm.calls[1]["messages"][0]["content"])

    async def test_a_failing_llm_degrades_to_markdown_only(self) -> None:
        llm = _FakeLLM([RuntimeError("boom")])
        digest = await generate_lens_digest(llm, markdown="# Report")
        self.assertFalse(digest.available)
        self.assertEqual(len(llm.calls), 2)

    async def test_an_empty_report_never_calls_the_model(self) -> None:
        llm = _FakeLLM([json.dumps(_FULL)])
        digest = await generate_lens_digest(llm, markdown="   ")
        self.assertFalse(digest.available)
        self.assertEqual(llm.calls, [])

    async def test_the_report_is_the_only_input(self) -> None:
        """The digest indexes the report, so the prompt must carry it verbatim."""
        llm = _FakeLLM([json.dumps(_FULL)])
        await generate_lens_digest(llm, markdown="# Report\n\nMulti-head attention [p.4].")
        user_msg = llm.calls[0]["messages"][1]["content"]
        self.assertIn("Multi-head attention [p.4].", user_msg)
        self.assertNotIn("repair", system_prompt())


if __name__ == "__main__":
    unittest.main()
