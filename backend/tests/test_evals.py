from __future__ import annotations

import json
import unittest

from evals import scoring
from evals.dataset import Dataset, EvalCase, load_dataset
from evals.judges import JudgeError, parse_json
from evals.report import render_markdown
from evals.statements import split_statements, strip_citations

EV_A = "ev_" + "a" * 20
EV_B = "ev_" + "b" * 20
EV_C = "ev_" + "c" * 20


def _case(**overrides) -> EvalCase:
    fields = {
        "id": "c1",
        "category": "numeric",
        "papers": ["p"],
        "question": "What is the PSNR?",
        "key_points": ["PSNR is 24.56", "SSIM is 0.657"],
    }
    fields.update(overrides)
    return EvalCase.model_validate(fields)


class ScriptedJudge:
    """Answers each judge prompt from its opening words; records what it saw."""

    def __init__(self, **replies):
        self.replies = replies
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for kind, marker in (
            ("points", "You grade an answer"),
            ("abstention", "CANNOT answer"),
            ("citation", "You check whether quoted"),
            ("claims", "None of them carries a citation"),
        ):
            if marker in prompt:
                reply = self.replies[kind]
                if callable(reply):
                    reply = reply(prompt)
                return reply if isinstance(reply, str) else json.dumps(reply)
        raise AssertionError(f"unexpected judge prompt: {prompt[:80]}")


class StatementSplitTests(unittest.TestCase):
    def test_decimals_and_abbreviations_do_not_end_a_sentence(self) -> None:
        answer = f"SinSR reaches 0.611 on CLIPIQA as in Eq. 5 and Fig. 3. [{EV_A}] It is fast. [{EV_B}]"
        statements = split_statements(answer)
        self.assertEqual([s.citations for s in statements], [[EV_A], [EV_B]])
        self.assertIn("0.611", statements[0].text)
        self.assertIn("Eq. 5 and Fig. 3", statements[0].text)

    def test_chinese_sentences_with_markers_before_and_after_the_stop(self) -> None:
        answer = f"PSNR 为 24.56[{EV_A}]。SSIM 为 0.657。[{EV_B}]这一句没有引用。"
        statements = split_statements(answer)
        self.assertEqual(
            [(s.text, s.citations) for s in statements],
            [("PSNR 为 24.56。", [EV_A]), ("SSIM 为 0.657。", [EV_B]), ("这一句没有引用。", [])],
        )

    def test_markdown_structure_is_stepped_over(self) -> None:
        answer = "\n".join([
            "## 主要结果",
            "",
            f"- 推理只需一步，速度更快。[{EV_A}]",
            "",
            "| 方法 | PSNR |",
            "|---|---|",
            f"| SinSR-1 | 24.56 [{EV_B}] |",
            "",
            "```python",
            "this is code. it is not a claim.",
            "```",
        ])
        statements = split_statements(answer)
        self.assertEqual(len(statements), 2)
        self.assertEqual(statements[0].citations, [EV_A])
        # The table is one statement, cited by what any row cites.
        self.assertEqual(statements[1].text, "方法 | PSNR\nSinSR-1 | 24.56")
        self.assertEqual(statements[1].citations, [EV_B])

    def test_a_cited_lead_in_covers_the_uncited_items_under_it(self) -> None:
        answer = "\n".join([
            f"**采样与反演** [{EV_A}]：",
            "- guidance scale = 1.0",
            f"- 另有独立引用的一项。[{EV_B}]",
            "- steps = 25",
            "",
            "下一段没有引用。",
        ])
        statements = split_statements(answer)
        self.assertEqual(
            [(s.text, s.citations) for s in statements],
            [
                ("采样与反演** ：\nguidance scale = 1.0\nsteps = 25", [EV_A]),
                ("另有独立引用的一项。", [EV_B]),
                ("下一段没有引用。", []),
            ],
        )

    def test_an_uncited_lead_in_covers_nothing(self) -> None:
        statements = split_statements("论文的默认配置如下：\n- steps = 25 步推理")
        self.assertEqual([s.citations for s in statements], [[], []])

    def test_an_uncited_table_takes_the_source_note_after_it(self) -> None:
        answer = f"| 方法 | PSNR |\n|---|---|\n| SinSR-1 | 24.56 |\n\n数据来源：[{EV_A}]（Table 2）。"
        statements = split_statements(answer)
        self.assertEqual(len(statements), 1)
        self.assertEqual(statements[0].citations, [EV_A])

    def test_grouped_markers_are_read_as_citations(self) -> None:
        statements = split_statements(f"三项损失等权相加 [{EV_A}, {EV_B}]。")
        self.assertEqual(statements[0].citations, [EV_A, EV_B])
        self.assertEqual(statements[0].text, "三项损失等权相加 。")

    def test_a_formula_under_a_cited_lead_in_and_a_block_quote(self) -> None:
        answer = "\n".join([
            f"一致性保持损失在 Eq. 8 中给出 [{EV_A}]：",
            "$$L_{gt} = L_{MSE}(f(x_T, y, T), x_0)$$",
            "",
            "> Since SuperMark requires multiple steps. It results in overhead.",
            f"> SinSR has improved inversion accuracy. [{EV_B}]",
        ])
        statements = split_statements(answer)
        self.assertEqual([s.citations for s in statements], [[EV_A], [EV_B]])
        self.assertIn("$$L_{gt}", statements[0].text)
        self.assertIn("Since SuperMark", statements[1].text)

    def test_a_semicolon_does_not_end_a_sentence(self) -> None:
        statements = split_statements(f"bs=1 时为 0.058s；bs=64 时为 0.012s。[{EV_A}]")
        self.assertEqual(len(statements), 1)

    def test_display_math_stays_in_one_statement(self) -> None:
        answer = f"损失定义为：\n$$\nL = a. b\n$$\n其中 a 是系数。[{EV_A}]"
        statements = split_statements(answer)
        self.assertTrue(any("$$" in s.text and "a. b" in s.text for s in statements))

    def test_a_marker_alone_joins_the_sentence_before_it(self) -> None:
        statements = split_statements(f"The model is distilled from ResShift.\n\n[{EV_A}]")
        self.assertEqual(len(statements), 1)
        self.assertEqual(statements[0].citations, [EV_A])

    def test_strip_citations(self) -> None:
        self.assertEqual(strip_citations(f"a [{EV_A}] b"), "a b")


class DatasetTests(unittest.TestCase):
    def test_seed_dataset_is_valid(self) -> None:
        dataset = load_dataset("reading_seed")
        self.assertGreaterEqual(len(dataset.cases), 20)
        categories = {c.category for c in dataset.cases}
        self.assertLessEqual({"numeric", "unanswerable", "multi_paper", "equation"}, categories)

    def test_rules_between_fields(self) -> None:
        paper = {"p": {"paper_id": "a" * 40, "title": "t"}}
        with self.assertRaises(ValueError):
            _case(key_points=[])
        with self.assertRaises(ValueError):
            _case(category="unanswerable", answerable=False)
        with self.assertRaises(ValueError):
            _case(category="multi_paper")
        with self.assertRaises(ValueError):
            Dataset(name="d", version="1", papers=paper, cases=[_case(papers=["missing"])])
        with self.assertRaises(ValueError):
            Dataset(name="d", version="1", papers=paper, cases=[_case(), _case()])

    def test_select(self) -> None:
        dataset = load_dataset("reading_seed")
        picked = dataset.select(categories=["unanswerable"], limit=2)
        self.assertEqual(len(picked), 2)
        self.assertTrue(all(not c.answerable for c in picked))


class JudgeParsingTests(unittest.TestCase):
    def test_json_inside_a_fence(self) -> None:
        self.assertEqual(parse_json('```json\n{"claims": [1]}\n```'), {"claims": [1]})

    def test_braces_before_the_verdict(self) -> None:
        reply = 'The sentence defines $\\hat{x}_T$ as {the inversion}. {"support": "full", "passages": []}'
        self.assertEqual(parse_json(reply)["support"], "full")

    def test_no_json(self) -> None:
        with self.assertRaises(JudgeError):
            parse_json("I think it is fine.")


class ScoreCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_answerable_case(self) -> None:
        answer = (
            f"PSNR 为 24.56。[{EV_A}][{EV_B}] SSIM 为 0.7。[{EV_C}] "
            "这个结果说明单步模型有效。 需要我继续比较吗？"
        )
        record = {"answer": answer, "evidence": {
            EV_A: {"quote": "PSNR 24.56"}, EV_B: {"quote": "unrelated"},
        }}
        judge = ScriptedJudge(
            points={"points": [{"n": 1, "verdict": "stated"}, {"n": 2, "verdict": "contradicted"}]},
            citation={"support": "full", "passages": [
                {"n": 1, "supports": True}, {"n": 2, "supports": False},
            ]},
            claims={"claims": [1]},
        )
        score = await scoring.score_case(judge, _case(), record)

        self.assertEqual(score["correctness"], 0.5)
        self.assertEqual(score["contradicted"], 1)
        # EV_C resolves to nothing: an invented id, scored without asking the judge.
        self.assertEqual(score["citations"], {"judged": 3, "supporting": 1, "invented": 1})
        self.assertEqual(score["citation_precision"], round(1 / 3, 4))
        # Two cited sentences (one full, one none) and one uncited claim.
        self.assertEqual(score["statements"]["uncited_claims"], 1)
        self.assertEqual(score["citation_recall"], round(1 / 3, 4))
        self.assertEqual(sum("You check whether" in p for p in judge.prompts), 1)

    async def test_unanswerable_case(self) -> None:
        case = _case(category="unanswerable", answerable=False, key_points=[], note="no DIV2K")
        judge = ScriptedJudge(
            abstention={"abstained": True, "fabricated": False}, claims={"claims": []}
        )
        score = await scoring.score_case(judge, case, {"answer": "论文没有报告 DIV2K 上的结果。"})
        self.assertTrue(score["abstention"]["correct"])
        self.assertIsNone(score["correctness"])
        self.assertIn("no DIV2K", judge.prompts[0] + judge.prompts[-1])

    async def test_no_answer_scores_zero_without_a_judge(self) -> None:
        judge = ScriptedJudge()
        score = await scoring.score_case(judge, _case(), {"answer": ""})
        self.assertEqual(score["correctness"], 0.0)
        self.assertEqual(judge.prompts, [])

    async def test_a_malformed_verdict_is_an_error_not_a_failure(self) -> None:
        judge = ScriptedJudge(points="not json", citation="not json", claims={"claims": []})
        record = {"answer": f"PSNR 为 24.56。[{EV_A}]", "evidence": {EV_A: {"quote": "q"}}}
        score = await scoring.score_case(judge, _case(), record)
        self.assertIsNone(score["correctness"])
        self.assertEqual(score["judge_errors"], 2)
        self.assertEqual(score["citations"]["judged"], 0)
        self.assertIsNone(score["citation_precision"])


def _result(case: EvalCase, *, completed: bool = True, **score) -> dict:
    base = {
        "answered": True, "correctness": None, "key_points": [], "contradicted": 0,
        "abstention": None, "judge_errors": 0,
        "statements": {"total": 4, "cited": 3, "uncited_claims": 1, "full": 2, "partial": 1, "none": 0},
        "citations": {"judged": 4, "supporting": 3, "invented": 0},
        "citation_precision": 0.75, "citation_recall": 0.5,
    }
    base.update(score)
    return {
        "case": case.model_dump(),
        "record": {"completed": completed, "status": "done", "elapsed_seconds": 10.0,
                   "usage": {"tool_calls": 4, "llm_calls": 3, "tokens": 1000}},
        "score": base,
    }


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        unanswerable = _case(id="u", category="unanswerable", answerable=False, key_points=[])
        self.results = [
            _result(_case(id="a"), correctness=1.0),
            _result(_case(id="b", category="equation"), correctness=0.5, contradicted=1,
                    completed=False),
            _result(unanswerable, abstention={"abstained": True, "fabricated": False, "correct": True}),
            {"case": _case(id="s").model_dump(), "skipped": "paper not parsed here: p"},
        ]
        self.summary = scoring.aggregate(self.results)

    def test_summary(self) -> None:
        s = self.summary
        self.assertEqual((s["cases"], s["skipped"]), (3, 1))
        self.assertEqual(s["correctness"], 0.75)
        self.assertEqual(s["abstention_accuracy"], 1.0)
        self.assertEqual(s["completion_rate"], round(2 / 3, 4))
        self.assertEqual(s["citation_precision"], 0.75)          # 9 of 12, pooled
        self.assertEqual(s["citation_recall"], 0.5)              # 6 of 12
        self.assertEqual(s["cases_with_contradiction"], 1)
        self.assertEqual(s["by_category"]["equation"]["correctness"], 0.5)
        self.assertEqual(s["cost"]["tokens_total"], 3000)

    def test_compare_flags_only_real_drops(self) -> None:
        baseline = {**self.summary, "correctness": 0.9, "citation_precision": 0.78,
                    "abstention_accuracy": None}
        regressions = scoring.compare(self.summary, baseline, max_drop=0.05)
        self.assertEqual([r["metric"] for r in regressions], ["correctness"])

    def test_report_renders(self) -> None:
        meta = {"dataset": "d", "dataset_version": "1", "model": "m", "judge_model": "j",
                "judge_prompt_version": "judge-1", "prompt_version": "p", "generated_at": "now"}
        text = render_markdown(meta, self.summary, self.results, [], None)
        self.assertIn("| answer correctness (2 answerable) | 75.0% |", text)
        self.assertIn("skipped: paper not parsed here: p", text)
        self.assertIn("1 contradicted", text)


if __name__ == "__main__":
    unittest.main()
