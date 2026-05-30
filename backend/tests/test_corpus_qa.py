from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.services import corpus_qa


_RECORDS = [
    {
        "document_id": "d1",
        "document_name": "BadCLIP_CVPR_2024.md",
        "segment_id": "s1",
        "content": "BadCLIP is a trigger-aware prompt-learning backdoor attack on CLIP.",
        "score": 0.95,
    },
    {
        "document_id": "d2",
        "document_name": "BadT2I.md",
        "segment_id": "s2",
        "content": "BadT2I backdoors text-to-image diffusion via multimodal data poisoning.",
        "score": 0.81,
    },
]


class CorpusQaTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_builds_cited_context_and_sources(self) -> None:
        chat = AsyncMock(return_value="BadCLIP attacks CLIP [L1].")
        with patch(
            "app.services.dify_client.search_records",
            new=AsyncMock(return_value=_RECORDS),
        ), patch("app.services.corpus_qa.get_llm_service") as gls:
            gls.return_value.chat = chat
            result = await corpus_qa.answer_corpus_question(
                "What is BadCLIP?", search_method="full_text_search", language="en"
            )

        self.assertEqual(result["markdown"], "BadCLIP attacks CLIP [L1].")
        self.assertEqual(result["blocks_used"], 2)
        self.assertEqual([s["idx"] for s in result["sources"]], [1, 2])
        self.assertEqual(result["sources"][0]["document_id"], "d1")
        self.assertEqual(result["sources"][0]["segment_id"], "s1")

        # The context handed to the LLM is numbered [L1]/[L2] with source names.
        messages = chat.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        self.assertIn("[L1]", user_content)
        self.assertIn("[L2]", user_content)
        self.assertIn("BadCLIP_CVPR_2024.md", user_content)
        # English prompt selected.
        self.assertEqual(messages[0]["content"], corpus_qa._SYSTEM_PROMPT_EN)

    async def test_no_records_skips_llm(self) -> None:
        chat = AsyncMock(return_value="should not be called")
        with patch(
            "app.services.dify_client.search_records",
            new=AsyncMock(return_value=[]),
        ), patch("app.services.corpus_qa.get_llm_service") as gls:
            gls.return_value.chat = chat
            result = await corpus_qa.answer_corpus_question(
                "obscure question", search_method="full_text_search", language="en"
            )

        self.assertEqual(result["sources"], [])
        self.assertEqual(result["blocks_used"], 0)
        chat.assert_not_called()

    async def test_chinese_language_selects_zh_prompt(self) -> None:
        chat = AsyncMock(return_value="答案 [L1]。")
        with patch(
            "app.services.dify_client.search_records",
            new=AsyncMock(return_value=_RECORDS),
        ), patch("app.services.corpus_qa.get_llm_service") as gls:
            gls.return_value.chat = chat
            await corpus_qa.answer_corpus_question(
                "什么是 BadCLIP?", search_method="full_text_search", language="zh"
            )

        messages = chat.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], corpus_qa._SYSTEM_PROMPT_ZH)
        self.assertIn("问题:", messages[1]["content"])

    async def test_blank_passages_are_skipped(self) -> None:
        records = [
            {"document_id": "d1", "document_name": "A.md", "segment_id": "s1", "content": "   ", "score": 0.5},
            {"document_id": "d2", "document_name": "B.md", "segment_id": "s2", "content": "real content", "score": 0.4},
        ]
        chat = AsyncMock(return_value="ok")
        with patch(
            "app.services.dify_client.search_records",
            new=AsyncMock(return_value=records),
        ), patch("app.services.corpus_qa.get_llm_service") as gls:
            gls.return_value.chat = chat
            result = await corpus_qa.answer_corpus_question(
                "q", search_method="full_text_search", language="en"
            )

        # Only the non-blank passage becomes a source, renumbered to L1.
        self.assertEqual(result["blocks_used"], 1)
        self.assertEqual(result["sources"][0]["document_id"], "d2")
        self.assertEqual(result["sources"][0]["idx"], 1)


if __name__ == "__main__":
    unittest.main()
