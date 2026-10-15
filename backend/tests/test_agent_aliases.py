"""Evidence aliases (P15): the model cites `[E12]`, everything else keeps `ev_…`.

Pure rewriting first (no database), then the alias table and whole turns.
"""

from __future__ import annotations

import json
import random
import unittest
from types import SimpleNamespace
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from app.agents.aliases import (
    AliasBook,
    CitationRewriter,
    EvidenceAliasMiddleware,
    rewrite_tool_content,
    strip_alias_markers,
)
from app.agents.context import AgentContext, BudgetUsage
from tests.test_agent_guard import GuardTestCase, StepModel

A = "ev_" + "a" * 20
B = "ev_" + "b" * 20
C = "ev_" + "c" * 20
OLD = "ev_" + "0" * 20   # a stored id from before aliases, not in the book


def _book() -> AliasBook:
    book = AliasBook()
    book._take({A: 1, B: 2, C: 12})
    return book


class CitationRewriterTests(unittest.TestCase):
    def rewrite(self, text: str, usage: BudgetUsage | None = None) -> str:
        return CitationRewriter(_book(), usage=usage).rewrite(text)

    def test_aliases_become_stored_ids(self) -> None:
        self.assertEqual(self.rewrite("x [E1]."), f"x [{A}].")
        self.assertEqual(self.rewrite("x [E1][E12]"), f"x [{A}][{C}]")
        self.assertEqual(self.rewrite("x [e2]"), f"x [{B}]")
        self.assertEqual(self.rewrite("x [ E12 ]"), f"x [{C}]")

    def test_a_group_is_split_into_one_bracket_each(self) -> None:
        usage = BudgetUsage()
        self.assertEqual(self.rewrite("x [E1, E2]", usage), f"x [{A}][{B}]")
        self.assertEqual(self.rewrite("x 【E1、E12】", usage), f"x [{A}][{C}]")
        self.assertEqual(self.rewrite("x [E1; E2 E12]", usage), f"x [{A}][{B}][{C}]")
        self.assertEqual(usage.citation_groups_expanded, 3)

    def test_other_brackets_pass_through(self) -> None:
        for text in ("see [12]", "the [E-step]", "in [Eq. 3]", "[E1 is great]", "[link](http://x/E1)",
                     "[E]", "an [ev_short] id", "[1, 2]"):
            self.assertEqual(self.rewrite(text), text, text)

    def test_an_unknown_alias_is_dropped_and_counted(self) -> None:
        usage = BudgetUsage()
        self.assertEqual(self.rewrite("x [E99].", usage), "x .")
        self.assertEqual(self.rewrite("x [E1, E99]", usage), f"x [{A}]")
        self.assertEqual(usage.citations_dropped, 2)

    def test_stored_ids_are_kept_for_the_older_citation_check(self) -> None:
        self.assertEqual(self.rewrite(f"x [{A}]"), f"x [{A}]")
        self.assertEqual(self.rewrite(f"x [{OLD}]"), f"x [{OLD}]")
        self.assertEqual(self.rewrite(f"x [E2, {OLD}]"), f"x [{B}][{OLD}]")

    def test_an_unclosed_bracket_is_released_at_the_end(self) -> None:
        rewriter = CitationRewriter(_book())
        self.assertEqual(rewriter.feed("x [E1"), "x ")
        self.assertEqual(rewriter.finish(), "[E1")

    def test_a_long_open_bracket_is_not_held(self) -> None:
        rewriter = CitationRewriter(_book())
        text = "x [" + ", ".join(["E1"] * 80)
        self.assertEqual(rewriter.feed(text), text)

    def test_any_split_streams_the_same_text_as_a_whole_rewrite(self) -> None:
        text = (
            f"Routing [E1] beats dense [E1, E2]. 【E12、E2】 and [12] or [E-step]; "
            f"[E99] gone, [{OLD}] kept, [E1 not a group] and [Eq. 2]. End [E12]"
            f" [{A}, E2] [ E1 ] [E1][E2][E12]. [link](u) [E"
        )
        expected = CitationRewriter(_book()).rewrite(text)
        rng = random.Random(7)
        for _ in range(1000):
            cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, 30)))
            pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)])]
            rewriter = CitationRewriter(_book())
            out = "".join(rewriter.feed(p) for p in pieces) + rewriter.finish()
            self.assertEqual(out, expected, pieces)

    def test_previews_lose_their_markers(self) -> None:
        self.assertEqual(strip_alias_markers("as said [E3, E4] before"), "as said  before")


class ToolContentTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_stored_id_in_a_result_becomes_its_alias(self) -> None:
        book = AliasBook()
        content = json.dumps({
            "status": "ok",
            "evidence_ids": [C, A],
            "data": {"hits": [{"evidence_id": A}, {"evidence_id": C, "metadata_evidence_id": B}]},
        })
        await book.assign([C, A, B])
        out = rewrite_tool_content(content, book)
        parsed = json.loads(out)
        self.assertEqual(parsed["evidence_ids"], ["E1", "E2"])   # numbered in the result's order
        self.assertEqual(parsed["data"]["hits"][1], {"evidence_id": "E1", "metadata_evidence_id": "E3"})
        self.assertNotIn("ev_", out)

    async def test_a_stored_id_the_result_did_not_offer_is_removed(self) -> None:
        book = AliasBook()
        await book.assign([A])
        content = json.dumps({"answer_excerpt": f"It is 3 [{A}] or 4 [{OLD}].", "x": OLD})
        parsed = json.loads(rewrite_tool_content(content, book))
        self.assertEqual(parsed["answer_excerpt"], "It is 3 [E1] or 4 .")
        self.assertEqual(parsed["x"], "")

    async def test_the_middleware_rewrites_what_the_tool_returned(self) -> None:
        book = AliasBook()
        context = AgentContext(owner_id="o", session_id="s", aliases=book)
        request = SimpleNamespace(runtime=SimpleNamespace(context=context))
        content = json.dumps({"status": "ok", "evidence_ids": [B, A], "data": {"x": [A, B]}})

        async def handler(_request: Any) -> ToolMessage:
            return ToolMessage(content=content, name="t", tool_call_id="c1")

        message = await EvidenceAliasMiddleware().awrap_tool_call(request, handler)
        self.assertEqual(json.loads(message.content)["data"]["x"], ["E2", "E1"])
        self.assertEqual(book.resolve("E1"), B)

    async def test_without_a_book_nothing_changes(self) -> None:
        request = SimpleNamespace(runtime=SimpleNamespace(context=None))

        async def handler(_request: Any) -> ToolMessage:
            return ToolMessage(content=json.dumps({"evidence_ids": [A]}), name="t", tool_call_id="c1")

        message = await EvidenceAliasMiddleware().awrap_tool_call(request, handler)
        self.assertIn(A, message.content)


# ── With a database ─────────────────────────────────────────────────────────


@tool
async def passages_tool(n: int, runtime: ToolRuntime[AgentContext]) -> str:
    """Records `n` passages of paper1 as evidence and returns them."""
    from app.models.agent_models import ToolResult
    from app.models.evidence_models import Locator
    from app.services import evidence_service

    ctx = runtime.context
    ids = await evidence_service.record_fulltext_evidence_many(
        paper_id="paper1",
        passages=[(f"Passage {i} on sparse routing.", Locator(page_index=i)) for i in range(n)],
        owner_id=ctx.owner_id,
        session_id=ctx.session_id,
    )
    hits = [{"evidence_id": e, "text": f"Passage {i}"} for i, e in enumerate(ids)]
    return ToolResult.ok({"hits": hits}, evidence_ids=ids).to_json()


class ChunkedStepModel(StepModel):
    """`StepModel` that streams a `("chunks", [piece, ...])` step piece by piece."""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        idx = self.state.get("step", 0)
        step = self.steps[idx] if idx < len(self.steps) else ("text", "done")
        if self.tools_bound and step[0] == "chunks":
            self.state["step"] = idx + 1
            self.state.setdefault("calls", []).append({"tools_bound": True, "messages": list(messages)})
            for piece in step[1]:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece, id="m-chunks"))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(piece, chunk=chunk)
                yield chunk
            return
        message = self._generate(messages).generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": n}
                    for n, c in enumerate(message.tool_calls)
                ],
            )
        )


class AliasTableTests(GuardTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._runner.ALL_AGENT_TOOLS = [*self._runner.ALL_AGENT_TOOLS, passages_tool]

    async def test_numbers_are_stable_and_follow_the_order_given(self) -> None:
        from app.db import agent_repository as repo

        sid = self.session.session_id
        self.assertEqual(await repo.allocate_evidence_aliases(sid, [B, A]), {B: 1, A: 2})
        self.assertEqual(await repo.allocate_evidence_aliases(sid, [C, A]), {C: 3, A: 2})
        self.assertEqual(await repo.load_evidence_aliases(sid), {B: 1, A: 2, C: 3})
        other = await repo.create_session(owner_id=self.owner, language="en")
        self.assertEqual(await repo.allocate_evidence_aliases(other.session_id, [C]), {C: 1})

    async def test_concurrent_assignments_never_share_a_number(self) -> None:
        import asyncio

        book = await AliasBook.load(self.session.session_id)
        ids = [f"ev_{i:020x}" for i in range(30)]
        await asyncio.gather(*(book.assign(ids[i::3]) for i in range(3)))
        stored = await AliasBook.load(self.session.session_id)
        self.assertEqual(sorted(stored.by_no), list(range(1, 31)))
        self.assertEqual(stored.by_id, book.by_id)

    async def test_deleting_the_session_deletes_its_aliases(self) -> None:
        from app.db import agent_repository as repo
        from app.services import data_lifecycle

        await repo.allocate_evidence_aliases(self.session.session_id, [A])
        await data_lifecycle.delete_session(self.session.session_id, owner_id=self.owner)
        self.assertEqual(await repo.load_evidence_aliases(self.session.session_id), {})

    async def test_an_aliased_id_is_citable_in_its_session(self) -> None:
        from app.db import agent_repository as repo

        # Recall brings evidence recorded under another conversation.
        await repo.allocate_evidence_aliases(self.session.session_id, [A])
        self.assertEqual(await repo.citable_in_session(self.session.session_id, [A, B]), {A})

    async def _streamed(self, run) -> str:
        return "".join(e.payload.get("text", "") for e in await self._events(run, "message.delta"))

    async def test_a_turn_cites_by_alias_and_stores_stored_ids(self) -> None:
        model = ChunkedStepModel(
            steps=[
                ("tools", [("passages_tool", {"n": 3})]),
                ("chunks", ["Routing helps [E", "1][E", "3] and [E2, E", "9].", " Done [E"]),
            ],
            state={},
        )
        self._runner.build_chat_model = lambda *_a, **_k: model
        run = await self._new_run()
        await self._runner._execute_turn(session=self.session, run=run, question="What?")
        self.assertEqual((await self._finished(run)).status.value, "done")

        book = await AliasBook.load(self.session.session_id)
        e1, e2, e3 = (book.resolve(f"E{i}") for i in (1, 2, 3))
        expected = f"Routing helps [{e1}][{e3}] and [{e2}]. Done [E"
        answer = await self._answer(run)
        self.assertEqual(answer.content, expected)
        self.assertEqual(await self._streamed(run), expected)
        self.assertEqual(answer.citations, [e1, e2, e3])

        # The model was shown aliases only.
        seen = "\n".join(str(m.content) for m in model.state["calls"][1]["messages"])
        self.assertIn('"E1"', seen)
        self.assertNotIn("ev_", seen)
        usage = (await self._finished(run)).usage
        self.assertEqual(usage["citations_dropped"], 1)
        self.assertEqual(usage["citation_groups_expanded"], 1)

    async def test_a_follow_up_cites_an_earlier_turn_by_its_alias(self) -> None:
        await self._turn([("tools", [("passages_tool", {"n": 2})]), ("text", "Read.")])
        run, _ = await self._turn([("text", "As read before [E2].")])
        book = await AliasBook.load(self.session.session_id)
        answer = await self._answer(run)
        self.assertEqual(answer.content, f"As read before [{book.resolve('E2')}].")
        self.assertEqual(answer.citations, [book.resolve("E2")])

    async def test_a_stored_id_still_cites_in_an_older_conversation(self) -> None:
        from app.agents.tools import search_paper_content

        # Read before aliases existed: recorded under the session, never aliased.
        earlier = (await self._call(search_paper_content, question="BLEU"))["evidence_ids"][0]
        run, _ = await self._turn([("text", f"29.7 BLEU [{earlier}].")])
        self.assertEqual((await self._answer(run)).citations, [earlier])

    async def test_a_stopped_turn_keeps_its_rewritten_text(self) -> None:
        import asyncio

        self._model([
            ("tools", [("passages_tool", {"n": 1})]),
            ("tools", [("slow_tool", {"seconds": 30})], "So far [E1]. "),
        ])
        run = await self._new_run()
        task = asyncio.create_task(
            self._runner._execute_turn(session=self.session, run=run, question="What?")
        )
        for _ in range(100):
            await asyncio.sleep(0.05)
            if [e for e in await self._events(run, "tool.started") if e.payload.get("tool") == "slow_tool"]:
                break
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        book = await AliasBook.load(self.session.session_id)
        answer = await self._answer(run)
        self.assertEqual(answer.content, f"So far [{book.resolve('E1')}].")


if __name__ == "__main__":
    unittest.main()
