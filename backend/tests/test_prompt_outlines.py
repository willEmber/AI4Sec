"""The parsed papers' outlines in the system prompt (`AGENT_PROMPT_OUTLINE_PAPERS`).

Nearly every reading turn spent its first step fetching the outline, so it
could not read a section before its second; with the outline in the prompt
the first step can.
"""

from __future__ import annotations

import os
import unittest

from app.agents.prompts import (
    PROMPT_OUTLINE_MAX_SECTIONS,
    build_system_prompt,
    current_prompt_version,
)
from app.config import get_settings
from app.models.agent_models import Availability, SessionPaper
from tests.test_agent_guard import GuardTestCase

OUTLINE = [
    {"section_id": "paper1:s1", "title": "1 Introduction", "depth": 1, "page": 1, "page_end": 2},
    {"section_id": "paper1:s2", "title": "2.1 Setup", "depth": 2, "page": 3},
]
NOTE_EN = "pass its `section_id` straight to `read_paper_section`"


def _paper(availability: Availability = Availability.PARSED) -> SessionPaper:
    return SessionPaper(
        session_id="as_x",
        literature_id="li_1",
        paper_id="paper1",
        availability=availability,
        title="SparseMoE Routing",
    )


class PromptOutlineFormatTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("AGENT_PROMPT_OUTLINE_PAPERS", None)
        get_settings.cache_clear()

    def test_a_parsed_paper_is_listed_with_its_outline(self) -> None:
        prompt = build_system_prompt(language="en", papers=[_paper()], outlines={"paper1": OUTLINE})
        self.assertIn("parsed\n  Outline:\n  - `paper1:s1` 1 Introduction (p.1–2)", prompt)
        self.assertIn("\n    - `paper1:s2` 2.1 Setup (p.3)", prompt)
        self.assertIn(NOTE_EN, prompt)

        zh = build_system_prompt(language="zh", papers=[_paper()], outlines={"paper1": OUTLINE})
        self.assertIn("  目录：\n  - `paper1:s1`", zh)
        self.assertIn("第一步就读这几节", zh)

    def test_without_outlines_the_prompt_is_as_before(self) -> None:
        for language in ("zh", "en"):
            self.assertEqual(
                build_system_prompt(language=language, papers=[_paper()], outlines={}),
                build_system_prompt(language=language, papers=[_paper()]),
            )
        self.assertNotIn(NOTE_EN, build_system_prompt(language="en", papers=[_paper()]))

    def test_only_a_parsed_paper_shows_its_outline(self) -> None:
        prompt = build_system_prompt(
            language="en", papers=[_paper(Availability.PDF_READY)], outlines={"paper1": OUTLINE}
        )
        self.assertNotIn("paper1:s1", prompt)
        self.assertNotIn(NOTE_EN, prompt)

    def test_a_long_outline_is_cut_and_points_to_the_tool(self) -> None:
        entries = [
            {"section_id": f"paper1:s{i}", "title": f"Section {i}", "depth": 1, "page": 1}
            for i in range(1, PROMPT_OUTLINE_MAX_SECTIONS + 11)
        ]
        prompt = build_system_prompt(language="en", papers=[_paper()], outlines={"paper1": entries})
        self.assertIn(f"`paper1:s{PROMPT_OUTLINE_MAX_SECTIONS}`", prompt)
        self.assertNotIn(f"`paper1:s{PROMPT_OUTLINE_MAX_SECTIONS + 1}`", prompt)
        self.assertIn("… 10 more sections: `get_paper_outline` lists them", prompt)

    def test_the_version_names_the_outline_while_it_is_on(self) -> None:
        get_settings.cache_clear()
        self.assertEqual(current_prompt_version(frozenset()), "p18-venue-1+outline1")
        os.environ["AGENT_PROMPT_OUTLINE_PAPERS"] = "0"
        get_settings.cache_clear()
        self.assertEqual(current_prompt_version(frozenset()), "p18-venue-1")


class PromptOutlineTurnTests(GuardTestCase):
    async def asyncTearDown(self) -> None:
        os.environ.pop("AGENT_PROMPT_OUTLINE_PAPERS", None)
        await super().asyncTearDown()

    async def _system_prompt(self) -> str:
        _run, model = await self._turn([("text", "ok")])
        return str(model.state["calls"][0]["messages"][0].content)

    async def test_a_turn_sees_the_parsed_papers_outline(self) -> None:
        prompt = await self._system_prompt()
        self.assertIn("`paper1:s2` 2 Method (p.3–5)", prompt)
        self.assertIn(NOTE_EN, prompt)

    async def test_at_zero_a_turn_sees_no_outline(self) -> None:
        os.environ["AGENT_PROMPT_OUTLINE_PAPERS"] = "0"
        get_settings.cache_clear()
        prompt = await self._system_prompt()
        self.assertNotIn("paper1:s2", prompt)
        self.assertNotIn(NOTE_EN, prompt)

    async def test_a_paper_outline_that_cannot_load_is_left_out(self) -> None:
        from app.agents.tools import reading

        async def broken(_paper_id: str):
            raise RuntimeError("nodes table unavailable")

        saved = reading.load_section_nodes
        reading.load_section_nodes = broken
        try:
            prompt = await self._system_prompt()
        finally:
            reading.load_section_nodes = saved
        self.assertNotIn(NOTE_EN, prompt)
        self.assertIn("paper_id=paper1", prompt)


if __name__ == "__main__":
    unittest.main()
