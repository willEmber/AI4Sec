"""How the reading tools present a paper's sections, and other pure helpers.

Section levels and extents come from `agents/tools/sections.py`, which works on
what a parse stored: a flat tree (every heading at depth 1, as MinerU leaves
some papers) and a NeurIPS checklist appended to the paper are the cases the
multi-turn verification ran into.
"""

from __future__ import annotations

import unittest

from app.agents.tools.sections import (
    find_section,
    numbered_level,
    paper_of_short_id,
    section_views,
)
from app.services.qa_retrieval import PaperNode

PID = "2ff3b770d9b5ff870b050872337fab9945cdc1de"
OTHER = "8ed59ad70de5aa609694f237478754cc239debc7"


def _section(idx: int, title: str, start: int, end: int, *, depth: int = 1, pid: str = PID) -> PaperNode:
    slug = title.replace(" ", "_")
    return PaperNode(
        node_id=f"{pid}:section:{idx}:{slug}", paper_id=pid, parent_id=f"{pid}:paper",
        depth=depth, node_type="section", block_type="", sub_type="", title=title,
        title_path=title, page_start=start // 10, page_end=end // 10, block_start=start,
        block_end=end, text=title, text_for_search=title, order_idx=start,
    )


# A flat parse: every heading at depth 1, as stored for ROBIN.
FLAT = [
    _section(1, "Abstract", 0, 4),
    _section(2, "3 Methodology", 10, 10),
    _section(3, "3.1 Overview", 11, 20),
    _section(4, "3.2 Algorithm", 21, 30),
    _section(5, "4 Experiments", 31, 31),
    _section(6, "4.1 Setting", 32, 40),
    _section(7, "NeurIPS Paper Checklist", 50, 50),
    _section(8, "1. Claims", 51, 55),
    _section(9, "Answer: [Yes]", 56, 57),
    _section(10, "Guidelines:", 58, 60),
]


class SectionViewTests(unittest.TestCase):
    def test_numbering_gives_the_level(self) -> None:
        self.assertEqual(numbered_level("3 Methodology"), 1)
        self.assertEqual(numbered_level("3.1 Overview"), 2)
        self.assertEqual(numbered_level("3.1. Overview"), 2)
        self.assertEqual(numbered_level("A.1 Image watermark design"), 2)
        self.assertEqual(numbered_level("3.2.1 Scaled Dot-Product"), 3)
        self.assertEqual(numbered_level("Abstract"), 0)
        self.assertEqual(numbered_level("Answer: [Yes]"), 0)

    def test_a_flat_parse_gets_its_hierarchy_back(self) -> None:
        views = {v.title: v for v in section_views(FLAT)}
        self.assertEqual(views["3.1 Overview"].level, 2)
        # "3 Methodology" stored as its own heading only; read, it covers 3.1–3.2.
        self.assertEqual(views["3 Methodology"].block_end, 30)
        self.assertEqual(views["4 Experiments"].block_end, 40)
        self.assertEqual(views["Abstract"].block_end, 4)

    def test_a_parse_with_the_title_above_every_heading_nests_by_numbering(self) -> None:
        # SinSR as stored: the title at depth 1, every heading at depth 2, so
        # "4. Methodology" covered its own heading only and Lens read each
        # subsection on its own.
        titled = [
            _section(1, "SinSR: Diffusion-Based Image Super-Resolution in a Single Step", 0, 2),
            _section(2, "Abstract", 3, 5, depth=2),
            _section(3, "1. Introduction", 6, 9, depth=2),
            _section(4, "4. Methodology", 10, 10, depth=2),
            _section(5, "4.1. Deterministic Sampling", 11, 20, depth=2),
            _section(6, "4.2. Consistency Preserving Distillation", 21, 30, depth=2),
            _section(7, "5. Experiment", 31, 31, depth=2),
            _section(8, "5.3. Analysis", 32, 40, depth=2),
            _section(9, "Is a more sophisticated distillation strategy necessary?", 41, 45, depth=2),
            _section(10, "6. Conclusion", 46, 48, depth=2),
            _section(11, "Acknowledgements", 49, 50, depth=2),
            _section(12, "References", 51, 90, depth=2),
        ]
        views = {v.title: v for v in section_views(titled)}
        self.assertEqual((views["4. Methodology"].level, views["4.1. Deterministic Sampling"].level), (2, 3))
        self.assertEqual(views["4. Methodology"].block_end, 30)
        # An unnumbered heading inside 5.3 is part of it, and of 5.
        self.assertEqual(views["5.3. Analysis"].block_end, 45)
        self.assertEqual(views["5. Experiment"].block_end, 45)
        # Back matter is not folded into the conclusion.
        self.assertEqual(views["6. Conclusion"].block_end, 48)
        self.assertEqual(views["Acknowledgements"].level, 2)
        self.assertEqual(views["Abstract"].block_end, 5)

    def test_a_title_that_reads_as_appendix_a_does_not_set_the_base(self) -> None:
        views = {
            v.title: v
            for v in section_views([
                _section(1, "A Survey of Routing", 0, 2),
                _section(2, "2 Method", 10, 10, depth=2),
                _section(3, "2.1 Gates", 11, 20, depth=2),
            ])
        }
        self.assertEqual(views["2 Method"].block_end, 20)

    def test_a_checklist_is_folded_into_one_entry(self) -> None:
        views = section_views(FLAT)
        listed = [v.title for v in views if not v.folded]
        self.assertNotIn("1. Claims", listed)
        self.assertNotIn("Answer: [Yes]", listed)
        checklist = next(v for v in views if v.title == "NeurIPS Paper Checklist")
        self.assertEqual(checklist.folded_count, 3)
        self.assertEqual(checklist.block_end, 60)

    def test_an_appendix_after_a_checklist_is_not_folded(self) -> None:
        views = section_views([*FLAT, _section(11, "A Proofs", 70, 80), _section(12, "A.1 Lemma", 81, 90)])
        listed = [v.title for v in views if not v.folded]
        self.assertIn("A Proofs", listed)
        self.assertEqual(next(v for v in views if v.title == "A Proofs").block_end, 90)

    def test_short_ids_name_the_paper_and_the_section(self) -> None:
        views = section_views(FLAT)
        self.assertEqual(views[1].short_id, "2ff3b770:s2")

    def test_ids_resolve_generously_but_only_within_the_paper(self) -> None:
        views = section_views(FLAT)

        def by(**kwargs: str) -> str | None:
            found = find_section(views, **kwargs)
            return found.title if found else None

        self.assertEqual(by(section_id="2ff3b770:s3"), "3.1 Overview")
        self.assertEqual(by(section_id=f"{PID}:section:3:3.1_Overview"), "3.1 Overview")
        # The title part retyped with spaces, as the model did: the index decides.
        self.assertEqual(by(section_id=f"{PID}:section:4:3.2 algorithm design"), "3.2 Algorithm")
        self.assertEqual(by(section_title="Methodology"), "3 Methodology")
        self.assertEqual(by(section_id="3.1 Overview"), "3.1 Overview")
        # Another paper's id never lands on a section of this one.
        self.assertIsNone(find_section(views, section_id=f"{OTHER[:8]}:s3"))
        self.assertIsNone(find_section(views, section_id=f"{OTHER}:section:3:3.1_Overview"))
        self.assertIsNone(find_section(views, section_id="2ff3b770:s99"))

    def test_a_short_prefix_names_one_session_paper(self) -> None:
        self.assertEqual(paper_of_short_id("2ff3b770:s3", [PID, OTHER]), PID)
        self.assertEqual(paper_of_short_id(f"{OTHER}:section:2:x", [PID, OTHER]), OTHER)
        self.assertEqual(paper_of_short_id("deadbeef:s3", [PID, OTHER]), "")
        self.assertEqual(paper_of_short_id("3 Methodology", [PID]), "")


class SnapSignalFactTests(unittest.TestCase):
    def test_each_signal_is_one_sourced_fact(self) -> None:
        from app.agents.tools.modes import snap_signal_facts

        signals = {
            "venue": "Neural Information Processing Systems", "venue_normalized": "NeurIPS",
            "year": 2017, "ccf_rank": "A", "sci_rank": "", "cited_by_count": 196706,
            "citations_per_year": 19670.6, "influential_citation_count": 20951,
            "intents": {"background": 3, "methodology": 1, "result": 0, "sampled": 200},
            "repos": [], "is_open_access": True, "oa_url": "https://x.test/p.pdf",
            "is_retracted": False,
            "provenance": {"venue": "Crossref", "ccf_rank": "EasyScholar",
                           "citations": "Semantic Scholar", "open_access": "OpenAlex",
                           "intents": "Semantic Scholar"},
            "unavailable": [],
        }
        facts = dict(snap_signal_facts(signals, "Attention Is All You Need"))
        self.assertEqual(
            facts["Attention Is All You Need — cited by 196,706, 19,670.6 per year, 20,951 influential"],
            "Semantic Scholar",
        )
        self.assertEqual(facts["NeurIPS — CCF A"], "EasyScholar")
        self.assertNotIn("JCR", " ".join(facts))   # an empty rank is not a fact
        self.assertIn("Attention Is All You Need — not marked as retracted", facts)
        self.assertIn("Attention Is All You Need — no code repository link found in the paper", facts)
        self.assertTrue(any("published at Neural Information Processing Systems (NeurIPS), 2017" in f for f in facts))

    def test_what_the_report_could_not_obtain_is_left_out(self) -> None:
        from app.agents.tools.modes import snap_signal_facts

        facts = snap_signal_facts(
            {"cited_by_count": 5, "is_retracted": None, "unavailable": ["citations"]}, "P"
        )
        self.assertEqual(facts, [])


class ScriptTests(unittest.TestCase):
    def test_the_writing_a_text_is_mostly_in(self) -> None:
        from app.agents.tools.memory import script_of

        self.assertEqual(script_of("以后都用英文回答我。"), "han")
        self.assertEqual(script_of("用户在做扩散模型水印鲁棒性（robustness of watermarks）研究"), "han")
        self.assertEqual(script_of("User prefers all answers in English."), "latin")
        self.assertEqual(script_of("2024"), "")


if __name__ == "__main__":
    unittest.main()
