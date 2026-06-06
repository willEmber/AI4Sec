from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch
from xml.etree import ElementTree as ET

from app.services import zotero_export

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "z": "http://www.zotero.org/namespaces/export#",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "link": "http://purl.org/rss/1.0/modules/link/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
}


def _identifiers(element) -> list[str]:
    """Direct dc:identifier text values of an element (plain "DOI ..."/"ISSN ...")."""
    return [n.text for n in element.findall("dc:identifier", NS) if n.text]


class BuildRdfTest(unittest.TestCase):
    def test_full_item_is_well_formed_and_complete(self):
        rdf = zotero_export.build_rdf(
            title="Attention Is All You Need",
            authors=[("Vaswani", "Ashish"), ("Shazeer", "Noam")],
            year=2017,
            venue="NeurIPS",
            doi="10.5555/3295222.3295349",
            abstract="The dominant sequence transduction models...",
            note_html="<h1>Report</h1><p>Body &amp; more</p>",
            pdf_rel_path="files/2/paper.pdf",
            volume="30",
            issue="2",
            pages="5998-6008",
            issn="1049-5258",
            url="https://doi.org/10.5555/3295222.3295349",
        )
        # Parses as XML (catches escaping / structural errors).
        root = ET.fromstring(rdf)

        article = root.find("bib:Article", NS)
        self.assertIsNotNone(article)
        self.assertEqual(article.find("z:itemType", NS).text, "journalArticle")
        self.assertEqual(article.find("dc:title", NS).text, "Attention Is All You Need")
        self.assertEqual(article.find("dc:date", NS).text, "2017")
        self.assertEqual(article.find("prism:volume", NS).text, "30")
        self.assertEqual(article.find("prism:number", NS).text, "2")
        self.assertEqual(article.find("bib:pages", NS).text, "5998-6008")
        self.assertEqual(
            article.find("dcterms:abstract", NS).text,
            "The dominant sequence transduction models...",
        )

        # Authors -> bib:authors / rdf:Seq / foaf:Person
        people = article.findall("bib:authors/rdf:Seq/rdf:li/foaf:Person", NS)
        self.assertEqual(len(people), 2)
        self.assertEqual(people[0].find("foaf:surname", NS).text, "Vaswani")
        self.assertEqual(people[0].find("foaf:givenName", NS).text, "Ashish")

        # DOI lives on the *item* (Zotero maps an item-level "DOI ..." identifier
        # to the DOI field); ISSN lives on the Journal container.
        self.assertIn("DOI 10.5555/3295222.3295349", _identifiers(article))
        journal = article.find("dcterms:isPartOf/bib:Journal", NS)
        self.assertEqual(journal.find("dc:title", NS).text, "NeurIPS")
        self.assertIn("ISSN 1049-5258", _identifiers(journal))
        # DOI must NOT be on the container.
        self.assertNotIn("DOI 10.5555/3295222.3295349", _identifiers(journal))

        # URL is exported via the dcterms:URI form.
        uri = article.find("dc:identifier/dcterms:URI/rdf:value", NS)
        self.assertEqual(uri.text, "https://doi.org/10.5555/3295222.3295349")

        # Note linked via dcterms:isReferencedBy -> bib:Memo with escaped HTML.
        ref = article.find("dcterms:isReferencedBy", NS)
        self.assertEqual(ref.get(f"{{{NS['rdf']}}}resource"), "#item_3")
        memo = root.find("bib:Memo", NS)
        self.assertIsNotNone(memo)
        self.assertEqual(memo.find("z:itemType", NS).text, "note")
        # rdf:value holds the HTML as text (so tags are escaped, not parsed).
        self.assertIn("<h1>Report</h1>", memo.find("rdf:value", NS).text)

        # Attachment linked via link:link -> z:Attachment with relative z:path.
        link = article.find("link:link", NS)
        self.assertEqual(link.get(f"{{{NS['rdf']}}}resource"), "#item_2")
        attach = root.find("z:Attachment", NS)
        self.assertEqual(attach.find("z:path", NS).text, "files/2/paper.pdf")
        self.assertEqual(attach.find("link:type", NS).text, "application/pdf")

    def test_minimal_item_omits_optional_nodes(self):
        rdf = zotero_export.build_rdf(
            title="No DOI No PDF",
            authors=[],
            year=0,
            venue="",
            doi="",
            abstract="",
            note_html="",
            pdf_rel_path=None,
        )
        root = ET.fromstring(rdf)
        self.assertIsNone(root.find("z:Attachment", NS))
        self.assertIsNone(root.find("bib:Memo", NS))
        self.assertIsNone(root.find("bib:Article/link:link", NS))
        self.assertIsNone(root.find("bib:Article/bib:authors", NS))
        self.assertEqual(root.find("bib:Article/dc:title", NS).text, "No DOI No PDF")


class ParseMetaTest(unittest.TestCase):
    def test_parse_crossref_extracts_structured_authors_and_fields(self):
        msg = {
            "title": ["Some Paper"],
            "author": [
                {"family": "Doe", "given": "Jane"},
                {"family": "Smith", "given": "John"},
                {"name": "The Collaboration"},  # corporate author, no family/given
            ],
            "container-title": ["Journal of Things"],
            "issued": {"date-parts": [[2020, 5]]},
            "volume": "12",
            "issue": "3",
            "page": "100-110",
            "ISSN": ["1234-5678"],
            "DOI": "10.1/abc",
            "URL": "https://doi.org/10.1/abc",
            "abstract": "<jats:p>Hello &amp; world</jats:p>",
        }
        meta = zotero_export._parse_crossref(msg)
        self.assertEqual(
            meta["authors"], [("Doe", "Jane"), ("Smith", "John"), ("The Collaboration", "")]
        )
        self.assertEqual(meta["venue"], "Journal of Things")
        self.assertEqual(meta["year"], 2020)
        self.assertEqual(meta["volume"], "12")
        self.assertEqual(meta["issue"], "3")
        self.assertEqual(meta["pages"], "100-110")
        self.assertEqual(meta["issn"], "1234-5678")
        self.assertEqual(meta["doi"], "10.1/abc")
        self.assertEqual(meta["abstract"], "Hello & world")  # tags stripped, unescaped

    def test_parse_openalex_splits_names_and_reconstructs_abstract(self):
        work = {
            "title": "Deep Thing",
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
                {"author": {"display_name": "Plato"}},
            ],
            "publication_year": 2019,
            "primary_location": {
                "source": {"display_name": "ICLR", "issn_l": "9999-0000"},
                "landing_page_url": "https://example.org/x",
            },
            "biblio": {"volume": "7", "issue": "1", "first_page": "1", "last_page": "9"},
            "abstract_inverted_index": {"Big": [0], "model": [1]},
            "ids": {"doi": "https://doi.org/10.2/xyz"},
        }
        meta = zotero_export._parse_openalex(work)
        self.assertEqual(meta["authors"], [("Vaswani", "Ashish"), ("Plato", "")])
        self.assertEqual(meta["venue"], "ICLR")
        self.assertEqual(meta["issn"], "9999-0000")
        self.assertEqual(meta["year"], 2019)
        self.assertEqual(meta["pages"], "1-9")
        self.assertEqual(meta["doi"], "10.2/xyz")
        self.assertEqual(meta["abstract"], "Big model")

    def test_merge_prefers_first_authors_and_fills_gaps(self):
        base = zotero_export._meta_skeleton()
        base["authors"] = [("A", "B")]
        base["venue"] = "X"
        extra = {
            "authors": [("C", "D")],  # ignored: base already has authors
            "venue": "Y",             # ignored: base already has venue
            "abstract": "filled",     # taken: base empty
            "year": 2022,
        }
        zotero_export._merge_meta(base, extra)
        self.assertEqual(base["authors"], [("A", "B")])
        self.assertEqual(base["venue"], "X")
        self.assertEqual(base["abstract"], "filled")
        self.assertEqual(base["year"], 2022)


class RenderNoteHtmlTest(unittest.TestCase):
    def test_first_line_becomes_h1_title(self):
        html_out = zotero_export._render_note_html(
            "## Some sub-heading\n\nbody", first_line="My Note Title"
        )
        self.assertTrue(html_out.startswith("<h1>My Note Title</h1>"))
        # The report's own heading stays below the title line.
        self.assertIn("Some sub-heading", html_out)

    def test_inline_and_block_math_become_zotero_math_nodes(self):
        md = "Inline $E=mc^2$ and a block:\n\n$$\\int_0^1 x\\,dx$$\n\ndone."
        html_out = zotero_export._render_note_html(md, first_line="T")
        self.assertIn('<span class="math">$E=mc^2$</span>', html_out)
        self.assertIn('<pre class="math">$$\\int_0^1 x\\,dx$$</pre>', html_out)
        # Block math is not left wrapped inside a <p> (invalid <p><pre>).
        self.assertNotIn("<p><pre", html_out.replace(" ", ""))

    def test_latex_delimiter_variants_supported(self):
        html_out = zotero_export._render_note_html(
            "see \\(a+b\\) and \\[c+d\\]", first_line="T"
        )
        self.assertIn('<span class="math">$a+b$</span>', html_out)
        self.assertIn('<pre class="math">$$c+d$$</pre>', html_out)

    def test_math_content_is_html_escaped(self):
        html_out = zotero_export._render_note_html("$a < b & c$", first_line="T")
        self.assertIn('<span class="math">$a &lt; b &amp; c$</span>', html_out)

    def test_dollar_inside_code_is_not_treated_as_math(self):
        html_out = zotero_export._render_note_html(
            "use `print($x)` here", first_line="T"
        )
        self.assertNotIn('class="math"', html_out)
        self.assertIn("print($x)", html_out)

    def test_note_first_line_variants(self):
        paper = {"title": "Some Paper"}
        self.assertEqual(
            zotero_export._note_first_line(paper, {"mode": "lens", "language": "en"}),
            "AI Reading Report · Logic Lens — Some Paper",
        )
        self.assertEqual(
            zotero_export._note_first_line(paper, {"mode": "snap", "language": "zh"}),
            "AI 阅读报告 · 快速洞察：Some Paper",
        )
        self.assertEqual(
            zotero_export._note_first_line(
                paper, {"mode": "qa", "language": "en", "user_question": "Why?"}
            ),
            "Smart Q&A: Why?",
        )


class BuildBundleTest(unittest.IsolatedAsyncioTestCase):
    async def test_zip_contains_rdf_and_pdf_with_matching_path(self):
        # Patch network enrichment so the test is deterministic and offline, and
        # so we can assert the looked-up authors land in the RDF.
        fake_meta = {
            "authors": [("Vaswani", "Ashish")],
            "abstract": "An abstract.",
            "venue": "",
            "year": 0,
            "volume": "30",
            "issue": "",
            "pages": "",
            "issn": "",
            "url": "",
            "doi": "",
        }
        with TemporaryDirectory() as tmp, patch.object(
            zotero_export, "fetch_item_meta", new_callable=AsyncMock, return_value=fake_meta
        ):
            pdf = Path(tmp) / "original.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake pdf bytes")

            paper = {
                "paper_id": "abc123def456",
                "title": "Test Paper",
                "doi": "",
                "venue": "Some Journal",
                "year": 2021,
            }
            zip_bytes, filename = await zotero_export.build_zotero_bundle(
                paper=paper,
                markdown_report="# Heading\n\nSome **bold** text with [p.3] citation.",
                pdf_path=pdf,
            )

            self.assertTrue(filename.endswith("_zotero.zip"))
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
            names = zf.namelist()
            rdf_name = next(n for n in names if n.endswith(".rdf"))
            pdf_name = next(n for n in names if n.endswith(".pdf"))

            # The PDF entry path must equal the z:path referenced in the RDF.
            rdf_text = zf.read(rdf_name).decode("utf-8")
            root = ET.fromstring(rdf_text)
            zpath = root.find("z:Attachment/z:path", NS).text
            self.assertEqual(zpath, pdf_name)
            self.assertEqual(zf.read(pdf_name), b"%PDF-1.4 fake pdf bytes")

            # Looked-up authors are present in the item.
            surname = root.find(
                "bib:Article/bib:authors/rdf:Seq/rdf:li/foaf:Person/foaf:surname", NS
            )
            self.assertEqual(surname.text, "Vaswani")

            # Report became a note; citation badge downgraded to plain text.
            memo_html = root.find("bib:Memo/rdf:value", NS).text
            self.assertIn("[p.3]", memo_html)

    async def test_no_pdf_still_produces_rdf_only_bundle(self):
        paper = {"paper_id": "x", "title": "T", "doi": "", "venue": "", "year": 0}
        with patch.object(
            zotero_export,
            "fetch_item_meta",
            new_callable=AsyncMock,
            return_value=zotero_export._meta_skeleton(),
        ):
            zip_bytes, _ = await zotero_export.build_zotero_bundle(
                paper=paper, markdown_report="hello", pdf_path=None
            )
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        self.assertEqual([n for n in zf.namelist() if n.endswith(".pdf")], [])
        self.assertTrue(any(n.endswith(".rdf") for n in zf.namelist()))


if __name__ == "__main__":
    unittest.main()
