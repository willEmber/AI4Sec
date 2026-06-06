from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
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
}


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
        )
        # Parses as XML (catches escaping / structural errors).
        root = ET.fromstring(rdf)

        article = root.find("bib:Article", NS)
        self.assertIsNotNone(article)
        self.assertEqual(article.find("z:itemType", NS).text, "journalArticle")
        self.assertEqual(article.find("dc:title", NS).text, "Attention Is All You Need")
        self.assertEqual(article.find("dc:date", NS).text, "2017")
        self.assertEqual(
            article.find("dcterms:abstract", NS).text,
            "The dominant sequence transduction models...",
        )

        # Authors -> bib:authors / rdf:Seq / foaf:Person
        people = article.findall("bib:authors/rdf:Seq/rdf:li/foaf:Person", NS)
        self.assertEqual(len(people), 2)
        self.assertEqual(people[0].find("foaf:surname", NS).text, "Vaswani")
        self.assertEqual(people[0].find("foaf:givenName", NS).text, "Ashish")

        # Venue + DOI live on the Journal container.
        journal = article.find("dcterms:isPartOf/bib:Journal", NS)
        self.assertEqual(journal.find("dc:title", NS).text, "NeurIPS")
        self.assertEqual(
            journal.find("dc:identifier", NS).text, "DOI 10.5555/3295222.3295349"
        )

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
        self.assertEqual(root.find("bib:Article/dc:title", NS).text, "No DOI No PDF")


class BuildBundleTest(unittest.IsolatedAsyncioTestCase):
    async def test_zip_contains_rdf_and_pdf_with_matching_path(self):
        with TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "original.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake pdf bytes")

            paper = {
                "paper_id": "abc123def456",
                "title": "Test Paper",
                "doi": "",  # empty -> no Crossref network call (deterministic)
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

            # Report became a note; citation badge downgraded to plain text.
            memo_html = root.find("bib:Memo/rdf:value", NS).text
            self.assertIn("[p.3]", memo_html)

    async def test_no_pdf_still_produces_rdf_only_bundle(self):
        paper = {"paper_id": "x", "title": "T", "doi": "", "venue": "", "year": 0}
        zip_bytes, _ = await zotero_export.build_zotero_bundle(
            paper=paper, markdown_report="hello", pdf_path=None
        )
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        self.assertEqual([n for n in zf.namelist() if n.endswith(".pdf")], [])
        self.assertTrue(any(n.endswith(".rdf") for n in zf.namelist()))


if __name__ == "__main__":
    unittest.main()
