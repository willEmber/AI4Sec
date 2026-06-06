from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from app.services import report_images


def _make_paper_image(root: Path, paper_id: str, name: str, data: bytes = b"PNGDATA") -> Path:
    img_dir = root / "papers" / paper_id / "mineru" / "raw" / "abc123" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    f = img_dir / name
    f.write_bytes(data)
    return f


class FigureNameTest(unittest.TestCase):
    def test_internal_figure_ref_recognized(self):
        self.assertEqual(
            report_images._figure_name("/api/papers/abc/images/deadbeef.png"),
            "deadbeef.png",
        )
        self.assertEqual(
            report_images._figure_name("/api/papers/abc/images/x.JPG?v=2"), "x.JPG"
        )

    def test_non_figure_urls_ignored(self):
        self.assertIsNone(report_images._figure_name("https://example.org/cat.png"))
        self.assertIsNone(report_images._figure_name("data:image/png;base64,AAAA"))
        self.assertIsNone(
            report_images._figure_name("/api/papers/abc/images/../../etc/passwd")
        )


class ResolveImageFileTest(unittest.TestCase):
    def test_resolves_only_files_under_images_dir(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_paper_image(root, "p1", "fig1.png")
            found = report_images._resolve_image_file("p1", "fig1.png", root)
            self.assertIsNotNone(found)
            self.assertTrue(found.name == "fig1.png")
            # Unknown name / wrong paper -> None
            self.assertIsNone(report_images._resolve_image_file("p1", "nope.png", root))
            self.assertIsNone(report_images._resolve_image_file("p2", "fig1.png", root))


class ProcessReportMarkdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_to_r2_and_rewrites_link(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_paper_image(root, "p1", "fig1.png")
            md = "Look:\n\n![Figure 1](/api/papers/p1/images/fig1.png)\n\ndone."
            with patch(
                "app.services.r2_storage.is_enabled", return_value=True
            ), patch(
                "app.services.r2_storage.upload_file",
                new_callable=AsyncMock,
                return_value="https://res.example.top/scholar/p1/fig1.png",
            ) as up:
                out = await report_images.process_report_markdown(
                    md, paper_id="p1", data_dir=root
                )
            self.assertIn(
                "![Figure 1](https://res.example.top/scholar/p1/fig1.png)", out
            )
            up.assert_awaited_once()
            # Uploaded under scholar/{paper_id}/{name}
            self.assertEqual(up.await_args.args[0], "scholar/p1/fig1.png")

    async def test_embed_fallback_when_no_oss(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_paper_image(root, "p1", "fig1.png", data=b"\x89PNG\r\n")
            md = "![f](/api/papers/p1/images/fig1.png)"
            with patch("app.services.r2_storage.is_enabled", return_value=False):
                out = await report_images.process_report_markdown(
                    md, paper_id="p1", data_dir=root, embed_fallback=True
                )
            self.assertIn("![f](data:image/png;base64,", out)

    async def test_keep_original_when_no_oss_and_no_embed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_paper_image(root, "p1", "fig1.png")
            md = "![f](/api/papers/p1/images/fig1.png)"
            with patch("app.services.r2_storage.is_enabled", return_value=False):
                out = await report_images.process_report_markdown(
                    md, paper_id="p1", data_dir=root, embed_fallback=False
                )
            self.assertEqual(out, md)  # unchanged (original behaviour)

    async def test_unresolved_figure_dropped_when_requested(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)  # no image file on disk
            md = "before ![f](/api/papers/p1/images/missing.png) after"
            with patch("app.services.r2_storage.is_enabled", return_value=False):
                out = await report_images.process_report_markdown(
                    md, paper_id="p1", data_dir=root, drop_unresolved=True
                )
            self.assertNotIn("missing.png", out)
            self.assertIn("before  after", out)

    async def test_external_images_left_untouched(self):
        md = "![cat](https://example.org/cat.png) and ![d](data:image/png;base64,AA)"
        with TemporaryDirectory() as tmp:
            out = await report_images.process_report_markdown(
                md, paper_id="p1", data_dir=Path(tmp), embed_fallback=True
            )
        self.assertEqual(out, md)

    async def test_no_data_dir_is_noop(self):
        md = "![f](/api/papers/p1/images/fig1.png)"
        out = await report_images.process_report_markdown(
            md, paper_id="p1", data_dir=None
        )
        self.assertEqual(out, md)


if __name__ == "__main__":
    unittest.main()
