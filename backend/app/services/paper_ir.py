from __future__ import annotations

import json
import re
from pathlib import Path

from app.db import database as db
from app.models.paper_ir import Block, PaperIR, Section


def _find_content_list(output_dir: Path) -> Path:
    """Find content_list.json in MinerU output directory.

    MinerU API outputs vary by version:
    - Exact: content_list.json
    - UUID-prefixed: {uuid}_content_list.json  (v1 flat format)
    - V2: content_list_v2.json  (nested format, less preferred)
    We prefer the v1 flat format (*_content_list.json) over v2.
    """
    # 1. Exact match
    direct = output_dir / "content_list.json"
    if direct.exists():
        return direct

    # 2. Recursive exact match
    for p in output_dir.rglob("content_list.json"):
        return p

    # 3. UUID-prefixed *_content_list.json (preferred v1 flat format)
    for p in sorted(output_dir.rglob("*_content_list.json")):
        return p

    # 4. content_list_v2.json (nested format, needs different parsing)
    for p in output_dir.rglob("content_list_v2.json"):
        return p

    raise FileNotFoundError(f"content_list.json not found under {output_dir}")


def _build_section_path(title_blocks: list[dict], current_idx: int) -> str:
    """Build hierarchical section path from title blocks seen so far."""
    # Collect title blocks up to current position
    titles: list[tuple[int, str]] = []
    for tb in title_blocks:
        if tb["order_idx"] <= current_idx:
            level = tb.get("level", 0)
            titles.append((level, tb["text"]))

    if not titles:
        return ""

    # Build path from most recent titles at each level
    path_parts: list[str] = []
    seen_levels: dict[int, str] = {}
    for level, text in titles:
        seen_levels[level] = text
        # Clear deeper levels when we see a higher-level title
        for k in list(seen_levels.keys()):
            if k > level:
                del seen_levels[k]

    for level in sorted(seen_levels.keys()):
        path_parts.append(seen_levels[level])

    return "/".join(path_parts)


def _guess_title_level(text: str) -> int:
    """Guess heading level from text patterns like '1.', '1.1', '1.1.1'."""
    m = re.match(r"^(\d+(?:\.\d+)*)\s", text.strip())
    if m:
        return m.group(1).count(".") + 1

    # Common top-level headings
    top_level = {"abstract", "introduction", "conclusion", "conclusions",
                 "references", "bibliography", "acknowledgments", "acknowledgements",
                 "related work", "background", "discussion"}
    if text.strip().lower() in top_level:
        return 1

    return 1


def parse_content_list(output_dir: Path, paper_id: str) -> PaperIR:
    """Parse content_list.json into PaperIR with sections and blocks.

    Handles MinerU v1 format where:
    - Titles are type="text" with a "text_level" field (1=H1, 2=H2, etc.)
    - Types include: text, image, table, equation, list, header, footer, page_number, aside_text
    - No explicit "title" type in v1; we promote text_level items to title role
    """
    content_list_path = _find_content_list(output_dir)
    with open(content_list_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError(f"Expected list in content_list.json, got {type(items)}")

    blocks: list[Block] = []
    title_entries: list[dict] = []
    paper_title = ""

    for idx, item in enumerate(items):
        block_type = item.get("type", "text")
        sub_type = item.get("sub_type", "")
        page_idx = item.get("page_idx", 0)
        bbox = item.get("bbox", [])
        text = item.get("text", "")
        text_level = item.get("text_level")

        # Skip boilerplate types
        if block_type in ("header", "footer", "page_number"):
            continue

        # Handle different content types
        if block_type == "image":
            text = item.get("img_caption", "") or item.get("text", "") or "[image]"
        elif block_type == "table":
            text = item.get("table_body", "") or item.get("text", "") or item.get("html", "")

        if isinstance(bbox, (list, tuple)):
            bbox = [float(x) for x in bbox]
        else:
            bbox = []

        # Detect titles: explicit "title" type OR text with text_level
        is_title = block_type == "title" or (text_level is not None and isinstance(text_level, int))

        if is_title:
            block_type = "title"
            level = int(text_level) if text_level is not None else _guess_title_level(text)
            title_entries.append({
                "order_idx": idx,
                "text": text.strip(),
                "level": level,
            })
            # First title-level-1 item in the first few blocks is the paper title
            if not paper_title and level <= 1 and idx < 10:
                paper_title = text.strip()

        block = Block(
            type=block_type,
            sub_type=sub_type,
            page_idx=page_idx,
            bbox=bbox,
            text=text,
            section_path="",
            order_idx=idx,
        )
        blocks.append(block)

    # Assign section paths
    for i, block in enumerate(blocks):
        block.section_path = _build_section_path(title_entries, block.order_idx)

    # Build sections
    sections: list[Section] = []
    current_section: Section | None = None
    for block in blocks:
        if block.type == "title":
            if current_section:
                sections.append(current_section)
            current_section = Section(
                path=block.section_path,
                title=block.text,
                level=_guess_title_level(block.text),
                blocks=[],
            )
        if current_section:
            current_section.blocks.append(block)
        else:
            # Blocks before first title
            if not sections:
                current_section = Section(path="", title="", level=0, blocks=[])
            if current_section:
                current_section.blocks.append(block)

    if current_section:
        sections.append(current_section)

    return PaperIR(
        paper_id=paper_id,
        title=paper_title,
        sections=sections,
        blocks=blocks,
    )


async def build_and_store_paper_ir(output_dir: Path, paper_id: str) -> PaperIR:
    """Parse content_list.json, build PaperIR, and store blocks in DB."""
    paper_ir = parse_content_list(output_dir, paper_id)

    # Update paper title if found
    if paper_ir.title:
        await db.execute(
            "UPDATE papers SET title = ? WHERE paper_id = ? AND (title IS NULL OR title = '')",
            (paper_ir.title, paper_id),
        )

    # Clear existing blocks for this paper
    await db.execute("DELETE FROM blocks WHERE paper_id = ?", (paper_id,))

    # Insert blocks
    rows = [
        (
            paper_id,
            b.type,
            b.sub_type,
            b.page_idx,
            json.dumps(b.bbox),
            b.text,
            b.section_path,
            b.order_idx,
        )
        for b in paper_ir.blocks
    ]
    if rows:
        await db.execute_many(
            """INSERT INTO blocks (paper_id, type, sub_type, page_idx, bbox_json, text, section_path, order_idx)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    # Save normalized PaperIR JSON
    normalized_dir = Path(output_dir).parent / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    ir_path = normalized_dir / "paper_ir.json"
    ir_path.write_text(paper_ir.model_dump_json(indent=2), encoding="utf-8")

    return paper_ir
