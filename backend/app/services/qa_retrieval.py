from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.db import database as db
from app.models.paper_ir import Block, PaperIR

logger = logging.getLogger("scholar.qa_retrieval")


_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "is", "are", "was", "were",
    "to", "for", "and", "or", "but", "with", "as", "by", "from", "this",
    "that", "these", "those", "it", "its", "what", "how", "why", "which",
    "who", "where", "when", "do", "does", "did", "can", "could", "should",
    "would", "will", "be", "been", "being", "have", "has", "had", "you",
    "your", "they", "their", "paper", "authors",
})
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SEARCHABLE_BLOCK_TYPES = frozenset({"text", "title", "list", "equation", "table", "image"})

_CJK_QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("指标", "数据", "性能", "表现", "评估", "实验", "结果", "对比", "准确率", "鲁棒", "保真", "攻击"),
        (
            "metric", "metrics", "performance", "evaluation", "experiment", "experiments",
            "result", "results", "accuracy", "robust", "robustness", "fidelity",
            "psnr", "ssim", "lpips", "fid", "ablation", "comparison", "baseline",
            "table",
        ),
    ),
    (
        ("方法", "方案", "算法", "模型", "框架", "架构"),
        ("method", "approach", "proposed", "model", "framework", "architecture"),
    ),
    (
        ("总结", "概括", "贡献", "创新", "全文"),
        ("abstract", "introduction", "conclusion", "contribution", "summary"),
    ),
)


@dataclass(frozen=True)
class PaperNode:
    node_id: str
    paper_id: str
    parent_id: str
    depth: int
    node_type: str
    block_type: str
    sub_type: str
    title: str
    title_path: str
    page_start: int
    page_end: int
    block_start: int
    block_end: int
    text: str
    text_for_search: str
    order_idx: int


@dataclass(frozen=True)
class QueryIntent:
    name: str
    prefer_tables: bool = False
    expand_section_siblings: bool = False
    section_terms: tuple[str, ...] = ()


def _tokenize(question: str) -> list[str]:
    """Lowercase tokens, drop stopwords, and add Chinese intent expansions."""
    tokens: list[str] = []
    question_l = question.lower()
    for match in _TOKEN_RE.findall(question_l):
        if match not in _STOPWORDS:
            tokens.append(match)
    for run in re.findall(r"[一-鿿]{2,4}", question):
        tokens.append(run)
    for triggers, expansions in _CJK_QUERY_EXPANSIONS:
        if any(trigger in question for trigger in triggers):
            tokens.extend(expansions)

    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _detect_intent(question: str, tokens: list[str]) -> QueryIntent:
    q_l = question.lower()
    token_set = set(tokens)
    if any(term in q_l for term in ("指标", "数据", "性能", "实验", "结果", "准确率", "鲁棒", "保真")) or (
        token_set & {"metric", "metrics", "performance", "accuracy", "psnr", "ssim", "fidelity", "robustness"}
    ):
        return QueryIntent(
            name="metrics",
            prefer_tables=True,
            expand_section_siblings=True,
            section_terms=(
                "experiment", "experiments", "evaluation", "result", "results", "analysis",
                "metric", "metrics", "table", "ablation", "comparison", "性能", "实验", "结果", "指标",
            ),
        )
    if any(term in q_l for term in ("方法", "方案", "算法", "模型", "框架", "架构")) or (
        token_set & {"method", "approach", "model", "framework", "architecture", "algorithm"}
    ):
        return QueryIntent(
            name="method",
            expand_section_siblings=True,
            section_terms=("method", "approach", "model", "framework", "architecture", "proposed", "方法", "模型"),
        )
    if any(term in q_l for term in ("总结", "概括", "贡献", "创新", "全文", "主要内容")) or (
        token_set & {"summary", "contribution", "abstract", "conclusion"}
    ):
        return QueryIntent(
            name="summary",
            expand_section_siblings=True,
            section_terms=("abstract", "introduction", "conclusion", "discussion", "contribution", "summary"),
        )
    return QueryIntent(name="fact")


def _section_prefixes(path: str) -> list[str]:
    parts = [part.strip() for part in path.split("/") if part.strip()]
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _page_range(blocks: list[Block]) -> tuple[int, int]:
    if not blocks:
        return 0, 0
    pages = [block.page_idx for block in blocks]
    return min(pages), max(pages)


def _block_range(blocks: list[Block]) -> tuple[int, int]:
    if not blocks:
        return 0, 0
    order_indexes = [block.order_idx for block in blocks]
    return min(order_indexes), max(order_indexes)


def _safe_node_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_")[:120] or "untitled"


def build_paper_nodes(paper_ir: PaperIR) -> list[PaperNode]:
    """Build a paper -> section -> original evidence chunk hierarchy from PaperIR."""
    blocks = sorted(paper_ir.blocks, key=lambda block: block.order_idx)
    page_start, page_end = _page_range(blocks)
    block_start, block_end = _block_range(blocks)

    root_id = f"{paper_ir.paper_id}:paper"
    section_paths: list[str] = []
    seen_paths: set[str] = set()
    for block in blocks:
        for prefix in _section_prefixes(block.section_path):
            if prefix not in seen_paths:
                seen_paths.add(prefix)
                section_paths.append(prefix)

    nodes: list[PaperNode] = [
        PaperNode(
            node_id=root_id,
            paper_id=paper_ir.paper_id,
            parent_id="",
            depth=0,
            node_type="paper",
            block_type="",
            sub_type="",
            title=paper_ir.title,
            title_path=paper_ir.title,
            page_start=page_start,
            page_end=page_end,
            block_start=block_start,
            block_end=block_end,
            text=paper_ir.title,
            text_for_search=" ".join(part for part in [paper_ir.title, " ".join(section_paths)] if part),
            order_idx=0,
        )
    ]

    section_id_by_path: dict[str, str] = {}
    for idx, path in enumerate(section_paths, start=1):
        parts = path.split("/")
        parent_path = "/".join(parts[:-1])
        parent_id = section_id_by_path.get(parent_path, root_id)
        matching_blocks = [
            block for block in blocks
            if block.section_path == path or block.section_path.startswith(path + "/")
        ]
        section_page_start, section_page_end = _page_range(matching_blocks)
        section_block_start, section_block_end = _block_range(matching_blocks)
        title = parts[-1]
        node_id = f"{paper_ir.paper_id}:section:{idx}:{_safe_node_suffix(path)}"
        section_id_by_path[path] = node_id
        section_text = " ".join(
            block.text.strip()
            for block in matching_blocks
            if block.type == "title" and block.text.strip()
        )[:2000]
        nodes.append(
            PaperNode(
                node_id=node_id,
                paper_id=paper_ir.paper_id,
                parent_id=parent_id,
                depth=len(parts),
                node_type="section",
                block_type="",
                sub_type="",
                title=title,
                title_path=path,
                page_start=section_page_start,
                page_end=section_page_end,
                block_start=section_block_start,
                block_end=section_block_end,
                text=section_text or title,
                text_for_search=" ".join(part for part in [paper_ir.title, path, section_text] if part),
                order_idx=section_block_start,
            )
        )

    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        section_path = block.section_path.strip()
        parent_id = section_id_by_path.get(section_path, root_id)
        title_path = section_path or paper_ir.title
        depth = len(_section_prefixes(section_path)) + 1
        text_for_search = " ".join(
            part for part in [paper_ir.title, title_path, block.type, block.sub_type, text] if part
        )
        nodes.append(
            PaperNode(
                node_id=f"{paper_ir.paper_id}:chunk:{block.order_idx}",
                paper_id=paper_ir.paper_id,
                parent_id=parent_id,
                depth=depth,
                node_type="chunk",
                block_type=block.type,
                sub_type=block.sub_type,
                title=title_path.split("/")[-1] if title_path else paper_ir.title,
                title_path=title_path,
                page_start=block.page_idx,
                page_end=block.page_idx,
                block_start=block.order_idx,
                block_end=block.order_idx,
                text=text,
                text_for_search=text_for_search,
                order_idx=block.order_idx,
            )
        )

    return nodes


def _score_node(
    node: PaperNode,
    tokens: list[str],
    intent: QueryIntent,
    boosted_node_ids: set[str] | None = None,
) -> int:
    if node.node_type != "chunk" or node.block_type not in _SEARCHABLE_BLOCK_TYPES:
        return 0

    text_l = node.text_for_search.lower()
    score = 0
    for token in tokens:
        if token and token.lower() in text_l:
            score += 1

    if boosted_node_ids and node.node_id in boosted_node_ids:
        score += 6
    if node.block_type == "table" and intent.prefer_tables:
        score += 5
    if node.block_type == "title" and any(term.lower() in text_l for term in intent.section_terms):
        score += 3
    if node.title_path and any(term.lower() in node.title_path.lower() for term in intent.section_terms):
        score += 4
    if intent.name == "metrics" and re.search(
        r"\b(psnr|ssim|lpips|fid|accuracy|fidelity|robust|robustness|metric|table)\b",
        text_l,
    ):
        score += 2
    if intent.name == "summary" and node.block_type in {"title", "text"} and node.page_start <= 1:
        score += 1

    return score


def _select_context_nodes(
    nodes: list[PaperNode],
    question: str,
    boosted_node_ids: set[str] | None = None,
) -> list[PaperNode]:
    tokens = _tokenize(question)
    intent = _detect_intent(question, tokens)
    chunks = [
        node for node in nodes
        if node.node_type == "chunk" and node.block_type in _SEARCHABLE_BLOCK_TYPES and node.text.strip()
    ]
    scored = [
        (_score_node(node, tokens, intent, boosted_node_ids), node)
        for node in chunks
    ]
    scored = [(score, node) for score, node in scored if score > 0]
    scored.sort(key=lambda item: (-item[0], item[1].page_start, item[1].order_idx))

    top = [node for _, node in scored[:30]]
    if not top:
        top = chunks[:12]

    children_by_parent: dict[str, list[PaperNode]] = defaultdict(list)
    by_order = {node.block_start: node for node in chunks}
    for node in chunks:
        children_by_parent[node.parent_id].append(node)
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: item.order_idx)

    selected: dict[str, PaperNode] = {}

    def add(node: PaperNode | None) -> None:
        if node is None or node.node_id in selected:
            return
        selected[node.node_id] = node

    for anchor in [
        node for node in chunks
        if node.page_start == 0 and node.block_type in {"title", "text"}
    ][:5]:
        add(anchor)

    for node in top:
        siblings = children_by_parent.get(node.parent_id, [])
        for sibling in siblings:
            if sibling.block_type == "title":
                add(sibling)

        for order_idx in range(node.block_start - 1, node.block_start + 2):
            add(by_order.get(order_idx))
        add(node)

        if intent.expand_section_siblings:
            matching_siblings = [
                sibling for sibling in siblings
                if sibling.block_type == "table"
                or _score_node(sibling, tokens, intent, boosted_node_ids) > 0
            ]
            for sibling in matching_siblings[:12]:
                add(sibling)
            for sibling in siblings[:8]:
                add(sibling)

    return sorted(selected.values(), key=lambda item: (item.page_start, item.order_idx))


def _format_context(nodes: list[PaperNode], max_chars: int) -> str:
    parts: list[str] = []
    for node in nodes:
        text = node.text.strip()
        if not text:
            continue
        section = f" [{node.title_path}]" if node.title_path else ""
        parts.append(f"[p.{node.page_start + 1}]{section} {text}")

    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[... truncated for length ...]"
    return context


def retrieve_qa_context(paper_ir: PaperIR, question: str, max_chars: int = 20000) -> tuple[str, int]:
    nodes = build_paper_nodes(paper_ir)
    chosen = _select_context_nodes(nodes, question)
    return _format_context(chosen, max_chars), len(chosen)


def _paper_node_from_row(row: dict[str, Any]) -> PaperNode:
    return PaperNode(
        node_id=row["node_id"],
        paper_id=row["paper_id"],
        parent_id=row.get("parent_id") or "",
        depth=int(row.get("depth") or 0),
        node_type=row.get("node_type") or "",
        block_type=row.get("block_type") or "",
        sub_type=row.get("sub_type") or "",
        title=row.get("title") or "",
        title_path=row.get("title_path") or "",
        page_start=int(row.get("page_start") or 0),
        page_end=int(row.get("page_end") or 0),
        block_start=int(row.get("block_start") or 0),
        block_end=int(row.get("block_end") or 0),
        text=row.get("text") or "",
        text_for_search=row.get("text_for_search") or "",
        order_idx=int(row.get("order_idx") or 0),
    )


async def ensure_paper_node_fts() -> bool:
    try:
        await db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS paper_node_fts "
            "USING fts5(node_id UNINDEXED, paper_id UNINDEXED, title_path, text_for_search)"
        )
        return True
    except Exception as exc:
        logger.warning("qa retrieval: FTS5 unavailable; continuing without paper_node_fts: %s", exc)
        return False


async def store_paper_nodes(paper_ir: PaperIR) -> int:
    nodes = build_paper_nodes(paper_ir)
    await db.execute("DELETE FROM paper_nodes WHERE paper_id = ?", (paper_ir.paper_id,))
    if not nodes:
        return 0

    await db.execute_many(
        """
        INSERT INTO paper_nodes (
            node_id, paper_id, parent_id, depth, node_type, block_type, sub_type,
            title, title_path, page_start, page_end, block_start, block_end,
            text, text_for_search, order_idx
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                node.node_id,
                node.paper_id,
                node.parent_id,
                node.depth,
                node.node_type,
                node.block_type,
                node.sub_type,
                node.title,
                node.title_path,
                node.page_start,
                node.page_end,
                node.block_start,
                node.block_end,
                node.text,
                node.text_for_search,
                node.order_idx,
            )
            for node in nodes
        ],
    )

    if await ensure_paper_node_fts():
        try:
            await db.execute("DELETE FROM paper_node_fts WHERE paper_id = ?", (paper_ir.paper_id,))
            await db.execute_many(
                "INSERT INTO paper_node_fts (node_id, paper_id, title_path, text_for_search) VALUES (?, ?, ?, ?)",
                [
                    (node.node_id, node.paper_id, node.title_path, node.text_for_search)
                    for node in nodes
                    if node.node_type == "chunk"
                ],
            )
        except Exception as exc:
            logger.warning("qa retrieval: failed to refresh paper_node_fts for %s: %s", paper_ir.paper_id, exc)

    return len(nodes)


async def load_paper_nodes(paper_id: str) -> list[PaperNode]:
    rows = await db.fetch_all(
        """
        SELECT node_id, paper_id, parent_id, depth, node_type, block_type, sub_type,
               title, title_path, page_start, page_end, block_start, block_end,
               text, text_for_search, order_idx
        FROM paper_nodes
        WHERE paper_id = ?
        ORDER BY depth, order_idx
        """,
        (paper_id,),
    )
    return [_paper_node_from_row(row) for row in rows]


def _fts_query(tokens: list[str]) -> str:
    terms = [
        token for token in tokens
        if re.fullmatch(r"[A-Za-z0-9_-]{3,}", token)
    ][:10]
    return " OR ".join(f'"{term}"' for term in terms)


async def _search_fts_node_ids(paper_id: str, question: str) -> set[str]:
    tokens = _tokenize(question)
    query = _fts_query(tokens)
    if not query:
        return set()
    try:
        rows = await db.fetch_all(
            """
            SELECT node_id
            FROM paper_node_fts
            WHERE paper_node_fts MATCH ? AND paper_id = ?
            LIMIT 80
            """,
            (query, paper_id),
        )
    except Exception as exc:
        logger.debug("qa retrieval: FTS query skipped for %s: %s", paper_id, exc)
        return set()
    return {row["node_id"] for row in rows}


async def retrieve_qa_context_for_paper(
    paper_id: str,
    paper_ir: PaperIR,
    question: str,
    max_chars: int = 20000,
) -> tuple[str, int]:
    try:
        nodes = await load_paper_nodes(paper_id)
    except Exception as exc:
        logger.debug("qa retrieval: DB node load skipped for %s: %s", paper_id, exc)
        nodes = []

    if not nodes:
        return retrieve_qa_context(paper_ir, question, max_chars=max_chars)

    boosted_ids = await _search_fts_node_ids(paper_id, question)
    chosen = _select_context_nodes(nodes, question, boosted_node_ids=boosted_ids)
    return _format_context(chosen, max_chars), len(chosen)
