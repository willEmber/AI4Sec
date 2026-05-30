from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db import database as db
from app.rate_limit import limiter
from app.models.schemas import PaperResponse, PaperUploadResponse

router = APIRouter(tags=["papers"])

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

# Figure images extracted by MinerU are named as a content hash + extension.
# Restrict to that shape so the filename can never contain path separators.
_SAFE_IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.(jpg|jpeg|png|gif|webp)$", re.IGNORECASE)
_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@router.post("/papers/upload", response_model=PaperUploadResponse)
@limiter.limit("5/minute")
async def upload_paper(request: Request, file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Read in chunks with size limit to prevent OOM
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB per chunk
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024 * 1024)} MB)",
            )
        chunks.append(chunk)
    content = b"".join(chunks)
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    # A .pdf extension is trivially spoofed; require the real PDF signature so
    # non-PDF payloads are rejected before they reach MinerU / storage.
    if not content[:1024].lstrip().startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    paper_id = hashlib.sha1(content).hexdigest()
    settings = get_settings()
    paper_dir = settings.data_dir / "papers" / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = paper_dir / "original.pdf"
    if not pdf_path.exists():
        tmp = pdf_path.with_suffix(".pdf.part")
        tmp.write_bytes(content)
        tmp.replace(pdf_path)

    existing = await db.fetch_one("SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,))
    if existing:
        return PaperUploadResponse(paper_id=paper_id, message="Paper already exists")

    rel_path = f"papers/{paper_id}/original.pdf"
    await db.execute(
        "INSERT INTO papers (paper_id, file_path) VALUES (?, ?)",
        (paper_id, rel_path),
    )
    return PaperUploadResponse(paper_id=paper_id, message="Upload successful")


@router.get("/papers/{paper_id}", response_model=PaperResponse)
@limiter.limit("30/minute")
async def get_paper(request: Request, paper_id: str):
    row = await db.fetch_one("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperResponse(**row)


@router.get("/papers/{paper_id}/pdf")
@limiter.limit("20/minute")
async def get_paper_pdf(request: Request, paper_id: str):
    row = await db.fetch_one("SELECT file_path FROM papers WHERE paper_id = ?", (paper_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Paper not found")

    settings = get_settings()
    pdf_path = (settings.data_dir / row["file_path"]).resolve()

    # Path traversal guard: resolved path must stay inside data_dir
    if not pdf_path.is_relative_to(settings.data_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{paper_id}.pdf",
    )


@router.get("/papers/{paper_id}/images/{filename}")
@limiter.limit("120/minute")
async def get_paper_image(request: Request, paper_id: str, filename: str):
    """Serve a MinerU-extracted figure image so reports can embed it inline.

    The file is located under the paper's own MinerU output (``.../images/``);
    the filename is restricted to a safe pattern and the resolved path is
    confirmed to stay inside ``data_dir`` to prevent path traversal.
    """
    if not _SAFE_IMAGE_NAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="Image not found")

    settings = get_settings()
    data_root = settings.data_dir.resolve()
    mineru_dir = (settings.data_dir / "papers" / paper_id / "mineru").resolve()
    if not mineru_dir.is_relative_to(data_root) or not mineru_dir.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    match: Path | None = None
    for cand in mineru_dir.rglob(filename):
        if cand.is_file() and cand.parent.name == "images":
            match = cand.resolve()
            break
    if match is None or not match.is_relative_to(data_root):
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = _IMAGE_MEDIA_TYPES.get(match.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(match), media_type=media_type)
