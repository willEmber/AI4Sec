from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


PUBLIC_FIELDS: tuple[str, ...] = (
    "title",
    "abstract",
    "url",
    "doi",
    "authors",
    "year",
    "venue",
    "source_platform",
)


@dataclass
class Paper:
    title: str
    abstract: str
    url: str
    doi: str
    authors: str
    source_platform: str
    year: int = 0
    venue: str = ""

    # Internal fields (not part of the public output schema)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, fields: Iterable[str] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "doi": self.doi,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "source_platform": self.source_platform,
        }
        if fields is None:
            return data

        out: dict[str, Any] = {}
        for key in fields:
            if key in data:
                out[key] = data[key]
        return out
