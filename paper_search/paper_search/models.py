from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


PUBLIC_FIELDS: tuple[str, ...] = (
    "title",
    "abstract",
    "url",
    "doi",
    "authors",
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

    # Internal fields (not part of the public output schema)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, fields: Iterable[str] | None = None) -> dict[str, str]:
        data = {
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "doi": self.doi,
            "authors": self.authors,
            "source_platform": self.source_platform,
        }
        if fields is None:
            return data

        out: dict[str, str] = {}
        for key in fields:
            if key in data:
                out[key] = data[key]
        return out
