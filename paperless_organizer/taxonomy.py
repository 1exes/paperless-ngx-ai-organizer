"""Tag taxonomy: canonical tags + synonym mapping from taxonomy_tags.json."""

from __future__ import annotations

import json
import os

from .config import MAX_PROMPT_TAG_CHOICES, log
from .constants import TAG_COLORS
from .utils import _normalize_tag_name, _normalize_text


class TagTaxonomy:
    """Canonical tags + synonym mapping loaded from taxonomy_tags.json."""

    def __init__(self, path: str):
        self.path = path
        self.canonical_tags: list[str] = []
        self.synonym_to_canonical: dict[str, str] = {}
        self.metadata_by_canonical: dict[str, dict] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            self.canonical_tags = []
            self.synonym_to_canonical = {}
            self.metadata_by_canonical = {}
            log.warning(f"Taxonomie-Datei nicht gefunden: {self.path}")
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        canonical = []
        synonym_map = {}
        metadata_map = {}
        for entry in data.get("tags", []):
            name = _normalize_text(entry.get("name", ""))
            if not name:
                continue
            canonical.append(name)
            synonym_map[_normalize_tag_name(name)] = name
            for syn in entry.get("synonyms", []):
                syn_name = _normalize_tag_name(str(syn))
                if syn_name:
                    synonym_map[syn_name] = name
            metadata_map[name] = {
                "color": _normalize_text(str(entry.get("color", ""))) if entry.get("color") else "",
                "description": _normalize_text(str(entry.get("description", ""))) if entry.get("description") else "",
            }

        self.canonical_tags = sorted(set(canonical), key=lambda x: x.lower())
        self.synonym_to_canonical = synonym_map
        self.metadata_by_canonical = metadata_map

    def canonical_from_any(self, value: str) -> str | None:
        key = _normalize_tag_name(value)
        if not key:
            return None
        return self.synonym_to_canonical.get(key)

    def metadata_for_tag(self, canonical_name: str) -> dict:
        return dict(self.metadata_by_canonical.get(canonical_name, {}))

    def color_for_tag(self, canonical_name: str) -> str:
        meta = self.metadata_for_tag(canonical_name)
        color = _normalize_text(meta.get("color", ""))
        if color.startswith("#") and len(color) == 7:
            return color.lower()
        idx = sum(ord(ch) for ch in _normalize_tag_name(canonical_name)) % len(TAG_COLORS)
        return TAG_COLORS[idx]

    def description_for_tag(self, canonical_name: str) -> str:
        meta = self.metadata_for_tag(canonical_name)
        desc = _normalize_text(meta.get("description", ""))
        if desc:
            return desc
        return f"Kategorie fuer {canonical_name}."

    def prompt_tags(self, limit: int = MAX_PROMPT_TAG_CHOICES) -> list[str]:
        return self.canonical_tags[:max(1, limit)]
