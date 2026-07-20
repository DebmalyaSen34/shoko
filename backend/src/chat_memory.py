import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

MemoryScope = Literal["clip", "project"]

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "use",
    "with",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_'-]{2,}", text.lower())
        if token not in STOP_WORDS
    }


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@dataclass
class MemoryCandidate:
    item: dict
    score: float
    reasons: list[str]


class ChatMemoryStore:
    schema_version = 2

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            return self._empty()
        return self._migrate(data)

    def save(self, data: dict) -> None:
        atomic_write_json(self.path, data)

    def add(
        self,
        text: str,
        *,
        scope: MemoryScope,
        clip_key: Optional[str] = None,
        source: str = "chat",
        tags: Optional[list[str]] = None,
        confidence: float = 0.8,
    ) -> dict:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            raise ValueError("Memory text cannot be empty")
        if scope == "clip" and not clip_key:
            raise ValueError("clip_key is required for clip memory")

        data = self.load()
        fingerprint = self._fingerprint(scope, clip_key, cleaned)
        existing = next((item for item in data["memories"] if item.get("fingerprint") == fingerprint), None)
        if existing:
            existing["updated_at"] = now_iso()
            existing["source"] = source
            existing["confidence"] = max(float(existing.get("confidence", 0.0)), confidence)
            existing["access_count"] = int(existing.get("access_count", 0)) + 1
            existing_tags = set(existing.get("tags", []))
            existing["tags"] = sorted(existing_tags.union(tags or []))
            self.save(data)
            return existing

        item = {
            "id": str(uuid.uuid4()),
            "scope": scope,
            "clip_key": clip_key if scope == "clip" else None,
            "text": cleaned,
            "normalized_text": normalize_text(cleaned),
            "fingerprint": fingerprint,
            "tags": sorted(set(tags or [])),
            "confidence": max(0.0, min(1.0, confidence)),
            "source": source,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "last_accessed_at": None,
            "access_count": 0,
            "archived": False,
        }
        data["memories"].append(item)
        self.save(data)
        return item

    def retrieve(
        self,
        query: str,
        *,
        clip_key: Optional[str] = None,
        limit: int = 12,
        include_project: bool = True,
        include_clip: bool = True,
    ) -> list[MemoryCandidate]:
        data = self.load()
        query_tokens = tokenize(query)
        candidates: list[MemoryCandidate] = []

        for item in data["memories"]:
            if item.get("archived"):
                continue
            scope = item.get("scope")
            if scope == "project" and not include_project:
                continue
            if scope == "clip" and (not include_clip or item.get("clip_key") != clip_key):
                continue
            score, reasons = self._score(item, query_tokens, clip_key)
            if score > 0:
                candidates.append(MemoryCandidate(item=item, score=score, reasons=reasons))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        selected = candidates[:limit]
        if selected:
            self._touch([candidate.item["id"] for candidate in selected])
        return selected

    def for_clip(self, clip_key: str, *, query: str = "", limit: int = 12) -> dict:
        data = self.load()
        project = [
            item for item in data["memories"]
            if item.get("scope") == "project" and not item.get("archived")
        ]
        clip = [
            item for item in data["memories"]
            if item.get("scope") == "clip" and item.get("clip_key") == clip_key and not item.get("archived")
        ]
        project.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        clip.sort(key=lambda item: item.get("updated_at", ""), reverse=True)

        relevant = [
            {
                **candidate.item,
                "relevance_score": round(candidate.score, 4),
                "relevance_reasons": candidate.reasons,
            }
            for candidate in self.retrieve(query, clip_key=clip_key, limit=limit)
        ] if query.strip() else []

        return {
            "project": project[:limit],
            "clip": clip[:limit],
            "relevant": relevant,
        }

    def _touch(self, ids: list[str]) -> None:
        data = self.load()
        changed = False
        target_ids = set(ids)
        for item in data["memories"]:
            if item.get("id") in target_ids:
                item["last_accessed_at"] = now_iso()
                item["access_count"] = int(item.get("access_count", 0)) + 1
                changed = True
        if changed:
            self.save(data)

    def _score(self, item: dict, query_tokens: set[str], clip_key: Optional[str]) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        if item.get("scope") == "clip" and item.get("clip_key") == clip_key:
            score += 0.35
            reasons.append("same clip")
        elif item.get("scope") == "project":
            score += 0.22
            reasons.append("project memory")

        item_tokens = tokenize(item.get("text", ""))
        if query_tokens and item_tokens:
            overlap = len(query_tokens.intersection(item_tokens))
            if overlap:
                similarity = overlap / math.sqrt(len(query_tokens) * len(item_tokens))
                score += similarity
                reasons.append(f"{overlap} keyword match")

        score += min(float(item.get("confidence", 0.0)), 1.0) * 0.14
        access_count = int(item.get("access_count", 0))
        if access_count:
            score += min(access_count, 8) * 0.015
            reasons.append("reinforced")

        return score, reasons

    def _migrate(self, data: Any) -> dict:
        if isinstance(data, dict) and data.get("schema_version") == self.schema_version and isinstance(data.get("memories"), list):
            data.setdefault("memories", [])
            return data

        migrated = self._empty()

        # v1 shape: {"project": [...], "clips": {"clip::0": [...]}}
        if isinstance(data, dict) and ("project" in data or "clips" in data):
            for item in data.get("project", []):
                text = item.get("text") if isinstance(item, dict) else str(item)
                if text:
                    migrated["memories"].append(self._migrated_item(text, scope="project", clip_key=None, source=item.get("source", "migration") if isinstance(item, dict) else "migration"))
            for clip_key, items in data.get("clips", {}).items():
                for item in items:
                    text = item.get("text") if isinstance(item, dict) else str(item)
                    if text:
                        migrated["memories"].append(self._migrated_item(text, scope="clip", clip_key=clip_key, source=item.get("source", "migration") if isinstance(item, dict) else "migration"))

        return migrated

    def _migrated_item(self, text: str, *, scope: MemoryScope, clip_key: Optional[str], source: str) -> dict:
        created_at = now_iso()
        return {
            "id": str(uuid.uuid4()),
            "scope": scope,
            "clip_key": clip_key,
            "text": text,
            "normalized_text": normalize_text(text),
            "fingerprint": self._fingerprint(scope, clip_key, text),
            "tags": [],
            "confidence": 0.7,
            "source": source,
            "created_at": created_at,
            "updated_at": created_at,
            "last_accessed_at": None,
            "access_count": 0,
            "archived": False,
        }

    def _fingerprint(self, scope: MemoryScope, clip_key: Optional[str], text: str) -> str:
        owner = clip_key if scope == "clip" else "project"
        return f"{scope}:{owner}:{normalize_text(text)}"

    def _empty(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "memories": [],
        }
