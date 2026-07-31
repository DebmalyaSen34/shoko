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


LessonScope = Literal["project", "clip"]

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
class LessonCandidate:
    item: dict
    score: float
    reasons: list[str]


class PromptLearningStore:
    schema_version = 1

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
        if not isinstance(data, dict):
            return self._empty()
        lessons = data.get("lessons")
        if not isinstance(lessons, list):
            lessons = []
        return {
            "schema_version": self.schema_version,
            "lessons": [self._normalize_lesson(item) for item in lessons if isinstance(item, dict)],
        }

    def save(self, data: dict) -> None:
        data.setdefault("schema_version", self.schema_version)
        data.setdefault("lessons", [])
        atomic_write_json(self.path, data)

    def add_lesson(
        self,
        lesson: str,
        *,
        scope: LessonScope = "project",
        clip_key: Optional[str] = None,
        category: str = "other",
        source_feedback_ids: Optional[list[str]] = None,
        confidence: float = 0.8,
        positive_examples: Optional[list[str]] = None,
        negative_examples: Optional[list[str]] = None,
    ) -> dict:
        cleaned = self._clean_lesson_text(lesson)
        self._validate_scope(scope, clip_key)
        created_at = now_iso()
        item = {
            "id": str(uuid.uuid4()),
            "scope": scope,
            "clip_key": clip_key if scope == "clip" else None,
            "category": category or "other",
            "lesson": cleaned,
            "normalized_lesson": normalize_text(cleaned),
            "source_feedback_ids": sorted(set(source_feedback_ids or [])),
            "confidence": self._clamp_confidence(confidence),
            "positive_examples": self._clean_examples(positive_examples),
            "negative_examples": self._clean_examples(negative_examples),
            "created_at": created_at,
            "updated_at": created_at,
            "last_accessed_at": None,
            "access_count": 0,
            "archived": False,
        }
        data = self.load()
        data["lessons"].append(item)
        self.save(data)
        return item

    def update_lesson(self, lesson_id: str, updates: dict) -> dict:
        data = self.load()
        item = next((candidate for candidate in data["lessons"] if candidate.get("id") == lesson_id), None)
        if item is None:
            raise KeyError(lesson_id)

        if "lesson" in updates:
            item["lesson"] = self._clean_lesson_text(str(updates.get("lesson") or ""))
            item["normalized_lesson"] = normalize_text(item["lesson"])
        if "scope" in updates or "clip_key" in updates:
            scope = updates.get("scope", item.get("scope"))
            clip_key = updates.get("clip_key", item.get("clip_key"))
            self._validate_scope(scope, clip_key)
            item["scope"] = scope
            item["clip_key"] = clip_key if scope == "clip" else None
        if "category" in updates:
            item["category"] = str(updates.get("category") or "other")
        if "source_feedback_ids" in updates:
            item["source_feedback_ids"] = sorted(set(str(value) for value in updates.get("source_feedback_ids") or [] if value))
        if "confidence" in updates:
            item["confidence"] = self._clamp_confidence(updates.get("confidence"))
        if "positive_examples" in updates:
            item["positive_examples"] = self._clean_examples(updates.get("positive_examples"))
        if "negative_examples" in updates:
            item["negative_examples"] = self._clean_examples(updates.get("negative_examples"))
        if "archived" in updates:
            item["archived"] = bool(updates.get("archived"))

        item["updated_at"] = now_iso()
        self.save(data)
        return item

    def list_lessons(
        self,
        *,
        clip_key: Optional[str] = None,
        category: Optional[str] = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        lessons = []
        for item in self.load()["lessons"]:
            if item.get("archived") and not include_archived:
                continue
            if category and item.get("category") != category:
                continue
            if clip_key and item.get("scope") == "clip" and item.get("clip_key") != clip_key:
                continue
            lessons.append(item)
        lessons.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return lessons[: max(1, min(limit, 500))]

    def retrieve_lessons(
        self,
        query: str,
        *,
        clip_key: Optional[str] = None,
        categories: Optional[list[str]] = None,
        limit: int = 6,
    ) -> list[LessonCandidate]:
        query_tokens = tokenize(query)
        category_set = set(categories or [])
        candidates: list[LessonCandidate] = []
        for item in self.load()["lessons"]:
            if item.get("archived"):
                continue
            if category_set and item.get("category") not in category_set:
                continue
            scope = item.get("scope")
            if scope == "clip" and item.get("clip_key") != clip_key:
                continue
            score, reasons = self._score(item, query_tokens, clip_key)
            if score > 0:
                candidates.append(LessonCandidate(item=item, score=score, reasons=reasons))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        selected = candidates[: max(1, min(limit, 50))]
        if selected:
            self._touch([candidate.item["id"] for candidate in selected])
        return selected

    def for_clip(self, clip_key: str, *, query: str = "", categories: Optional[list[str]] = None, limit: int = 6) -> dict:
        project = [
            item
            for item in self.list_lessons(limit=100)
            if item.get("scope") == "project"
        ][:limit]
        clip = [
            item
            for item in self.list_lessons(clip_key=clip_key, limit=100)
            if item.get("scope") == "clip" and item.get("clip_key") == clip_key
        ][:limit]
        relevant = [
            {
                **candidate.item,
                "relevance_score": round(candidate.score, 4),
                "relevance_reasons": candidate.reasons,
            }
            for candidate in self.retrieve_lessons(query, clip_key=clip_key, categories=categories, limit=limit)
        ] if query.strip() else []
        return {"project": project, "clip": clip, "relevant": relevant}

    def create_eval_case_from_feedback(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("Eval case creation is planned for a later stage.")

    def mark_feedback_resolved(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("Feedback resolution is handled by prompt feedback endpoints.")

    def _score(self, item: dict, query_tokens: set[str], clip_key: Optional[str]) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        if item.get("scope") == "clip" and item.get("clip_key") == clip_key:
            score += 0.38
            reasons.append("same clip")
        elif item.get("scope") == "project":
            score += 0.22
            reasons.append("project lesson")

        lesson_tokens = tokenize(" ".join([
            item.get("lesson", ""),
            item.get("category", ""),
            " ".join(item.get("positive_examples") or []),
            " ".join(item.get("negative_examples") or []),
        ]))
        if query_tokens and lesson_tokens:
            overlap = len(query_tokens.intersection(lesson_tokens))
            if overlap:
                similarity = overlap / math.sqrt(len(query_tokens) * len(lesson_tokens))
                score += similarity
                reasons.append(f"{overlap} keyword match")

        score += min(float(item.get("confidence", 0.0)), 1.0) * 0.14
        access_count = int(item.get("access_count", 0))
        if access_count:
            score += min(access_count, 8) * 0.015
            reasons.append("reinforced")
        return score, reasons

    def _touch(self, ids: list[str]) -> None:
        data = self.load()
        target_ids = set(ids)
        changed = False
        for item in data["lessons"]:
            if item.get("id") in target_ids:
                item["last_accessed_at"] = now_iso()
                item["access_count"] = int(item.get("access_count", 0)) + 1
                changed = True
        if changed:
            self.save(data)

    def _normalize_lesson(self, item: dict) -> dict:
        lesson = self._clean_lesson_text(str(item.get("lesson") or item.get("text") or ""))
        scope = item.get("scope") if item.get("scope") in {"project", "clip"} else "project"
        clip_key = item.get("clip_key") if scope == "clip" else None
        return {
            "id": str(item.get("id") or uuid.uuid4()),
            "scope": scope,
            "clip_key": clip_key,
            "category": str(item.get("category") or "other"),
            "lesson": lesson,
            "normalized_lesson": normalize_text(lesson),
            "source_feedback_ids": [str(value) for value in item.get("source_feedback_ids") or [] if value],
            "confidence": self._clamp_confidence(item.get("confidence", 0.8)),
            "positive_examples": self._clean_examples(item.get("positive_examples")),
            "negative_examples": self._clean_examples(item.get("negative_examples")),
            "created_at": item.get("created_at") or now_iso(),
            "updated_at": item.get("updated_at") or item.get("created_at") or now_iso(),
            "last_accessed_at": item.get("last_accessed_at"),
            "access_count": int(item.get("access_count") or 0),
            "archived": bool(item.get("archived", False)),
        }

    def _clean_lesson_text(self, lesson: str) -> str:
        cleaned = re.sub(r"\s+", " ", lesson).strip()
        if not cleaned:
            raise ValueError("Lesson text cannot be empty")
        return cleaned

    def _clean_examples(self, examples: Optional[list[str]]) -> list[str]:
        return [re.sub(r"\s+", " ", str(example)).strip() for example in examples or [] if str(example).strip()]

    def _clamp_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except Exception:
            confidence = 0.8
        return max(0.0, min(1.0, confidence))

    def _validate_scope(self, scope: str, clip_key: Optional[str]) -> None:
        if scope not in {"project", "clip"}:
            raise ValueError("Lesson scope must be project or clip")
        if scope == "clip" and not clip_key:
            raise ValueError("clip_key is required for clip lessons")

    def _empty(self) -> dict:
        return {"schema_version": self.schema_version, "lessons": []}
