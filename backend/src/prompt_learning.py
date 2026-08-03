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


@dataclass
class EvalCaseCandidate:
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


class PromptEvalCaseStore:
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
        cases = data.get("cases")
        if not isinstance(cases, list):
            cases = []
        return {
            "schema_version": self.schema_version,
            "cases": [self._normalize_case(item) for item in cases if isinstance(item, dict)],
        }

    def save(self, data: dict) -> None:
        data.setdefault("schema_version", self.schema_version)
        data.setdefault("cases", [])
        atomic_write_json(self.path, data)

    def add_eval_case(
        self,
        *,
        name: str,
        source_feedback_id: Optional[str],
        clip_index: int,
        clip_key: Optional[str],
        input: dict,
        expected_behavior: list[str],
        failure_categories: list[str],
        enabled: bool = True,
    ) -> dict:
        cleaned_expected = self._clean_list(expected_behavior)
        if not cleaned_expected:
            raise ValueError("expected_behavior must include at least one item")
        created_at = now_iso()
        item = {
            "id": str(uuid.uuid4()),
            "name": self._clean_name(name),
            "source_feedback_id": source_feedback_id,
            "clip_index": int(clip_index),
            "clip_key": clip_key,
            "input": self._normalize_input(input),
            "expected_behavior": cleaned_expected,
            "failure_categories": self._clean_list(failure_categories) or ["other"],
            "enabled": bool(enabled),
            "created_at": created_at,
            "updated_at": created_at,
        }
        data = self.load()
        data["cases"].append(item)
        self.save(data)
        return item

    def add_eval_case_from_feedback(
        self,
        feedback_item: dict,
        *,
        clip_summary: str = "",
        selected_assets: Optional[list[str]] = None,
        feedback_items: Optional[list[dict]] = None,
    ) -> dict:
        expected = self._expected_behavior_from_feedback(feedback_item)
        categories = self._clean_list(feedback_item.get("categories") or []) or ["other"]
        name = self._name_from_feedback(categories, expected)
        return self.add_eval_case(
            name=name,
            source_feedback_id=feedback_item.get("id"),
            clip_index=int(feedback_item.get("clip_index") or 0),
            clip_key=feedback_item.get("clip_key"),
            input={
                "feedback_items": feedback_items or [self._feedback_excerpt(feedback_item)],
                "clip_summary": clip_summary or "",
                "selected_assets": selected_assets or [],
            },
            expected_behavior=expected,
            failure_categories=categories,
            enabled=True,
        )

    def list_eval_cases(
        self,
        *,
        clip_index: Optional[int] = None,
        enabled: Optional[bool] = None,
        limit: int = 100,
    ) -> list[dict]:
        cases = []
        for item in self.load()["cases"]:
            if clip_index is not None and item.get("clip_index") != clip_index:
                continue
            if enabled is not None and bool(item.get("enabled", True)) is not enabled:
                continue
            cases.append(item)
        cases.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return cases[: max(1, min(limit, 500))]

    def update_eval_case(self, case_id: str, updates: dict) -> dict:
        data = self.load()
        item = next((candidate for candidate in data["cases"] if candidate.get("id") == case_id), None)
        if item is None:
            raise KeyError(case_id)
        if "name" in updates:
            item["name"] = self._clean_name(str(updates.get("name") or ""))
        if "input" in updates:
            item["input"] = self._normalize_input(updates.get("input") or {})
        if "expected_behavior" in updates:
            expected = self._clean_list(updates.get("expected_behavior") or [])
            if not expected:
                raise ValueError("expected_behavior must include at least one item")
            item["expected_behavior"] = expected
        if "failure_categories" in updates:
            item["failure_categories"] = self._clean_list(updates.get("failure_categories") or []) or ["other"]
        if "enabled" in updates:
            item["enabled"] = bool(updates.get("enabled"))
        item["updated_at"] = now_iso()
        self.save(data)
        return item

    def retrieve_eval_cases(
        self,
        query: str,
        *,
        clip_key: Optional[str] = None,
        clip_index: Optional[int] = None,
        limit: int = 6,
    ) -> list[EvalCaseCandidate]:
        query_tokens = tokenize(query)
        candidates: list[EvalCaseCandidate] = []
        for item in self.load()["cases"]:
            if not item.get("enabled", True):
                continue
            score, reasons = self._score_case(item, query_tokens, clip_key, clip_index)
            if score > 0:
                candidates.append(EvalCaseCandidate(item=item, score=score, reasons=reasons))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[: max(1, min(limit, 50))]

    def _score_case(
        self,
        item: dict,
        query_tokens: set[str],
        clip_key: Optional[str],
        clip_index: Optional[int],
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        if clip_key and item.get("clip_key") == clip_key:
            score += 0.42
            reasons.append("same clip")
        elif clip_index is not None and item.get("clip_index") == clip_index:
            score += 0.28
            reasons.append("same clip index")
        case_tokens = tokenize(" ".join([
            item.get("name", ""),
            " ".join(item.get("expected_behavior") or []),
            " ".join(item.get("failure_categories") or []),
            normalize_text(json.dumps(item.get("input") or {}, ensure_ascii=False)),
        ]))
        if query_tokens and case_tokens:
            overlap = len(query_tokens.intersection(case_tokens))
            if overlap:
                score += overlap / math.sqrt(len(query_tokens) * len(case_tokens))
                reasons.append(f"{overlap} keyword match")
        return score, reasons

    def _normalize_case(self, item: dict) -> dict:
        return {
            "id": str(item.get("id") or uuid.uuid4()),
            "name": self._clean_name(str(item.get("name") or "Prompt eval case")),
            "source_feedback_id": item.get("source_feedback_id"),
            "clip_index": int(item.get("clip_index") or 0),
            "clip_key": item.get("clip_key"),
            "input": self._normalize_input(item.get("input") or {}),
            "expected_behavior": self._clean_list(item.get("expected_behavior") or []) or ["Address the saved feedback explicitly."],
            "failure_categories": self._clean_list(item.get("failure_categories") or []) or ["other"],
            "enabled": bool(item.get("enabled", True)),
            "created_at": item.get("created_at") or now_iso(),
            "updated_at": item.get("updated_at") or item.get("created_at") or now_iso(),
        }

    def _normalize_input(self, value: dict) -> dict:
        return {
            "feedback_items": value.get("feedback_items") if isinstance(value.get("feedback_items"), list) else [],
            "clip_summary": str(value.get("clip_summary") or ""),
            "selected_assets": value.get("selected_assets") if isinstance(value.get("selected_assets"), list) else [],
        }

    def _expected_behavior_from_feedback(self, feedback_item: dict) -> list[str]:
        values = self._clean_list([
            feedback_item.get("correction"),
            feedback_item.get("remember_note"),
            feedback_item.get("comment"),
        ])
        if not values:
            values = ["Address this feedback explicitly without introducing unsupported visual details."]
        return values[:5]

    def _feedback_excerpt(self, feedback_item: dict) -> dict:
        return {
            "rating": feedback_item.get("rating"),
            "categories": feedback_item.get("categories") or [],
            "severity": feedback_item.get("severity"),
            "comment": feedback_item.get("comment") or "",
            "correction": feedback_item.get("correction") or "",
            "remember_note": feedback_item.get("remember_note") or "",
        }

    def _name_from_feedback(self, categories: list[str], expected: list[str]) -> str:
        category = (categories or ["prompt"])[0].replace("_", " ")
        first_expected = (expected or ["feedback"])[0]
        words = re.findall(r"[A-Za-z0-9'-]+", first_expected)[:5]
        suffix = " ".join(words) if words else "feedback"
        return f"{category.title()} - {suffix}"

    def _clean_name(self, name: str) -> str:
        cleaned = re.sub(r"\s+", " ", name).strip()
        if not cleaned:
            raise ValueError("Eval case name cannot be empty")
        return cleaned[:120]

    def _clean_list(self, values: Optional[list[Any]]) -> list[str]:
        return [re.sub(r"\s+", " ", str(value)).strip() for value in values or [] if str(value).strip()]

    def _empty(self) -> dict:
        return {"schema_version": self.schema_version, "cases": []}
