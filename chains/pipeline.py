"""End-to-end lesson pipeline: RAG -> objectives -> parallel grade content -> timetable -> validate."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import config
import rag
from chains.accessible_chain import generate_accessible_worksheets
from chains.content_chain import generate_grade_contents
from chains.intake_chain import generate_outline
from chains.schemas import AccessibilitySettings, LessonRequest, LessonResult
from chains.timetable_chain import (
    apply_default_texts,
    build_rotation_skeleton,
    fill_timetable_texts,
    teacher_minutes_by_grade,
)
from validators import validate_timetable

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def generate_lesson(req: LessonRequest) -> LessonResult:
    t0 = time.perf_counter()
    warnings: list[str] = []

    # 1. Retrieve curriculum (per grade)
    retrieval = rag.retrieve(req.topic, req.subject, req.grades)
    if retrieval.error:
        warnings.append(retrieval.error)
    if retrieval.store_empty:
        warnings.append("The curriculum library is empty, so no sources were used. "
                        "Upload .txt/.pdf files to ground the lesson.")
    else:
        missing = retrieval.grades_without_sources()
        if missing:
            warnings.append("No relevant curriculum source was retrieved for Grade(s) "
                            f"{', '.join(map(str, missing))}; their content uses general practice.")
    if any(s.is_demo for s in retrieval.sources):
        warnings.append("Some retrieved sources are DEMO curriculum written for this prototype, "
                        "not official syllabus documents.")

    # 2. Fixed rotation skeleton (gives each grade's teacher minutes)
    skeleton = build_rotation_skeleton(req.duration, req.grades, req.language)

    # 3. Objectives (structured output)
    outline, w = generate_outline(req, retrieval)
    warnings += w

    # 4. Grade content in parallel
    contents, w = generate_grade_contents(req, outline, retrieval, teacher_minutes_by_grade(skeleton))
    warnings += w

    # 5. Timetable wording and (optional) accessible worksheets run side by side
    with ThreadPoolExecutor(max_workers=2) as pool:
        tt_future = pool.submit(fill_timetable_texts, skeleton, req, outline, contents)
        acc_future = pool.submit(_safe_accessible, req, req.accessibility, outline, contents, retrieval)
        timetable, w = tt_future.result()
        accessible, w_acc = acc_future.result()
    warnings += w + w_acc
    report = validate_timetable(timetable, req.duration, req.grades)
    if not report.ok:
        log.warning("Timetable invalid (%s); falling back to default wording", report.errors)
        timetable = apply_default_texts(skeleton, outline, contents, req.language, only_empty=False)
        report = validate_timetable(timetable, req.duration, req.grades)

    result = LessonResult(
        lesson_id=uuid.uuid4().hex,
        request=req,
        retrieval=retrieval,
        outline=outline,
        contents=sorted(contents, key=lambda c: c.grade),
        timetable=timetable,
        validation=report,
        warnings=warnings,
        model=config.GROQ_MODEL,
        elapsed_s=round(time.perf_counter() - t0, 1),
        accessible=accessible,
    )
    save_lesson(result)
    return result


def _safe_accessible(req, settings, outline, contents, retrieval):
    """Accessible sheets are optional: a failure becomes a warning, never a lost lesson."""
    try:
        return generate_accessible_worksheets(req, settings, outline, contents, retrieval)
    except Exception as exc:  # noqa: BLE001
        log.exception("Accessible worksheet generation failed")
        return [], [f"Accessible worksheets could not be generated: {str(exc)[:160]}"]


def regenerate_accessible(result: LessonResult, settings: AccessibilitySettings) -> LessonResult:
    """(Re)build accessible worksheets for an existing lesson with new settings.
    Raises on LLM failure so the UI can show the error; the saved lesson is unchanged then."""
    result = result.model_copy(deep=True)
    result.request.accessibility = settings
    if settings.any_enabled:
        sheets, warnings = generate_accessible_worksheets(
            result.request, settings, result.outline, result.contents, result.retrieval)
        if not sheets:
            raise RuntimeError("; ".join(warnings) or "No accessible worksheets were generated.")
        result.accessible = sheets
        result.warnings = [w for w in result.warnings if "accessible" not in w.lower()] + warnings
    else:
        result.accessible = []
    save_lesson(result)
    return result


# --------------------------------------------------------------------------------------
# Teacher edits (inline editing on the results page)
# --------------------------------------------------------------------------------------

EDITABLE_ROOTS = ("outline", "contents", "accessible", "timetable")
_FORBIDDEN_KEYS = {"grade", "start_min", "end_min", "teacher_focus", "mode", "label", "source_labels",
                   "lesson_id", "chunk_id"}


def apply_edits(result: LessonResult, edits: dict[str, Any]) -> tuple[LessonResult, list[str]]:
    """Apply {dotted.path: new_text} edits to string fields only.

    Paths look like ``contents.0.worksheet_tasks.1.task``. Only existing string values
    under the editable roots can change; structure, grades and times cannot.
    Returns (updated_result, errors). The rotation timetable is re-validated afterwards."""
    data = result.model_dump(mode="json")
    errors: list[str] = []
    changed = 0
    for path, value in edits.items():
        parts = str(path).split(".")
        if not parts or parts[0] not in EDITABLE_ROOTS or _FORBIDDEN_KEYS & set(parts):
            errors.append(f"'{path}' cannot be edited.")
            continue
        if not isinstance(value, str):
            errors.append(f"'{path}': value must be text.")
            continue
        value = value.replace("\r\n", "\n").strip()
        if len(value) > 2000:
            errors.append(f"'{path}': text is too long (max 2000 characters).")
            continue
        node: Any = data
        try:
            for key in parts[:-1]:
                node = node[int(key)] if isinstance(node, list) else node[key]
            last = parts[-1]
            idx: Any = int(last) if isinstance(node, list) else last
            current = node[idx]
        except (KeyError, IndexError, ValueError, TypeError):
            errors.append(f"'{path}' does not exist.")
            continue
        if not isinstance(current, str):
            errors.append(f"'{path}' is not a text field.")
            continue
        if not value and parts[0] != "accessible":
            errors.append(f"'{path}' cannot be empty.")
            continue
        if current != value:
            node[idx] = value
            changed += 1

    if changed:
        updated = LessonResult.model_validate(data)
        updated.edited_at = datetime.now()
        updated.validation = validate_timetable(updated.timetable, updated.request.duration,
                                                updated.request.grades)
        save_lesson(updated)
        return updated, errors
    return result, errors


def _lesson_path(lesson_id: str) -> Path:
    if not _ID_RE.match(lesson_id or ""):
        raise ValueError("Invalid lesson id.")
    return config.LESSON_DIR / f"{lesson_id}.json"


def save_lesson(result: LessonResult) -> Path:
    path = _lesson_path(result.lesson_id)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def list_lessons(limit: int | None = None) -> list[dict]:
    """Lightweight summaries of saved lessons, newest first (for the dashboard).
    Reads the JSON directly so one unreadable file never breaks the page."""
    files = sorted(config.LESSON_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        if not _ID_RE.match(f.stem):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            req = data.get("request", {})
            out.append({
                "lesson_id": f.stem,
                "title": data.get("outline", {}).get("lesson_title") or req.get("topic", "Lesson"),
                "topic": req.get("topic", ""),
                "subject": req.get("subject", ""),
                "grades": req.get("grades", []),
                "language": req.get("language", "English"),
                "created_at": data.get("created_at", "")[:16].replace("T", " "),
                "accessible": bool(data.get("accessible")),
            })
        except (OSError, ValueError, AttributeError):
            log.warning("Skipping unreadable lesson file %s", f.name)
        if limit and len(out) >= limit:
            break
    return out


def count_lessons() -> int:
    return sum(1 for f in config.LESSON_DIR.glob("*.json") if _ID_RE.match(f.stem))


def load_lesson(lesson_id: str) -> LessonResult | None:
    try:
        path = _lesson_path(lesson_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return LessonResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        log.exception("Could not load lesson %s", lesson_id)
        return None
