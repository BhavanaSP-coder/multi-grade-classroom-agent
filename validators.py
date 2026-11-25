"""Input validation (form + uploads) and the teacher-rotation timetable validator."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import config
from chains.schemas import (
    ACCESSIBILITY_OPTIONS,
    AccessibilitySettings,
    LessonRequest,
    Timetable,
    ValidationReport,
)

ATTENTION_HINTS = ("Very High", "High", "Moderate", "Low", "Very Low")

# --------------------------------------------------------------------------------------
# Timetable validator
# --------------------------------------------------------------------------------------


def validate_timetable(tt: Timetable, duration: int, grades: Iterable[int]) -> ValidationReport:
    """Check that blocks cover [0, duration] exactly, with no gaps or overlaps, and that
    every selected grade has exactly one non-empty assignment in every block."""
    grades = sorted(set(grades))
    errors: list[str] = []
    warnings: list[str] = []
    blocks = tt.blocks

    if not blocks:
        return ValidationReport(ok=False, errors=["Timetable has no blocks."])

    ordered = sorted(blocks, key=lambda b: (b.start_min, b.end_min))
    if [id(b) for b in ordered] != [id(b) for b in blocks]:
        warnings.append("Blocks were not in chronological order.")

    if ordered[0].start_min != 0:
        errors.append(f"First block starts at {ordered[0].start_min} min, expected 0.")
    if ordered[-1].end_min != duration:
        errors.append(f"Last block ends at {ordered[-1].end_min} min, expected {duration}.")

    prev_end = None
    for i, b in enumerate(ordered):
        name = f"Block {i + 1} ({b.start_min}-{b.end_min})"
        if b.end_min <= b.start_min:
            errors.append(f"{name}: end time must be after start time.")
        if b.start_min < 0 or b.end_min > duration:
            errors.append(f"{name}: outside the class duration.")
        if prev_end is not None:
            if b.start_min > prev_end:
                errors.append(f"Gap of {b.start_min - prev_end} min before {name}.")
            elif b.start_min < prev_end:
                errors.append(f"{name} overlaps the previous block by {prev_end - b.start_min} min.")
        prev_end = b.end_min

        bad_focus = [g for g in b.teacher_focus if g not in grades]
        if bad_focus:
            errors.append(f"{name}: teacher assigned to unselected grade(s) {bad_focus}.")

        seen: dict[int, int] = {}
        for a in b.assignments:
            seen[a.grade] = seen.get(a.grade, 0) + 1
            if a.grade not in grades:
                errors.append(f"{name}: assignment for unselected Grade {a.grade}.")
            if not a.task or not a.task.strip():
                errors.append(f"{name}: Grade {a.grade} has an empty task.")
            expected_mode = (
                "whole_class" if not b.teacher_focus
                else "with_teacher" if a.grade in b.teacher_focus else "independent"
            )
            if a.mode != expected_mode:
                errors.append(f"{name}: Grade {a.grade} mode is '{a.mode}', expected '{expected_mode}'.")
        for g in grades:
            if seen.get(g, 0) == 0:
                errors.append(f"{name}: Grade {g} has no work assigned.")
            elif seen[g] > 1:
                errors.append(f"{name}: Grade {g} is assigned {seen[g]} times.")

    total = sum(max(0, b.minutes) for b in ordered)
    if total != duration and not errors:
        errors.append(f"Blocks add up to {total} min, expected {duration}.")

    taught = {g for b in ordered for g in b.teacher_focus}
    for g in grades:
        if g not in taught and len(ordered) > 1:
            warnings.append(f"Grade {g} never gets dedicated teacher time.")

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)


# --------------------------------------------------------------------------------------
# Form validation
# --------------------------------------------------------------------------------------


def validate_lesson_form(form: Any) -> tuple[LessonRequest | None, list[str], dict]:
    """Validate the lesson form (a Flask ``request.form`` or a dict).

    Returns (request_or_None, errors, cleaned_values_for_redisplay)."""
    errors: list[str] = []
    field_errors: dict[str, str] = {}

    def err(field: str, message: str) -> None:
        errors.append(message)
        field_errors.setdefault(field, message)

    def get(name: str) -> str:
        return (form.get(name) or "").strip()

    def getlist(name: str) -> list[str]:
        if hasattr(form, "getlist"):
            return form.getlist(name)
        val = form.get(name, [])
        return val if isinstance(val, list) else [val]

    topic = " ".join(get("topic").split())
    subject = get("subject")
    language = get("language") or "English"
    duration_raw = get("duration")

    if not topic:
        err("topic", "Please enter a topic (for example: Fractions).")
    elif len(topic) < 2:
        err("topic", "Topic is too short.")
    elif len(topic) > config.TOPIC_MAX_LEN:
        err("topic", f"Topic must be at most {config.TOPIC_MAX_LEN} characters.")
    elif not any(ch.isalpha() for ch in topic):
        err("topic", "Topic must contain letters.")

    if subject not in config.SUBJECTS:
        err("subject", "Please choose a subject from the list.")

    grades: list[int] = []
    for g in getlist("grades"):
        try:
            gi = int(str(g).strip())
        except ValueError:
            err("grades", f"Invalid grade value: {g!r}.")
            continue
        if gi not in config.SUPPORTED_GRADES:
            err("grades", f"Grade {gi} is not supported (choose 1-5).")
        elif gi not in grades:
            grades.append(gi)
    grades.sort()
    if not grades:
        err("grades", "Select at least one grade.")

    duration = 0
    try:
        duration = int(duration_raw)
        if not config.MIN_DURATION <= duration <= config.MAX_DURATION:
            err("duration", 
                f"Class duration must be between {config.MIN_DURATION} and "
                f"{config.MAX_DURATION} minutes."
            )
    except (TypeError, ValueError):
        err("duration", "Class duration must be a whole number of minutes.")

    if language not in config.LANGUAGES:
        err("language", "Please choose English, Kannada or Hindi.")

    accessibility = parse_accessibility(form)
    attention = get("attention_hint")
    if attention and attention not in ATTENTION_HINTS:
        err("attention_hint", "Unknown attention level for this period.")
        attention = ""

    values = {"topic": topic, "subject": subject, "grades": grades,
              "duration": duration_raw, "language": language,
              "accessibility": accessibility, "attention_hint": attention,
              "field_errors": field_errors}
    if errors:
        return None, errors, values
    return LessonRequest(topic=topic, subject=subject, grades=grades,
                         duration=duration, language=language,
                         accessibility=accessibility, attention_hint=attention), [], values


def parse_accessibility(form: Any) -> AccessibilitySettings:
    """Read 'Worksheet Accessibility Settings' from a form.

    ``acc_preset`` = off | dyslexia | custom. With "dyslexia" the recommended preset is
    applied, then any individual box the teacher also ticked (e.g. reduce_questions) is added."""
    preset = (form.get("acc_preset") or "off").strip()
    ticked = {k for k, _, _ in ACCESSIBILITY_OPTIONS if _truthy(form.get(f"acc_{k}"))}
    if preset == "dyslexia":
        base = AccessibilitySettings.dyslexia_preset()
        return base.model_copy(update={k: True for k in ticked})
    if preset == "custom" and ticked:
        return AccessibilitySettings(preset="custom", **{k: True for k in ticked})
    return AccessibilitySettings()


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "on", "true", "yes")


# --------------------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------------------


def validate_upload(filename: str, data: bytes) -> list[str]:
    """Validate an uploaded curriculum file's name and bytes."""
    errors: list[str] = []
    if not filename:
        return ["Please choose a file to upload."]
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return [f"'{filename}' is not supported. Upload a .pdf or .txt file."]
    if not data:
        return [f"'{filename}' is empty."]
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return [f"'{filename}' is larger than {config.MAX_UPLOAD_MB} MB."]

    if ext == ".pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            errors.append(f"'{filename}' does not look like a valid PDF file.")
    else:
        text = None
        encodings = ("utf-16",) if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1252")
        for enc in encodings:
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            errors.append(f"'{filename}' is not readable text. Save it as UTF-8.")
        elif len(text.strip()) < 30:
            errors.append(f"'{filename}' has too little text to be useful (min 30 characters).")
        elif "\x00" in text:
            errors.append(f"'{filename}' looks like a binary file, not plain text.")
    return errors
