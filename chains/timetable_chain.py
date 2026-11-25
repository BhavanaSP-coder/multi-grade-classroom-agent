"""Teacher rotation timetable.

Two stages, so the result is always valid:
1. ``build_rotation_skeleton`` (pure Python) fixes the time blocks and who the teacher is with.
   It handles durations / grade counts that do not divide evenly.
2. ``fill_timetable_texts`` asks the LLM (LCEL + structured output) to word each grade's task
   per block. Times and modes come from the skeleton, never from the LLM; any missing text
   falls back to deterministic wording built from the generated grade content.
"""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

import config
from chains.llm import get_llm, language_instruction, structured
from chains.schemas import (
    GradeAssignment,
    GradeContent,
    LessonPlanOutline,
    LessonRequest,
    TimeBlock,
    Timetable,
    TimetableText,
)

log = logging.getLogger(__name__)

MIN_TEACH_BLOCK = 5   # minutes; shorter teacher slots are not useful
MAX_TEACH_BLOCK = 25  # longer than this -> do a second rotation round

LABELS = {
    "English": {"teacher": "With teacher", "ws": "Worksheet", "quiz": "Quiz",
                "start": "tasks 1-2 on your own", "finish": "finish all tasks, then",
                "opening": "Whole-class opening", "closing": "Whole-class closing",
                "rotation": "Rotation"},
    "Hindi": {"teacher": "शिक्षक के साथ", "ws": "कार्यपत्रक", "quiz": "प्रश्नोत्तरी",
              "start": "प्रश्न 1-2 स्वयं करें", "finish": "सभी प्रश्न पूरे करें, फिर",
              "opening": "पूरी कक्षा - शुरुआत", "closing": "पूरी कक्षा - समापन",
              "rotation": "बारी"},
    "Kannada": {"teacher": "ಶಿಕ್ಷಕರೊಂದಿಗೆ", "ws": "ಕಾರ್ಯಹಾಳೆ", "quiz": "ರಸಪ್ರಶ್ನೆ",
                "start": "ಪ್ರಶ್ನೆ 1-2 ಸ್ವತಃ ಮಾಡಿ", "finish": "ಎಲ್ಲಾ ಪ್ರಶ್ನೆಗಳನ್ನು ಮುಗಿಸಿ, ನಂತರ",
                "opening": "ಇಡೀ ತರಗತಿ - ಆರಂಭ", "closing": "ಇಡೀ ತರಗತಿ - ಮುಕ್ತಾಯ",
                "rotation": "ಸರದಿ"},
}


# --------------------------------------------------------------------------------------
# Stage 1: deterministic skeleton
# --------------------------------------------------------------------------------------


def split_evenly(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` integers that differ by at most 1 (larger first)."""
    parts = max(1, parts)
    base, rem = divmod(total, parts)
    return [base + 1 if i < rem else base for i in range(parts)]


def segment_lengths(duration: int) -> tuple[int, int, int]:
    """(opening, rotation, closing) minutes. Short classes skip opening/closing."""
    opening = min(10, max(5, round(duration * 0.1))) if duration >= 25 else 0
    closing = min(10, max(5, round(duration * 0.1))) if duration >= 30 else 0
    middle = duration - opening - closing
    if middle < MIN_TEACH_BLOCK:  # defensive; cannot happen with MIN_DURATION >= 15
        return 0, duration, 0
    return opening, middle, closing


def build_rotation_skeleton(duration: int, grades: list[int], language: str = "English") -> Timetable:
    """Blocks covering [0, duration] exactly. Teacher starts with the youngest grade
    (older children can start independent work sooner). Remainder minutes go to the
    earliest (youngest) slots. If slots would be < 5 min, adjacent grades are paired."""
    grades = sorted(set(grades))
    if not grades:
        raise ValueError("At least one grade is required.")
    lab = LABELS.get(language, LABELS["English"])
    opening, middle, closing = segment_lengths(duration)

    n = len(grades)
    n_groups = max(1, min(n, middle // MIN_TEACH_BLOCK))
    groups, i = [], 0
    for size in split_evenly(n, n_groups):
        groups.append(grades[i:i + size])
        i += size

    rounds = 1
    if n > 1 and middle / len(groups) > MAX_TEACH_BLOCK and middle // (2 * len(groups)) >= MIN_TEACH_BLOCK:
        rounds = 2
    slots = groups * rounds

    blocks: list[TimeBlock] = []
    t = 0

    def add(length: int, focus: list[int], title: str) -> None:
        nonlocal t
        if length <= 0:
            return
        if not focus:
            assigns = [GradeAssignment(grade=g, mode="whole_class", task="") for g in grades]
        else:
            assigns = [GradeAssignment(grade=g, mode="with_teacher" if g in focus else "independent",
                                       task="") for g in grades]
        blocks.append(TimeBlock(start_min=t, end_min=t + length, teacher_focus=list(focus),
                                title=title, assignments=assigns))
        t += length

    add(opening, [], lab["opening"])
    for idx, (focus, length) in enumerate(zip(slots, split_evenly(middle, len(slots))), start=1):
        add(length, focus, f"{lab['rotation']} {idx}")
    add(closing, [], lab["closing"])
    return Timetable(blocks=blocks)


def teacher_minutes_by_grade(tt: Timetable) -> dict[int, int]:
    """Length of each grade's first dedicated teacher slot (used to size the activity)."""
    out: dict[int, int] = {}
    for b in tt.blocks:
        for g in b.teacher_focus:
            out.setdefault(g, b.minutes)
    return out


# --------------------------------------------------------------------------------------
# Deterministic fallback text
# --------------------------------------------------------------------------------------


def apply_default_texts(tt: Timetable, outline: LessonPlanOutline | None,
                        contents: list[GradeContent], language: str = "English",
                        only_empty: bool = True) -> Timetable:
    lab = LABELS.get(language, LABELS["English"])
    by_grade = {c.grade: c for c in contents}
    taught: set[int] = set()
    tt = tt.model_copy(deep=True)
    for b in tt.blocks:
        for a in b.assignments:
            if only_empty and a.task.strip():
                continue
            c = by_grade.get(a.grade)
            if a.mode == "whole_class":
                if b.start_min == 0:
                    a.task = (outline.whole_class_opening if outline else "") or lab["opening"]
                else:
                    a.task = (outline.whole_class_closing if outline else "") or lab["closing"]
            elif a.mode == "with_teacher":
                a.task = f"{lab['teacher']}: {c.activity_title if c else ''}".strip(": ")
            else:
                ws = c.worksheet_title if c else lab["ws"]
                if a.grade in taught:
                    a.task = f"{lab['ws']} \"{ws}\": {lab['finish']} {lab['quiz']}"
                else:
                    a.task = f"{lab['ws']} \"{ws}\": {lab['start']}"
        taught.update(b.teacher_focus)
    return tt


# --------------------------------------------------------------------------------------
# Stage 2: LLM wording (LCEL)
# --------------------------------------------------------------------------------------

SYSTEM = """You write the task wording for a multi-grade classroom rotation timetable. One teacher, Grades {grade_list}, one room, low resources.
The time blocks and which grade is WITH the teacher are FIXED. Do not change them.

For EVERY block and EVERY grade, write one short, concrete sentence (max 20 words) saying what that grade does:
- with_teacher: the teacher-led activity for that grade (use its activity title / steps).
- independent: work the grade can do ALONE: their worksheet, then their quiz, or peer help (an older grade may help a younger one). Before a grade's first teacher slot, give an easy start such as the first worksheet tasks or a review; after it, finishing the worksheet and then the quiz.
- whole_class: the shared opening or closing that all grades do together.
Also give 2-3 practical tips for managing the rotation (silent signals, monitors, peer helpers).
{language_instruction}"""

HUMAN = """Lesson: {lesson_title}
Whole-class opening idea: {opening}
Whole-class closing idea: {closing}

GRADE MATERIALS:
{materials}

FIXED SKELETON (block_index | time | teacher with | grade: mode):
{skeleton}"""

timetable_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


def build_timetable_chain():
    return timetable_prompt | structured(TimetableText)


def _describe_skeleton(tt: Timetable) -> str:
    lines = []
    for i, b in enumerate(tt.blocks):
        modes = ", ".join(f"G{a.grade}: {a.mode}" for a in b.assignments)
        lines.append(f"{i} | {b.start_min}-{b.end_min} min | {b.teacher_label} | {modes}")
    return "\n".join(lines)


def _describe_materials(contents: list[GradeContent]) -> str:
    return "\n".join(
        f"- Grade {c.grade}: activity '{c.activity_title}'; worksheet '{c.worksheet_title}' "
        f"({len(c.worksheet_tasks)} tasks); quiz ({len(c.quiz)} questions)"
        for c in contents
    )


def fill_timetable_texts(skeleton: Timetable, req: LessonRequest, outline: LessonPlanOutline,
                         contents: list[GradeContent]) -> tuple[Timetable, list[str]]:
    """Return (timetable with every task filled, warnings)."""
    warnings: list[str] = []
    tt = skeleton.model_copy(deep=True)

    if config.TIMETABLE_LLM:
        try:
            text: TimetableText | None = build_timetable_chain().invoke({
                "grade_list": ", ".join(map(str, req.grades)),
                "language_instruction": language_instruction(req.language),
                "lesson_title": outline.lesson_title,
                "opening": outline.whole_class_opening,
                "closing": outline.whole_class_closing,
                "materials": _describe_materials(contents),
                "skeleton": _describe_skeleton(tt),
            })
            if text is not None:
                lookup = {(bt.block_index, t.grade): t.task.strip()
                          for bt in text.blocks for t in bt.tasks if t.task and t.task.strip()}
                for i, b in enumerate(tt.blocks):
                    for a in b.assignments:
                        a.task = lookup.get((i, a.grade), "")
                tt.teacher_tips = [tip.strip() for tip in text.teacher_tips if tip.strip()][:4]
                missing = sum(1 for b in tt.blocks for a in b.assignments if not a.task)
                if missing:
                    warnings.append(f"Timetable: {missing} task(s) filled with default wording.")
        except Exception as exc:  # noqa: BLE001 - timetable must never block the lesson
            log.warning("Timetable LLM wording failed: %s", exc)
            warnings.append("Timetable wording used built-in text (AI step failed).")

    tt = apply_default_texts(tt, outline, contents, req.language, only_empty=True)
    if not tt.teacher_tips:
        tt.teacher_tips = [
            "Agree on a silent signal (raised hand / clap) for 'I need help' during independent work.",
            "Make one older student a monitor who helps younger grades with instructions.",
            "Write each grade's independent task on its own section of the blackboard before class.",
        ]
    return tt, warnings
