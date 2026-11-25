"""Lesson intake -> grade-specific, syllabus-aligned objectives (structured output)."""
from __future__ import annotations

import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from chains.llm import get_llm, grade_guide, language_instruction, structured
from chains.schemas import GradeObjectives, LessonPlanOutline, LessonRequest, RetrievalResult

log = logging.getLogger(__name__)

SYSTEM = """You are an expert multi-grade primary teacher-trainer for rural government schools in India.
ONE teacher teaches Grades {grade_list} together in ONE room with few resources (blackboard, chalk, slates, notebooks, and local objects such as stones, seeds, sticks, leaves and old paper).

Your job: plan ONE shared topic so every grade works on the same theme at its own level, and write grade-specific, syllabus-aligned learning objectives.

Rules:
- Ground objectives in the CURRICULUM CONTEXT. In curriculum_alignment, cite the source labels you used, like [S2].
- Only cite labels that appear in the context. Never invent textbook names, chapter numbers, page numbers or official outcome codes.
- If the context says no source was retrieved for a grade, write in curriculum_alignment that no curriculum source was retrieved and the objectives follow general grade-level practice.
- Sources marked DEMO MATERIAL are sample content; you may use them, but do not call them official.
- Objectives are measurable, start with a verb, 2-3 per grade, and clearly progress in difficulty from lower to higher grades.
- The `grades` list must contain exactly one entry for EACH of these grades: {grade_list}.
- {language_instruction}"""

HUMAN = """Topic: {topic}
Subject: {subject}
Class duration: {duration} minutes
Selected grades: {grade_list}

Grade level guide:
{grade_guide}

CURRICULUM CONTEXT (retrieved):
{context}
{reminder}"""

intake_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])

_LABEL_RE = re.compile(r"\[(S\d+)\]")


def build_intake_chain():
    """LCEL: prompt | Groq with structured output -> LessonPlanOutline."""
    return intake_prompt | structured(LessonPlanOutline)


def _inputs(req: LessonRequest, retrieval: RetrievalResult, reminder: str = "") -> dict:
    return {
        "topic": req.topic,
        "subject": req.subject,
        "duration": req.duration,
        "grade_list": ", ".join(str(g) for g in req.grades),
        "grade_guide": grade_guide(req.grades),
        "context": retrieval.context_for_all(),
        "language_instruction": language_instruction(req.language),
        "reminder": reminder,
    }


def _strip_invalid_labels(text: str, valid: set[str]) -> str:
    """Remove citations to labels that were never retrieved (no invented citations)."""
    return _LABEL_RE.sub(lambda m: m.group(0) if m.group(1) in valid else "", text).strip()


def _normalize(outline: LessonPlanOutline, req: LessonRequest,
               retrieval: RetrievalResult) -> tuple[LessonPlanOutline, list[int]]:
    by_grade: dict[int, GradeObjectives] = {}
    for g in outline.grades:
        if g.grade in req.grades and g.grade not in by_grade:
            by_grade[g.grade] = g
    missing = [g for g in req.grades if g not in by_grade]

    for grade, obj in by_grade.items():
        valid = set(retrieval.labels_for_grade(grade)) or {s.label for s in retrieval.sources}
        obj.curriculum_alignment = _strip_invalid_labels(obj.curriculum_alignment, valid)
        if not retrieval.labels_for_grade(grade):
            obj.curriculum_alignment = (
                "No curriculum source retrieved for this grade - objectives follow general "
                "grade-level practice. " + obj.curriculum_alignment
            ).strip()
        obj.objectives = [o.strip() for o in obj.objectives if o and o.strip()][:4]

    outline.grades = [by_grade[g] for g in sorted(by_grade)]
    return outline, missing


def generate_outline(req: LessonRequest,
                     retrieval: RetrievalResult) -> tuple[LessonPlanOutline, list[str]]:
    """Return (outline, warnings). Retries once if grades are missing."""
    warnings: list[str] = []
    chain = build_intake_chain()

    outline = chain.invoke(_inputs(req, retrieval))
    if outline is None:
        raise RuntimeError("The model did not return a lesson outline.")
    outline, missing = _normalize(outline, req, retrieval)

    if missing:
        log.info("Outline missing grades %s; retrying once", missing)
        reminder = (f"\nIMPORTANT: your previous answer was missing Grades "
                    f"{', '.join(map(str, missing))}. Include every selected grade.")
        retry = chain.invoke(_inputs(req, retrieval, reminder))
        if retry is not None:
            retry, missing_retry = _normalize(retry, req, retrieval)
            if len(missing_retry) < len(missing):
                outline, missing = retry, missing_retry

    for g in missing:
        warnings.append(f"Objectives for Grade {g} were not generated; a placeholder is shown.")
        outline.grades.append(GradeObjectives(
            grade=g,
            objectives=[f"Explore '{req.topic}' at a Grade {g} level (teacher to refine)."],
            success_criteria="Teacher to define.",
            key_vocabulary=[],
            curriculum_alignment="Not generated - please review.",
        ))
    outline.grades.sort(key=lambda o: o.grade)
    return outline, warnings
