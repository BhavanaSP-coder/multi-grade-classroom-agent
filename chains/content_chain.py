"""Per-grade content generation: activity + worksheet + quiz, run in parallel with LCEL batch."""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

import config
from chains.llm import GRADE_LEVEL_GUIDE, get_llm, language_instruction, structured
from chains.schemas import (
    GradeContent,
    LessonPlanOutline,
    LessonRequest,
    QuizQuestion,
    RetrievalResult,
    WorksheetTask,
)

log = logging.getLogger(__name__)

SYSTEM = """You are an expert multi-grade primary teacher in a rural Indian government school.
You are preparing materials for GRADE {grade} only. Level: {grade_level}
The teacher teaches several grades in one room, so while this grade works alone the teacher is busy with other grades.

Create:
1. ACTIVITY - a low-resource hands-on or blackboard activity the teacher leads with this grade for about {teach_minutes} minutes. Use only free or local materials (stones, seeds, sticks, leaves, old paper, chalk, slates). 4-6 short steps in order.
2. WORKSHEET - 3 or 4 tasks the children can do completely ALONE on a slate or notebook, without the teacher. One simple instruction line. Give the correct answer for each task.
3. QUIZ - exactly 3 short questions with the correct answers. For Grades 1-2 prefer oral, picture or choose-one questions.

Rules:
- Ground everything in the CURRICULUM CONTEXT for this grade. Put the labels you used (like "S1") in source_labels. If no source was retrieved, use [] and rely on general age-appropriate practice.
- Never invent textbook names, chapters, page numbers or official codes.
- Match numbers, vocabulary and sentence length to Grade {grade}. Use familiar village and daily-life examples (farm, market, food, festivals).
- No materials that cost money (no printers, scissors kits, projectors).
- {language_instruction}"""

HUMAN = """Topic: {topic}
Subject: {subject}
Shared lesson: {lesson_title} - {big_idea}

Learning objectives for Grade {grade}:
{objectives}
Key vocabulary: {vocabulary}

CURRICULUM CONTEXT FOR GRADE {grade}:
{context}
{attention_note}"""

ATTENTION_NOTES = {
    "Very High": "This period is one the teacher observed as VERY HIGH attention: deeper reasoning tasks suit it well.",
    "High": "This period is one the teacher observed as HIGH attention: concept-heavy work suits it.",
    "Moderate": "This period is one the teacher observed as MODERATE attention: mix explanation with short hands-on steps.",
    "Low": "This period is one the teacher observed as LOW attention: favour movement, objects, games and short tasks (keep the same objectives).",
    "Very Low": "This period is one the teacher observed as VERY LOW attention: keep tasks short, active and playful (keep the same objectives).",
}

content_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


def build_content_chain():
    """LCEL: prompt | Groq with structured output -> GradeContent."""
    return content_prompt | structured(GradeContent)


def _inputs(grade: int, req: LessonRequest, outline: LessonPlanOutline,
            retrieval: RetrievalResult, teach_minutes: int) -> dict:
    obj = outline.objectives_for(grade)
    objectives = "\n".join(f"- {o}" for o in (obj.objectives if obj else [])) or "- (not available)"
    return {
        "grade": grade,
        "grade_level": GRADE_LEVEL_GUIDE.get(grade, f"Grade {grade}"),
        "teach_minutes": max(teach_minutes, 3),
        "topic": req.topic,
        "subject": req.subject,
        "lesson_title": outline.lesson_title,
        "big_idea": outline.big_idea,
        "objectives": objectives,
        "vocabulary": ", ".join(obj.key_vocabulary) if obj and obj.key_vocabulary else "-",
        "context": retrieval.context_for_grade(grade),
        "language_instruction": language_instruction(req.language),
        "attention_note": ("\nCLASS PERIOD: " + ATTENTION_NOTES[req.attention_hint])
        if req.attention_hint in ATTENTION_NOTES else "",
    }


def _placeholder(grade: int, req: LessonRequest) -> GradeContent:
    return GradeContent(
        grade=grade,
        activity_title=f"{req.topic} - teacher-led discussion",
        activity_type="blackboard",
        materials=["Blackboard", "Chalk"],
        activity_steps=["Content could not be generated automatically. Please try again."],
        worksheet_title=f"{req.topic} practice",
        worksheet_instructions="Teacher to provide tasks.",
        worksheet_tasks=[WorksheetTask(task="(not generated)", answer="-")],
        quiz=[QuizQuestion(question="(not generated)", options=[], answer="-")],
        teacher_tip="Generation failed for this grade; regenerate or prepare manually.",
        source_labels=[],
    )


def _normalize(content: GradeContent, grade: int, retrieval: RetrievalResult,
               warnings: list[str]) -> GradeContent:
    content.grade = grade
    valid = set(retrieval.labels_for_grade(grade))
    cleaned = [lbl.strip().strip("[]") for lbl in content.source_labels]
    content.source_labels = [lbl for lbl in dict.fromkeys(cleaned) if lbl in valid]
    content.worksheet_tasks = [t for t in content.worksheet_tasks if t.task.strip()][:4]
    content.quiz = [q for q in content.quiz if q.question.strip()][:3]
    if len(content.worksheet_tasks) < 3:
        warnings.append(f"Grade {grade}: worksheet has only {len(content.worksheet_tasks)} task(s).")
    if len(content.quiz) < 3:
        warnings.append(f"Grade {grade}: quiz has only {len(content.quiz)} question(s).")
    return content


def generate_grade_contents(req: LessonRequest, outline: LessonPlanOutline,
                            retrieval: RetrievalResult,
                            teach_minutes: dict[int, int]) -> tuple[list[GradeContent], list[str]]:
    """Generate content for every selected grade in parallel. Returns (contents, warnings)."""
    chain = build_content_chain()
    grades = list(req.grades)
    inputs = [_inputs(g, req, outline, retrieval, teach_minutes.get(g, 10)) for g in grades]

    # LCEL .batch runs the per-grade calls concurrently (thread pool).
    results = chain.batch(
        inputs, config={"max_concurrency": config.MAX_CONCURRENCY}, return_exceptions=True
    )

    warnings: list[str] = []
    contents: list[GradeContent] = []
    for grade, inp, res in zip(grades, inputs, results):
        if isinstance(res, Exception) or res is None:
            log.warning("Grade %s content failed (%s); retrying once", grade, res)
            try:
                res = chain.invoke(inp)
            except Exception as exc:  # noqa: BLE001
                res = exc
        if isinstance(res, Exception) or res is None:
            warnings.append(f"Grade {grade}: content generation failed ({str(res)[:120]}).")
            contents.append(_placeholder(grade, req))
            continue
        contents.append(_normalize(res, grade, retrieval, warnings))

    # A total failure is an error, not a warning.
    if all(c.teacher_tip.startswith("Generation failed") for c in contents):
        first = next((r for r in results if isinstance(r, Exception)), None)
        if first is not None:
            raise first
    return contents, warnings
