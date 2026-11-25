"""Dyslexia-friendly worksheet generation (LCEL + structured output, parallel per grade).

The accessible sheet is generated FROM the grade's existing objectives, worksheet tasks and
retrieved curriculum, so the learning outcomes stay the same. Only presentation and
reading load change. Layout settings (font size, spacing, answer space, clutter) are applied
by the HTML preview and the PDF generator, not by the model.
"""
from __future__ import annotations

import logging
import re

from langchain_core.prompts import ChatPromptTemplate

import config
from chains.llm import GRADE_LEVEL_GUIDE, get_llm, language_instruction, structured
from chains.schemas import (
    AccessibilitySettings,
    AccessibleWorksheet,
    GradeContent,
    LessonPlanOutline,
    LessonRequest,
    RetrievalResult,
)

log = logging.getLogger(__name__)

SYSTEM = """You are a special-education specialist who adapts primary worksheets for children with dyslexia and reading difficulties in rural Indian schools.
Rewrite the GRADE {grade} worksheet below as an accessible worksheet. Level: {grade_level}

NON-NEGOTIABLE:
- Keep the SAME learning objectives and the SAME level of mathematical or subject challenge. Accessibility changes how the work is presented, NOT what the child must learn. Do not swap hard questions for easier ones, and do not make numbers smaller.
- Show a worked example BEFORE the questions, similar to the hardest question.
- Use clear, consistent formatting: every question has a heading, short steps, the question, and keywords.
- Offer another way to show understanding for every question (draw it, use objects, tell the teacher).
- Stay grounded in the CURRICULUM CONTEXT. Never invent textbook names, chapters or official codes.

TEACHER-SELECTED ADAPTATIONS:
{rules}

QUESTION COUNT: {count_rule}
{language_instruction}"""

HUMAN = """Topic: {topic}
Subject: {subject}

Learning objectives for Grade {grade}:
{objectives}

ORIGINAL WORKSHEET "{worksheet_title}" (instruction: {worksheet_instructions}):
{tasks}

CURRICULUM CONTEXT FOR GRADE {grade}:
{context}"""

accessible_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])

RULE_TEXT = {
    "simple_instructions": "- Break every instruction into 1-3 small numbered steps, one action per step.",
    "short_sentences": "- Use short sentences (about 8-10 words) and everyday words. Explain any subject word you must keep in brackets.",
    "one_question_per_section": "- Put exactly ONE question in each section. Never combine two questions.",
    "visual_hints": "- Give each question a simple visual hint the child can draw or use (circles, sticks, stones, a number line).",
    "highlight_keywords": "- For each question list 1-3 important keywords copied EXACTLY from its steps or question.",
    "reduced_clutter": "- Keep text minimal: no extra stories, no decorative wording.",
    "oral_alternatives": "- In other_ways, always include an ORAL option (say the answer to the teacher or a partner) so a child can answer without writing.",
}


def build_rules(settings: AccessibilitySettings) -> str:
    rules = [RULE_TEXT[k] for k in settings.enabled if k in RULE_TEXT]
    rules.append("- Use simple, age-appropriate language and avoid unnecessarily complicated vocabulary.")
    return "\n".join(rules)


def count_rule(settings: AccessibilitySettings, n_tasks: int) -> str:
    n_tasks = max(1, n_tasks)
    if settings.reduce_questions and n_tasks > 3:
        return (f"The original has {n_tasks} tasks. Use 3 questions, keeping the ones that together "
                "still cover EVERY learning objective, including the hardest one.")
    return f"Keep exactly {n_tasks} questions, one for each original task, in the same order."


def build_accessible_chain():
    return accessible_prompt | structured(AccessibleWorksheet)


def _inputs(grade: int, req: LessonRequest, settings: AccessibilitySettings,
            outline: LessonPlanOutline, content: GradeContent, retrieval: RetrievalResult) -> dict:
    obj = outline.objectives_for(grade)
    tasks = "\n".join(f"{i}. {t.task}  (answer: {t.answer})"
                      for i, t in enumerate(content.worksheet_tasks, start=1))
    return {
        "grade": grade,
        "grade_level": GRADE_LEVEL_GUIDE.get(grade, f"Grade {grade}"),
        "rules": build_rules(settings),
        "count_rule": count_rule(settings, len(content.worksheet_tasks)),
        "language_instruction": language_instruction(req.language),
        "topic": req.topic,
        "subject": req.subject,
        "objectives": "\n".join(f"- {o}" for o in (obj.objectives if obj else [])) or "- (not available)",
        "worksheet_title": content.worksheet_title,
        "worksheet_instructions": content.worksheet_instructions,
        "tasks": tasks or "(no tasks)",
        "context": retrieval.context_for_grade(grade),
    }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def normalize_accessible(ws: AccessibleWorksheet, grade: int, settings: AccessibilitySettings,
                         n_original: int, warnings: list[str]) -> AccessibleWorksheet:
    """Enforce counts and make keywords real substrings so highlighting is truthful."""
    ws.grade = grade
    ws.questions = [q for q in ws.questions if q.question.strip()]
    expected_max = 3 if (settings.reduce_questions and n_original > 3) else max(n_original, 1)
    if len(ws.questions) > expected_max:
        ws.questions = ws.questions[:expected_max]
    if len(ws.questions) < min(expected_max, 3):
        warnings.append(f"Grade {grade}: accessible worksheet has only {len(ws.questions)} question(s).")
    for q in ws.questions:
        q.steps = [s.strip() for s in q.steps if s and s.strip()][:4]
        text = _norm(" ".join([q.question, *q.steps]))
        q.keywords = [k.strip() for k in dict.fromkeys(q.keywords)
                      if k and k.strip() and _norm(k) in text][:3]
        if not settings.visual_hints:
            q.visual_hint = ""
    ws.how_to_use = [t.strip() for t in ws.how_to_use if t and t.strip()][:3]
    ws.example.steps = [s.strip() for s in ws.example.steps if s and s.strip()][:4]
    return ws


def generate_accessible_worksheets(req: LessonRequest, settings: AccessibilitySettings,
                                   outline: LessonPlanOutline, contents: list[GradeContent],
                                   retrieval: RetrievalResult
                                   ) -> tuple[list[AccessibleWorksheet], list[str]]:
    """Generate one accessible worksheet per grade in parallel. Returns (sheets, warnings)."""
    if not settings.any_enabled:
        return [], []
    usable = [c for c in contents if c.worksheet_tasks and c.worksheet_tasks[0].task != "(not generated)"]
    if not usable:
        return [], ["Accessible worksheets skipped: no standard worksheet content to adapt."]

    chain = build_accessible_chain()
    inputs = [_inputs(c.grade, req, settings, outline, c, retrieval) for c in usable]
    results = chain.batch(inputs, config={"max_concurrency": config.MAX_CONCURRENCY},
                          return_exceptions=True)

    sheets: list[AccessibleWorksheet] = []
    warnings: list[str] = []
    for c, inp, res in zip(usable, inputs, results):
        if isinstance(res, Exception) or res is None:
            log.warning("Accessible worksheet for grade %s failed (%s); retrying", c.grade, res)
            try:
                res = chain.invoke(inp)
            except Exception as exc:  # noqa: BLE001
                res = exc
        if isinstance(res, Exception) or res is None:
            warnings.append(f"Grade {c.grade}: accessible worksheet could not be generated "
                            f"({str(res)[:120]}).")
            continue
        sheets.append(normalize_accessible(res, c.grade, settings, len(c.worksheet_tasks), warnings))
    return sorted(sheets, key=lambda s: s.grade), warnings
