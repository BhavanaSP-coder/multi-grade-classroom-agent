"""Shared fixtures: isolated data folders and a realistic sample lesson (no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from chains.schemas import (  # noqa: E402
    AccessibilitySettings,
    AccessibleQuestion,
    AccessibleWorksheet,
    GradeContent,
    GradeObjectives,
    LessonPlanOutline,
    LessonRequest,
    LessonResult,
    QuizQuestion,
    RetrievalResult,
    SourceSnippet,
    ValidationReport,
    WorkedExample,
    WorksheetTask,
)


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Every test writes lessons, PDFs and class profiles to a temp folder."""
    for name in ("LESSON_DIR", "PDF_DIR", "ADAPTIVE_DIR"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(config, name, d)
    monkeypatch.setattr(config, "ADAPTIVE_USE_AI", False)
    monkeypatch.setattr(config, "ADAPTIVE_RESTARTS", 4)
    yield tmp_path


def make_content(grade: int) -> GradeContent:
    return GradeContent(
        grade=grade, activity_title=f"Stone sharing G{grade}", activity_type="hands-on",
        materials=["stones", "chalk"], activity_steps=["Give stones", "Share equally", "Talk"],
        worksheet_title=f"Halves G{grade}", worksheet_instructions="Work alone.",
        worksheet_tasks=[WorksheetTask(task=f"Task {i} for G{grade}: shade one half", answer=f"A{i}")
                         for i in range(1, 5)],
        quiz=[QuizQuestion(question=f"Q{i}?", options=[], answer=f"QA{i}") for i in range(1, 4)],
        teacher_tip="Check equal parts.", source_labels=["S1"])


def make_accessible(grade: int, n: int = 4) -> AccessibleWorksheet:
    return AccessibleWorksheet(
        grade=grade, title="Halves", goal="I can find one half.", how_to_use=["One step at a time."],
        example=WorkedExample(problem="Share 6 stones by 2.", steps=["Count 6.", "Share."], answer="3"),
        questions=[AccessibleQuestion(heading=f"Question {i}", steps=["Look at the circle.", "Draw a line."],
                                      question="Colour one half of the circle.", keywords=["half", "circle"],
                                      visual_hint="Draw a circle", other_ways="Fold paper.", answer="Half coloured")
                   for i in range(1, n + 1)],
        teacher_note="Same objective.")


@pytest.fixture
def sample_lesson():
    from chains.timetable_chain import apply_default_texts, build_rotation_skeleton
    from validators import validate_timetable

    grades = [1, 3]
    req = LessonRequest(topic="Fractions", subject="Maths", grades=grades, duration=45,
                        accessibility=AccessibilitySettings.dyslexia_preset())
    outline = LessonPlanOutline(
        lesson_title="Sharing fairly", big_idea="Equal parts", whole_class_opening="Song",
        whole_class_closing="Share", grades=[GradeObjectives(grade=g, objectives=["Identify halves"],
                                                              success_criteria="Shows half", key_vocabulary=["half"],
                                                              curriculum_alignment="[S1]") for g in grades])
    contents = [make_content(g) for g in grades]
    tt = apply_default_texts(build_rotation_skeleton(45, grades), outline, contents)
    retrieval = RetrievalResult(query="Maths: Fractions", by_grade={1: ["S1"], 3: ["S1"]}, sources=[
        SourceSnippet(label="S1", chunk_id="x", source="demo.txt", grade=0, score=0.6, text="Halves", is_demo=True)])
    return LessonResult(lesson_id="a" * 32, request=req, retrieval=retrieval, outline=outline,
                        contents=contents, timetable=tt,
                        validation=validate_timetable(tt, 45, grades),
                        accessible=[make_accessible(g) for g in grades])


@pytest.fixture
def client():
    import app as appmod

    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()
