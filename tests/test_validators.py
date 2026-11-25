"""Tests for the rotation scheduler, timetable validator and input validators (no API key needed).

Run:  python -m pytest -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chains.schemas import GradeAssignment, TimeBlock, Timetable  # noqa: E402
from chains.timetable_chain import (  # noqa: E402
    apply_default_texts,
    build_rotation_skeleton,
    split_evenly,
)
from validators import validate_lesson_form, validate_timetable, validate_upload  # noqa: E402

ALL_GRADE_SETS = [list(c) for r in range(1, 6) for c in itertools.combinations([1, 2, 3, 4, 5], r)]


def _filled(duration, grades):
    return apply_default_texts(build_rotation_skeleton(duration, grades), None, [])


def test_split_evenly():
    assert split_evenly(45, 5) == [9, 9, 9, 9, 9]
    assert split_evenly(37, 5) == [8, 8, 7, 7, 7]
    assert sum(split_evenly(101, 7)) == 101


@pytest.mark.parametrize("duration", [15, 16, 23, 25, 29, 30, 37, 40, 45, 53, 60, 61, 90, 119, 180])
@pytest.mark.parametrize("grades", ALL_GRADE_SETS)
def test_skeleton_always_valid(duration, grades):
    tt = _filled(duration, grades)
    report = validate_timetable(tt, duration, grades)
    assert report.ok, report.errors
    # every grade gets teacher time when more than one block exists
    assert not [w for w in report.warnings if "never gets" in w]


def test_short_class_pairs_grades():
    tt = build_rotation_skeleton(15, [1, 2, 3, 4, 5])
    focuses = [b.teacher_focus for b in tt.blocks]
    assert all(b.minutes >= 5 for b in tt.blocks)
    assert sorted(g for f in focuses for g in f) == [1, 2, 3, 4, 5]


def test_teacher_starts_with_youngest():
    tt = build_rotation_skeleton(45, [2, 4, 5])
    rotation = [b for b in tt.blocks if b.teacher_focus]
    assert rotation[0].teacher_focus == [2]


def _block(s, e, focus, grades, task="x"):
    return TimeBlock(start_min=s, end_min=e, teacher_focus=focus, assignments=[
        GradeAssignment(grade=g, task=task,
                        mode="whole_class" if not focus else "with_teacher" if g in focus else "independent")
        for g in grades])


def test_detects_gap_overlap_and_coverage():
    g = [1, 2]
    gap = Timetable(blocks=[_block(0, 10, [1], g), _block(12, 30, [2], g)])
    assert any("Gap" in e for e in validate_timetable(gap, 30, g).errors)

    overlap = Timetable(blocks=[_block(0, 15, [1], g), _block(10, 30, [2], g)])
    assert any("overlaps" in e for e in validate_timetable(overlap, 30, g).errors)

    short = Timetable(blocks=[_block(0, 10, [1], g), _block(10, 25, [2], g)])
    assert any("expected 30" in e for e in validate_timetable(short, 30, g).errors)


def test_detects_missing_grade_and_empty_task():
    tt = Timetable(blocks=[_block(0, 20, [1], [1]), _block(20, 30, [2], [1, 2], task=" ")])
    errors = validate_timetable(tt, 30, [1, 2]).errors
    assert any("Grade 2 has no work" in e for e in errors)
    assert any("empty task" in e for e in errors)


def test_lesson_form_validation():
    class Form(dict):
        def getlist(self, k):
            return self.get(k, [])

    ok = Form(topic="Fractions", subject="Maths", grades=["3", "1"], duration="45", language="Hindi")
    req, errors, _ = validate_lesson_form(ok)
    assert not errors and req.grades == [1, 3] and req.language == "Hindi"

    bad = Form(topic="", subject="Cooking", grades=["7"], duration="5", language="French")
    req, errors, _ = validate_lesson_form(bad)
    assert req is None and len(errors) >= 5


def test_upload_validation():
    assert validate_upload("notes.docx", b"hello")
    assert validate_upload("fake.pdf", b"not a pdf at all")
    assert validate_upload("short.txt", b"hi")
    assert not validate_upload("grade3.txt", b"Grade 3 fractions: halves, thirds and quarters of shapes.")
    assert not validate_upload("doc.pdf", b"%PDF-1.7\n...")
