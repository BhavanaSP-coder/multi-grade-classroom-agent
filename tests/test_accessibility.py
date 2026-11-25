"""Dyslexia-friendly worksheet feature: settings, generation logic, preview, PDF, edits (offline)."""
from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda
from werkzeug.datastructures import MultiDict

import config
from chains import accessible_chain, pipeline
from chains.schemas import ACCESSIBILITY_OPTIONS, DYSLEXIA_PRESET, AccessibilitySettings, LessonRequest
from tests.conftest import make_accessible, make_content
from validators import parse_accessibility, validate_lesson_form

ALL = {k for k, _, _ in ACCESSIBILITY_OPTIONS}


# ---------------- settings ----------------

def test_settings_default_off_and_old_lessons_load():
    req = LessonRequest.model_validate({"topic": "x", "subject": "Maths", "grades": [1], "duration": 30})
    assert not req.accessibility.any_enabled and req.attention_hint == ""


def test_dyslexia_preset_keeps_question_count():
    s = AccessibilitySettings.dyslexia_preset()
    assert set(s.enabled) == DYSLEXIA_PRESET == ALL - {"reduce_questions"}
    assert not s.reduce_questions


@pytest.mark.parametrize("form,expected", [
    ({}, set()),
    ({"acc_preset": "off", "acc_large_font": "1"}, set()),
    ({"acc_preset": "dyslexia"}, DYSLEXIA_PRESET),
    ({"acc_preset": "dyslexia", "acc_reduce_questions": "1"}, ALL),
    ({"acc_preset": "custom", "acc_large_font": "1", "acc_highlight_keywords": "on"},
     {"large_font", "highlight_keywords"}),
    ({"acc_preset": "custom"}, set()),
])
def test_parse_accessibility(form, expected):
    assert set(parse_accessibility(MultiDict(form)).enabled) == expected


def test_lesson_form_carries_accessibility_and_attention():
    form = MultiDict([("topic", "Fractions"), ("subject", "Maths"), ("grades", "4"), ("duration", "40"),
                      ("language", "English"), ("acc_preset", "dyslexia"), ("attention_hint", "Low")])
    req, errors, _ = validate_lesson_form(form)
    assert not errors and req.accessibility.preset == "dyslexia" and req.attention_hint == "Low"
    form["attention_hint"] = "Sleepy"
    assert validate_lesson_form(form)[1]


# ---------------- prompt rules + normalisation ----------------

def test_rules_follow_settings_and_never_lower_expectations():
    s = AccessibilitySettings(preset="custom", simple_instructions=True, visual_hints=True)
    rules = accessible_chain.build_rules(s)
    assert "numbered steps" in rules and "visual hint" in rules and "keywords" not in rules
    assert "SAME level" in accessible_chain.SYSTEM and "worked example" in accessible_chain.SYSTEM
    assert "Keep exactly 4" in accessible_chain.count_rule(s, 4)
    assert "Use 3 questions" in accessible_chain.count_rule(AccessibilitySettings(reduce_questions=True), 4)
    assert "Keep exactly 3" in accessible_chain.count_rule(AccessibilitySettings(reduce_questions=True), 3)


def test_normalize_filters_fake_keywords_and_caps_questions():
    ws = make_accessible(2, n=6)
    ws.questions[0].keywords = ["half", "elephant", "half"]
    warnings: list[str] = []
    out = accessible_chain.normalize_accessible(ws, 2, AccessibilitySettings(reduce_questions=True), 4, warnings)
    assert len(out.questions) == 3
    assert out.questions[0].keywords == ["half"]
    assert all(q.visual_hint == "" for q in out.questions)  # visual hints not enabled
    out2 = accessible_chain.normalize_accessible(make_accessible(2, n=6), 2,
                                                 AccessibilitySettings(visual_hints=True), 4, [])
    assert len(out2.questions) == 4 and out2.questions[0].visual_hint


def test_generation_runs_per_grade_and_reports_failures(monkeypatch, sample_lesson):
    calls = []

    def fake(inputs):
        calls.append(inputs["grade"])
        if inputs["grade"] == 3:
            raise RuntimeError("model down")
        assert "Learning objectives" in accessible_chain.HUMAN and inputs["objectives"]
        return make_accessible(inputs["grade"])

    monkeypatch.setattr(accessible_chain, "build_accessible_chain", lambda: RunnableLambda(fake))
    r = sample_lesson
    sheets, warnings = accessible_chain.generate_accessible_worksheets(
        r.request, AccessibilitySettings.dyslexia_preset(), r.outline, r.contents, r.retrieval)
    assert [s.grade for s in sheets] == [1]
    assert any("Grade 3" in w for w in warnings)
    assert calls.count(3) == 2  # retried once
    assert accessible_chain.generate_accessible_worksheets(
        r.request, AccessibilitySettings(), r.outline, r.contents, r.retrieval) == ([], [])


def test_regenerate_accessible_saves_and_turns_off(monkeypatch, sample_lesson):
    monkeypatch.setattr(accessible_chain, "build_accessible_chain",
                        lambda: RunnableLambda(lambda i: make_accessible(i["grade"], 2)))
    pipeline.save_lesson(sample_lesson)
    s = AccessibilitySettings(preset="custom", large_font=True)
    out = pipeline.regenerate_accessible(sample_lesson, s)
    loaded = pipeline.load_lesson(sample_lesson.lesson_id)
    assert loaded.request.accessibility.large_font and len(loaded.accessible) == 2
    off = pipeline.regenerate_accessible(out, AccessibilitySettings())
    assert off.accessible == [] and not pipeline.load_lesson(out.lesson_id).accessible


# ---------------- preview + PDF ----------------

def test_highlight_filter_escapes_and_marks():
    import app as appmod

    html = str(appmod.highlight_filter("Colour <b>half</b> of the Circle", ["half", "circle"], True))
    assert "&lt;b&gt;" in html and '<mark class="kw">half</mark>' in html and '<mark class="kw">Circle</mark>' in html
    assert "<mark" not in str(appmod.highlight_filter("half", ["half"], False))


def test_pdf_highlight_markup_is_safe():
    from pdf_generator import highlight_markup

    out = highlight_markup("A & B half", ["half"], True)
    assert "&amp;" in out and "<b>half</b>" in out


def _pages(path):
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def test_pdf_packet_includes_accessible_pages(sample_lesson):
    from pdf_generator import build_lesson_pdf

    with_acc = build_lesson_pdf(sample_lesson, config.PDF_DIR / "with.pdf")
    plain = sample_lesson.model_copy(deep=True)
    plain.accessible = []
    without = build_lesson_pdf(plain, config.PDF_DIR / "without.pdf")
    assert _pages(with_acc) > _pages(without)
    only = build_lesson_pdf(sample_lesson, config.PDF_DIR / "only.pdf", only_accessible=True)
    assert _pages(only) >= 2
    with pytest.raises(ValueError):
        build_lesson_pdf(plain, config.PDF_DIR / "none.pdf", only_accessible=True)


@pytest.mark.parametrize("language", ["Kannada", "Hindi"])
def test_accessible_pdf_in_indic_language(sample_lesson, language):
    from pdf_generator import build_lesson_pdf

    r = sample_lesson.model_copy(deep=True)
    r.request.language = language
    r.accessible[0].questions[0].question = "ವೃತ್ತದ ಅರ್ಧ ಭಾಗಕ್ಕೆ ಬಣ್ಣ ಹಚ್ಚಿ" if language == "Kannada" else "वृत्त का आधा भाग रंगें"
    assert _pages(build_lesson_pdf(r, config.PDF_DIR / f"{language}.pdf", only_accessible=True)) >= 2


# ---------------- teacher edits ----------------

def test_apply_edits_whitelist(sample_lesson):
    pipeline.save_lesson(sample_lesson)
    updated, errors = pipeline.apply_edits(sample_lesson, {
        "contents.0.activity_title": "New title",
        "accessible.1.questions.0.question": "Shade half.",
        "timetable.blocks.1.assignments.0.task": "Read quietly",
        "contents.0.grade": "9", "request.topic": "x", "contents.9.activity_title": "x",
        "outline.lesson_title": "", "contents.0.materials": "x"})
    assert updated.contents[0].activity_title == "New title"
    assert updated.accessible[1].questions[0].question == "Shade half."
    assert updated.timetable.blocks[1].assignments[0].task == "Read quietly"
    assert updated.validation.ok and updated.edited_at
    assert len(errors) == 5
    assert pipeline.load_lesson(sample_lesson.lesson_id).contents[0].activity_title == "New title"


def test_routes_preview_edit_and_regenerate(client, sample_lesson, monkeypatch):
    pipeline.save_lesson(sample_lesson)
    lid = sample_lesson.lesson_id
    page = client.get(f"/lesson/{lid}")
    assert page.status_code == 200 and b"teacher preview" in page.data and b'mark class="kw"' in page.data
    assert client.get(f"/lesson/{lid}/pdf?only=accessible").mimetype == "application/pdf"
    r = client.post(f"/lesson/{lid}/edit", json={"edits": {"outline.big_idea": "Fair shares"}})
    assert r.json["ok"] and pipeline.load_lesson(lid).outline.big_idea == "Fair shares"
    assert client.post(f"/lesson/{lid}/edit", json={}).status_code == 400

    monkeypatch.setattr(accessible_chain, "build_accessible_chain",
                        lambda: RunnableLambda(lambda i: make_accessible(i["grade"], 3)))
    r = client.post(f"/lesson/{lid}/accessible", data={"acc_preset": "custom", "acc_large_font": "1"})
    assert r.status_code == 302
    assert pipeline.load_lesson(lid).request.accessibility.enabled == ["large_font"]


def test_full_pipeline_offline_includes_accessible(monkeypatch, sample_lesson):
    """generate_lesson end to end with the LLM steps replaced by fakes (structure check)."""
    import rag
    from chains import pipeline as pl

    r = sample_lesson
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: r.retrieval)
    monkeypatch.setattr(pl, "generate_outline", lambda req, ret: (r.outline, []))
    monkeypatch.setattr(pl, "generate_grade_contents", lambda req, o, ret, tm: ([make_content(g) for g in req.grades], []))
    monkeypatch.setattr(config, "TIMETABLE_LLM", False)
    monkeypatch.setattr(accessible_chain, "build_accessible_chain",
                        lambda: RunnableLambda(lambda i: make_accessible(i["grade"])))
    out = pl.generate_lesson(r.request)
    assert out.validation.ok and [a.grade for a in out.accessible] == [1, 3]
    assert pl.load_lesson(out.lesson_id) is not None
