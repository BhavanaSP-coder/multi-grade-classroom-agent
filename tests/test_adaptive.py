"""AI Adaptive Classroom Assistant: timing, scheduling, validation, learning, approval flow (offline)."""
from __future__ import annotations

import pytest
from werkzeug.datastructures import MultiDict

import config
from adaptive import ai, service, storage
from adaptive.forms import parse_feedback_form, parse_period_list, parse_setup_form
from adaptive.models import (
    FREE_LABEL,
    AdviceItem,
    AttentionWindow,
    BreakConfig,
    Cell,
    FeedbackEntry,
    FixedActivity,
    SubjectConfig,
    TeacherUnavailability,
)
from adaptive.timing import compute_schedule, default_attention_windows
from adaptive.validator import validate_class_timetable


def make_profile(levels=None, **kw):
    p = service.new_profile(kw.pop("grade", 4), "A")
    for k, v in kw.items():
        setattr(p, k, v)
    p.attention = default_attention_windows(p.start_time, p.end_time)
    for w, lvl in zip(p.attention, levels or []):
        w.level = lvl
    storage.save_profile(p)
    return p


MORNING = [5, 5, 4, 3, 2, 2, 2]
AFTERNOON = [2, 2, 2, 3, 4, 5, 5]


def high_share(version):
    highs = [c for c in version.cells if c.load == "high"]
    return sum(1 for c in highs if c.attention >= 3.5) / len(highs)


# ---------------- timing ----------------

def test_schedule_times_and_breaks():
    p = make_profile()
    periods, breaks, errors = compute_schedule(p)
    assert not errors and len(periods) == 7
    assert (periods[0].start, periods[0].end) == (540, 580)
    assert [b.name for b in breaks] == ["Short break", "Lunch"]
    assert periods[2].start == breaks[0].end and periods[4].start == breaks[1].end  # no overlap
    assert periods[4].bucket == "afternoon" and periods[0].bucket == "early_morning"


def test_schedule_errors_when_day_overflows_or_break_misplaced():
    p = make_profile(end_time="12:00")
    assert any("after the closing time" in e for e in compute_schedule(p)[2])
    p = make_profile(breaks=[BreakConfig(name="Late", after_period=7, minutes=10)])
    assert any("Breaks can only follow periods 1-6" in e for e in compute_schedule(p)[2])
    with pytest.raises(ValueError):
        service.generate_proposal(make_profile(end_time="12:00"), use_ai=False)


# ---------------- attention patterns + timings + subject mixes ----------------

@pytest.mark.parametrize("levels", [MORNING, AFTERNOON, [3, 5, 2, 4, 1, 5, 3]])
def test_high_load_follows_attention_pattern(levels):
    p = make_profile(levels)
    v = service.generate_proposal(p, use_ai=False)
    assert v.validation.ok, v.validation.errors

    def avg(load):
        xs = [c.attention for c in v.cells if c.load == load]
        return sum(xs) / len(xs)
    assert avg("high") > avg("medium") > avg("low")
    assert v.validation.metrics["consecutive_high_pairs"] == 0
    if levels == MORNING:  # plenty of attentive, non-adjacent slots
        assert high_share(v) >= 0.8


def test_patterns_differ_between_classes():
    a = service.generate_proposal(make_profile(MORNING), use_ai=False)
    b = service.generate_proposal(make_profile(AFTERNOON), use_ai=False)
    math_a = sum(c.period for c in a.cells if c.subject == "Mathematics")
    math_b = sum(c.period for c in b.cells if c.subject == "Mathematics")
    assert math_a < math_b  # maths earlier for the morning-focused class


@pytest.mark.parametrize("start,end,periods,minutes,days", [
    ("07:30", "13:30", 6, 45, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]),
    ("10:00", "15:00", 5, 40, ["Mon", "Tue", "Wed", "Thu", "Fri"]),
    ("08:00", "16:00", 8, 35, ["Mon", "Wed", "Fri"]),
])
def test_different_school_timings(start, end, periods, minutes, days):
    p = make_profile(start_time=start, end_time=end, num_periods=periods, period_minutes=minutes,
                     working_days=days, breaks=[BreakConfig(name="Lunch", after_period=periods // 2, minutes=30)])
    slots = periods * len(days)
    for s in p.subjects:  # scale demand to fit
        s.periods_per_week = min(s.periods_per_week, max(1, len(days) * s.max_per_day // 2))
    while sum(s.periods_per_week for s in p.subjects) > slots:
        max(p.subjects, key=lambda s: s.periods_per_week).periods_per_week -= 1
    v = service.generate_proposal(p, use_ai=False)
    assert v.validation.ok, v.validation.errors
    assert len(v.cells) == slots


@pytest.mark.parametrize("subjects", [
    [SubjectConfig(name="Art and Craft", load="low", periods_per_week=5),
     SubjectConfig(name="Music", load="low", periods_per_week=5)],
    [SubjectConfig(name="Mathematics", load="high", periods_per_week=10, max_per_day=2),
     SubjectConfig(name="Science", load="high", periods_per_week=10, max_per_day=2),
     SubjectConfig(name="English", load="medium", periods_per_week=10, max_per_day=2),
     SubjectConfig(name="Physical Education", load="low", periods_per_week=5)],
    [SubjectConfig(name="Kannada", load="high", periods_per_week=6, max_per_day=2),  # teacher-reclassified
     SubjectConfig(name="Mathematics", load="medium", periods_per_week=6, max_per_day=2)],
])
def test_subject_combinations(subjects):
    p = make_profile(MORNING, subjects=subjects)
    v = service.generate_proposal(p, use_ai=False)
    assert v.validation.ok, v.validation.errors
    for s in subjects:
        assert sum(1 for c in v.cells if c.subject == s.name) == s.periods_per_week
    assert sum(1 for c in v.cells if c.subject == FREE_LABEL) == 35 - sum(s.periods_per_week for s in subjects)


# ---------------- breaks ----------------

def test_breaks_separate_demanding_periods():
    p = make_profile([4, 4, 4, 4, 4, 4, 4])
    v = service.generate_proposal(p, use_ai=False)
    blocks = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
    for d in p.working_days:
        row = sorted((c for c in v.cells if c.day == d), key=lambda c: c.period)
        for a, b in zip(row, row[1:]):
            if a.load == b.load == "high":
                assert blocks[a.period] != blocks[b.period]
    assert all(1 <= c.period <= 7 for c in v.cells)


# ---------------- teacher availability ----------------

def test_teacher_unavailability_respected():
    p = make_profile(MORNING)
    p.subjects[0].teacher = "Mrs Rao"  # Mathematics
    p.unavailability = [TeacherUnavailability(teacher="Mrs Rao", day=d, periods=[1, 2]) for d in p.working_days]
    v = service.generate_proposal(p, use_ai=False)
    assert v.validation.ok
    assert not [c for c in v.cells if c.subject == "Mathematics" and c.period in (1, 2)]


def test_impossible_availability_is_reported_not_hidden():
    p = make_profile(MORNING)
    p.subjects[0].teacher = "Mrs Rao"
    p.unavailability = [TeacherUnavailability(teacher="Mrs Rao", day=d, periods=list(range(1, 8)))
                        for d in p.working_days[:4]]
    v = service.generate_proposal(p, use_ai=False)
    assert not v.validation.ok
    assert any("Mathematics" in n and "Options" in n for n in v.notes)
    with pytest.raises(ValueError):
        service.approve(storage.load_profile(p.id))


def test_teacher_busy_in_another_class():
    a = make_profile(MORNING)
    a.subjects[0].teacher = "Mr Khan"
    service.generate_proposal(a, use_ai=False)
    service.approve(a)
    b = make_profile(MORNING, grade=5)
    b.subjects[0].teacher = "Mr Khan"
    v = service.generate_proposal(b, use_ai=False)
    busy = {(c.day, c.period) for c in a.current.cells if c.subject == "Mathematics"}
    assert v.validation.ok
    assert not [c for c in v.cells if c.subject == "Mathematics" and (c.day, c.period) in busy]


# ---------------- locks, fixed activities, regeneration, approval ----------------

def test_locked_and_fixed_preserved_on_regenerate():
    p = make_profile(MORNING)
    p.fixed = [FixedActivity(day="Mon", period=1, label="Assembly")]
    service.generate_proposal(p, use_ai=False)
    service.approve(p)
    service.apply_edit(p, "current", "set", ("Tue", 6), subject="Music")
    service.apply_edit(p, "current", "lock", ("Tue", 6))
    before = p.current.model_copy(deep=True)
    for _ in range(3):
        v = service.generate_proposal(p, use_ai=False, regenerate=True)
        assert v.cell("Tue", 6).subject == "Music" and v.cell("Tue", 6).locked
        assert v.cell("Mon", 1).subject == "Assembly"
    assert p.current.cells == before.cells  # current untouched until approval
    if v.validation.ok:
        service.approve(p)
        assert p.current.id == v.id and p.history[0].id == before.id


def test_regeneration_and_reject_never_change_current():
    p = make_profile(MORNING)
    service.generate_proposal(p, use_ai=False)
    service.approve(p)
    cur = p.current.model_copy(deep=True)
    v = service.generate_proposal(p, use_ai=False, regenerate=True)
    assert v.validation.ok and v.status == "proposed"
    service.reject(p)
    loaded = storage.load_profile(p.id)
    assert loaded.proposal is None and loaded.current.cells == cur.cells


def test_manual_edits_are_checked():
    p = make_profile(MORNING)
    service.generate_proposal(p, use_ai=False)
    service.approve(p)
    maths = next(c for c in p.current.cells if c.subject == "Mathematics")
    other = next(c for c in p.current.cells if c.subject == "Music")
    v = service.apply_edit(p, "current", "set", (maths.day, maths.period), subject="Music")
    assert not v.validation.ok
    assert any("Mathematics has" in e for e in v.validation.errors)
    assert any("Music appears" in e for e in v.validation.errors)
    service.apply_edit(p, "current", "lock", (other.day, other.period))
    with pytest.raises(ValueError):
        service.apply_edit(p, "current", "swap", (other.day, other.period), (maths.day, maths.period))


# ---------------- validator ----------------

def test_validator_catches_violations():
    p = make_profile(MORNING)
    v = service.generate_proposal(p, use_ai=False)
    cells = [c.model_copy() for c in v.cells]
    assert validate_class_timetable(p, cells).ok

    dup = cells + [cells[0].model_copy()]
    assert any("Overlap" in e for e in validate_class_timetable(p, dup).errors)
    in_break = cells[:-1] + [Cell(day="Mon", period=9, subject="Music")]
    errs = validate_class_timetable(p, in_break).errors
    assert any("not a teaching period" in e for e in errs) and any("empty" in e for e in errs)
    locked = {(cells[0].day, cells[0].period): "Science" if cells[0].subject != "Science" else "Music"}
    assert any("Locked period" in e for e in validate_class_timetable(p, cells, locked_ref=locked).errors)
    p.subjects[0].teacher = "T1"
    p.unavailability = [TeacherUnavailability(teacher="T1", day=c.day, periods=[c.period])
                        for c in cells if c.subject == "Mathematics"][:1]
    assert any("not available" in e for e in validate_class_timetable(p, cells).errors)
    p.subjects[0].max_per_day = 1
    if any(n > 1 for n in [sum(1 for c in cells if c.subject == "Mathematics" and c.day == d) for d in p.working_days]):
        assert any("limit is 1" in e for e in validate_class_timetable(p, cells).errors)


def test_overbooked_week_explains_conflict():
    p = make_profile(MORNING)
    p.subjects.append(SubjectConfig(name="Computer Science", load="medium", periods_per_week=10, max_per_day=2))
    v = service.generate_proposal(p, use_ai=False)
    assert not v.validation.ok
    assert any("periods a week but only 35" in n for n in v.notes)


# ---------------- learning from feedback ----------------

def test_feedback_moves_struggling_subject():
    levels = [4, 4, 4, 4, 5, 5, 5]  # afternoon looks attentive on paper
    p = make_profile(levels)
    before = service.generate_proposal(p, use_ai=False)
    afternoon_before = sum(1 for c in before.cells if c.subject == "Mathematics" and c.period >= 5)
    service.approve(p)
    for d in p.working_days:
        for per in (5, 6):
            service.add_feedback(p, FeedbackEntry(day=d, period=per, subject="Mathematics",
                                                  understanding=1, engagement=1, concentration=1))
    after = service.generate_proposal(p, use_ai=False, regenerate=True)
    afternoon_after = sum(1 for c in after.cells if c.subject == "Mathematics" and c.period >= 5)
    assert afternoon_before > 0 and afternoon_after < afternoon_before
    assert any(e.kind == "feedback" and "Mathematics" in e.text for e in after.explanations)
    assert after.changes  # explained diff against the approved timetable
    assert storage.load_profile(p.id).current.id == before.id  # not applied until approved


# ---------------- explanations + AI guard rails ----------------

def test_explanations_reference_real_cells():
    p = make_profile(MORNING)
    v = service.generate_proposal(p, use_ai=False)
    cells = {f"{c.day} P{c.period}": c for c in v.cells}
    for e in v.explanations:
        for ref in e.cells:
            assert ref in cells
        if e.kind == "placement" and e.cells and "load" in e.text:
            subject = e.text.split(" (")[0]
            assert all(cells[r].subject == subject for r in e.cells)


def test_ai_summary_contradictions_are_dropped():
    cells = [Cell(day="Mon", period=1, subject="Mathematics", load="high"),
             Cell(day="Mon", period=2, subject="Music", load="low")]
    kept, dropped = ai.verify_bullets(
        ["Mathematics is in Mon P1 because attention is highest.",
         "Music is in Mon P1 to wake students up.",
         "Science is in Fri P9.",
         "Low-load work is kept for later in the day."], cells, ["Mathematics", "Music", "Science"])
    assert kept == ["Mathematics is in Mon P1 because attention is highest.",
                    "Low-load work is kept for later in the day."]
    assert dropped == 2


def test_ai_advice_used_and_failure_is_graceful(monkeypatch):
    p = make_profile(MORNING)
    monkeypatch.setattr(service, "get_advice", lambda *a: (
        [AdviceItem(subject="Music", preferred_time="afternoon", reason="Low focus after lunch")], ["Use songs"]))
    monkeypatch.setattr(service, "summarize", lambda *a: (["Mathematics is in Mon P1."], 0))
    v = service.generate_proposal(p, use_ai=True)
    assert v.validation.ok and v.advice[0].honoured and any("Use songs" in n for n in v.notes)
    assert any(e.kind == "ai_advice" for e in v.explanations)

    def boom(*a):
        raise RuntimeError("429 rate limit")
    monkeypatch.setattr(service, "get_advice", boom)
    v = service.generate_proposal(p, use_ai=True)
    assert v.validation.ok and any("AI advice unavailable" in n for n in v.notes)


# ---------------- forms, storage, PDF ----------------

def _setup_form(p, **over):
    data = MultiDict([("grade", str(p.grade)), ("section", "B"), ("num_students", "32"),
                      ("start_time", p.start_time), ("end_time", p.end_time), ("num_periods", "7"),
                      ("period_minutes", "40"), ("att_count", "0")])
    for d in p.working_days:
        data.add("working_days", d)
    for b in p.breaks:
        data.add("br_name", b.name); data.add("br_after", str(b.after_period)); data.add("br_minutes", str(b.minutes))
    for s in p.subjects:
        data.add("subj_name", s.name); data.add("subj_load", s.load); data.add("subj_ppw", str(s.periods_per_week))
        data.add("subj_max", str(s.max_per_day)); data.add("subj_teacher", s.teacher)
    for k, v in over.items():
        data.setlist(k, v if isinstance(v, list) else [v])
    return data


def test_setup_form_validation():
    p = make_profile(MORNING)
    updated, errors = parse_setup_form(_setup_form(p), p)
    assert not errors and updated.section == "B" and [w.level for w in updated.attention] == MORNING
    _, errors = parse_setup_form(_setup_form(p, start_time="9am"), p)
    assert any("HH:MM" in e for e in errors)
    _, errors = parse_setup_form(_setup_form(p, un_teacher="Nobody", un_day="Mon", un_periods="1"), p)
    assert any("no subject is taught by" in e for e in errors)
    _, errors = parse_setup_form(_setup_form(p, fx_day="Sat", fx_period="1", fx_label="Assembly"), p)
    assert any("not a working day" in e for e in errors)
    names = [s.name for s in p.subjects]
    _, errors = parse_setup_form(_setup_form(p, subj_name=[names[0]] + [names[0]] + names[2:]), p)
    assert any("listed twice" in e for e in errors)
    updated, _ = parse_setup_form(_setup_form(p, start_time="08:00", end_time="14:30"), p)
    assert updated.attention[0].start == "08:00"
    assert parse_period_list("1-3, 6", 7) == [1, 2, 3, 6]


def test_feedback_form_and_storage_roundtrip():
    p = make_profile()
    entry, errors = parse_feedback_form(MultiDict({"day": "Mon", "period": "2", "subject": "Science",
                                                   "understanding": "4", "engagement": "5", "concentration": "3",
                                                   "took_longer": "1"}), p)
    assert not errors and entry.took_longer
    assert parse_feedback_form(MultiDict({"day": "Sun", "period": "9", "subject": "X"}), p)[1]
    service.add_feedback(p, entry)
    assert storage.load_profile(p.id).feedback[0].understanding == 4
    assert storage.load_profile("../etc") is None
    assert service.delete_feedback(p, entry.id) and not storage.load_profile(p.id).feedback


def test_timetable_pdf_export():
    from pypdf import PdfReader

    from adaptive.pdf import build_timetable_pdf

    p = make_profile(MORNING)
    v = service.generate_proposal(p, use_ai=False)
    path = build_timetable_pdf(p, v)
    text = "".join(page.extract_text() for page in PdfReader(str(path)).pages)
    assert "Weekly timetable" in text and "Mathematics" in text and "Why this timetable" in text


def test_ai_advice_cannot_override_teacher_attention(monkeypatch):
    """Even if the AI suggests a poor time, the teacher's attention data wins."""
    p = make_profile(MORNING)
    monkeypatch.setattr(service, "get_advice", lambda *a: (
        [AdviceItem(subject=s, preferred_time="afternoon", reason="bad advice") for s in ("Mathematics", "Science")]
        + [AdviceItem(subject=s, preferred_time="early_morning", reason="bad advice") for s in ("Music", "Art and Craft")],
        []))
    monkeypatch.setattr(service, "summarize", lambda *a: ([], 0))
    v = service.generate_proposal(p, use_ai=True)
    assert v.validation.ok and high_share(v) >= 0.8
