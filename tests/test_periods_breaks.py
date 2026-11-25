"""Period-count limit, break positions, end-time calculation and setup form round trips."""
from __future__ import annotations

import pytest
from werkzeug.datastructures import MultiDict

import config
from adaptive import service, storage
from adaptive.forms import parse_setup_form
from adaptive.models import BreakConfig
from adaptive.timing import (
    break_errors,
    compute_schedule,
    fmt12,
    period_count_errors,
    school_day_end,
    valid_break_positions,
)
from adaptive.validator import validate_class_timetable


def setup_form(p, periods, breaks, **over):
    """Build a setup form like the browser sends it. breaks = [(name, after, minutes), ...]."""
    data = MultiDict([("grade", str(p.grade)), ("section", p.section), ("num_students", "30"),
                      ("start_time", over.get("start", "09:00")), ("end_time", over.get("end", "16:00")),
                      ("num_periods", str(periods)), ("period_minutes", str(over.get("length", 40))),
                      ("att_count", "0")])
    for d in p.working_days:
        data.add("working_days", d)
    for name, after, mins in breaks:
        data.add("br_name", name); data.add("br_after", str(after)); data.add("br_minutes", str(mins))
    for s in over.get("subjects", p.subjects[:3]):
        data.add("subj_name", s.name); data.add("subj_load", s.load); data.add("subj_ppw", "2")
        data.add("subj_max", "1"); data.add("subj_teacher", "")
    return data


@pytest.fixture
def profile():
    p = service.new_profile(4, "A")
    storage.save_profile(p)
    return p


# ---------------- period count ----------------

def test_default_limits():
    assert config.MIN_PERIODS_PER_DAY == 1 and config.MAX_PERIODS_PER_DAY == 10


@pytest.mark.parametrize("n,ok", [(0, False), (1, True), (5, True), (8, True), (10, True), (11, False), (12, False)])
def test_period_count_bounds(profile, n, ok):
    assert (not period_count_errors(n)) == ok
    _, errors = parse_setup_form(setup_form(profile, n, []), profile)
    assert (not any("Number of periods" in e for e in errors)) == ok


def test_limit_is_configurable(profile, monkeypatch):
    monkeypatch.setattr(config, "MAX_PERIODS_PER_DAY", 6)
    assert period_count_errors(7) and not period_count_errors(6)


def test_legacy_profile_over_limit_cannot_generate(profile):
    profile.num_periods = 11
    profile.breaks = []
    profile.end_time = "18:00"
    with pytest.raises(ValueError, match="between 1 and 10 periods"):
        service.generate_proposal(profile, use_ai=False)


# ---------------- break positions ----------------

@pytest.mark.parametrize("n,expected", [(1, []), (3, [1, 2]), (4, [1, 2, 3]), (5, [1, 2, 3, 4]),
                                        (8, [1, 2, 3, 4, 5, 6, 7])])
def test_valid_positions_exclude_last_period(n, expected):
    assert valid_break_positions(n) == expected


def test_break_after_final_period_rejected_and_not_inserted(profile):
    profile.num_periods = 5
    profile.breaks = [BreakConfig(name="Late", after_period=5, minutes=20)]
    periods, breaks, errors = compute_schedule(profile)
    assert breaks == [] and periods[-1].end == 9 * 60 + 5 * 40
    assert any("Breaks can only follow periods 1-4" in e for e in errors)


def test_duplicate_and_too_many_breaks():
    errs = break_errors([BreakConfig(name="A", after_period=2, minutes=10),
                         BreakConfig(name="B", after_period=2, minutes=10)], 4)
    assert any("Only one break is allowed" in e for e in errs)
    errs = break_errors([BreakConfig(name=f"B{i}", after_period=i, minutes=10) for i in (1, 2, 3)], 3)
    assert any("room for at most 2" in e for e in errs)
    assert any("1-period" not in e for e in break_errors([BreakConfig(name="X", after_period=1, minutes=10)], 1))
    assert any("no gap" in e for e in break_errors([BreakConfig(name="X", after_period=1, minutes=10)], 1))


@pytest.mark.parametrize("mins,ok", [(4, False), (5, True), (120, True), (121, False)])
def test_break_duration_bounds(mins, ok):
    assert (not break_errors([BreakConfig(name="X", after_period=1, minutes=mins)], 4)) == ok


# ---------------- setup form (backend validation, independent of the UI) ----------------

def test_reducing_periods_keeps_breaks_and_explains(profile):
    # Lunch after period 4 is fine with 7 periods but not with 4.
    updated, errors = parse_setup_form(setup_form(profile, 4, [("Short break", 2, 15), ("Lunch", 4, 40)]), profile)
    assert any("'Lunch' is set after period 4" in e and "remove this break" in e for e in errors)
    assert [(b.name, b.after_period) for b in updated.breaks] == [("Short break", 2), ("Lunch", 4)]  # not discarded
    assert storage.load_profile(profile.id).num_periods == 7  # nothing saved


def test_add_rename_and_delete_breaks(profile):
    updated, errors = parse_setup_form(
        setup_form(profile, 6, [("Snack Break", 1, 10), ("Lunch", 3, 35), ("Activity Break", 5, 15)]), profile)
    assert not errors
    assert [(b.name, b.after_period, b.minutes) for b in updated.breaks] == [
        ("Snack Break", 1, 10), ("Lunch", 3, 35), ("Activity Break", 5, 15)]
    updated, errors = parse_setup_form(setup_form(profile, 6, [("Lunch", 3, 35)]), profile)
    assert not errors and len(updated.breaks) == 1
    updated, errors = parse_setup_form(setup_form(profile, 6, []), profile)
    assert not errors and updated.breaks == []


def test_form_rejects_duplicates_and_bad_values(profile):
    _, errors = parse_setup_form(setup_form(profile, 6, [("A", 2, 10), ("B", 2, 10)]), profile)
    assert any("Only one break is allowed" in e for e in errors)
    _, errors = parse_setup_form(setup_form(profile, 6, [("A", "x", "y")]), profile)
    assert any("'A' is set after period 0" in e for e in errors) and any("between 5 and 120" in e for e in errors)


# ---------------- end time ----------------

@pytest.mark.parametrize("start,n,length,breaks,end", [
    ("09:00", 7, 40, [("Short break", 2, 15), ("Lunch", 4, 40)], "2:35 PM"),
    ("08:00", 1, 45, [], "8:45 AM"),
    ("07:30", 8, 35, [("A", 1, 10), ("B", 4, 30), ("C", 7, 5)], "12:55 PM"),
    ("09:00", 4, 40, [("Lunch", 4, 40)], "11:40 AM"),  # invalid break ignored for the end time
])
def test_school_day_end(profile, start, n, length, breaks, end):
    profile.start_time, profile.num_periods, profile.period_minutes = start, n, length
    profile.breaks = [BreakConfig(name=a, after_period=b, minutes=c) for a, b, c in breaks]
    assert fmt12(school_day_end(profile)) == end
    periods, _, _ = compute_schedule(profile)
    assert fmt12(periods[-1].end) == end


def test_breaks_inserted_at_correct_positions(profile):
    profile.num_periods, profile.period_minutes, profile.start_time = 5, 40, "09:00"
    profile.breaks = [BreakConfig(name="Lunch", after_period=3, minutes=30),
                      BreakConfig(name="Short", after_period=1, minutes=10)]
    periods, breaks, errors = compute_schedule(profile)
    assert not [e for e in errors if "closing" not in e]
    assert [(b.name, b.after_period, fmt12(b.start)) for b in breaks] == [
        ("Short", 1, "9:40 AM"), ("Lunch", 3, "11:10 AM")]
    assert fmt12(periods[1].start) == "9:50 AM" and fmt12(periods[3].start) == "11:40 AM"


# ---------------- generated timetable respects the period count ----------------

@pytest.mark.parametrize("n", [1, 3, 8])
def test_timetable_uses_selected_period_count(profile, n):
    profile.num_periods = n
    profile.end_time = "17:00"
    profile.breaks = [BreakConfig(name="Lunch", after_period=n // 2, minutes=30)] if n > 1 else []
    for s in profile.subjects:
        s.periods_per_week, s.max_per_day = 0, 1
    for s in profile.subjects[: min(len(profile.subjects), n)]:
        s.periods_per_week = 5
    v = service.generate_proposal(profile, use_ai=False)
    assert v.validation.ok, v.validation.errors
    assert {c.period for c in v.cells} == set(range(1, n + 1))
    bad = [c.model_copy(update={"period": n + 1}) for c in v.cells[:1]]
    assert not validate_class_timetable(profile, v.cells + bad).ok


# ---------------- routes ----------------

def test_setup_route_rerenders_draft_without_saving(client, profile):
    r = client.post(f"/assistant/{profile.id}/setup",
                    data=setup_form(profile, 4, [("Short break", 2, 15), ("Lunch", 4, 40)]))
    assert r.status_code == 400
    html = r.data.decode()
    assert "Your changes have not been saved yet" in html
    assert 'value="Lunch"' in html and "Period 4 (invalid)" in html
    assert "School day ends at" in html
    assert storage.load_profile(profile.id).num_periods == 7

    ok = client.post(f"/assistant/{profile.id}/setup", data=setup_form(profile, 4, [("Short break", 2, 15)]))
    assert ok.status_code == 302 and storage.load_profile(profile.id).num_periods == 4


def test_setup_page_offers_only_valid_positions(client, profile):
    html = client.get(f"/assistant/?class={profile.id}&tab=setup").data.decode()
    assert '<option value="10"' in html and '<option value="11"' not in html  # periods dropdown 1..10
    assert "After period 6" in html and "After period 7" not in html  # 7 periods -> positions 1..6
    assert "School day ends at <strong>2:35 PM</strong>" in html
