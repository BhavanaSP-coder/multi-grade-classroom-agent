"""Parse and validate the Adaptive Assistant forms (setup + feedback)."""
from __future__ import annotations

import re
from datetime import date
from typing import Any

import config
from adaptive.models import (
    ALL_DAYS,
    AttentionWindow,
    BreakConfig,
    ClassProfile,
    FeedbackEntry,
    FixedActivity,
    SubjectConfig,
    TeacherUnavailability,
)
from adaptive.timing import compute_schedule, fmt, parse_hhmm, realign_windows, school_day_end


def _get(form: Any, key: str) -> str:
    return (form.get(key) or "").strip()


def _list(form: Any, key: str) -> list[str]:
    return [v.strip() for v in form.getlist(key)] if hasattr(form, "getlist") else list(form.get(key, []))


def _int(value: str, name: str, lo: int, hi: int, errors: list[str]) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a whole number.")
        return None
    if not lo <= n <= hi:
        errors.append(f"{name} must be between {lo} and {hi}.")
        return None
    return n


def _lenient_int(value: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def parse_period_list(text: str, n_periods: int) -> list[int]:
    """'1,2' or '1-3, 6' -> [1, 2, 3, 6]."""
    out: set[int] = set()
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            out.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.add(int(part))
    bad = [p for p in out if not 1 <= p <= n_periods]
    if bad:
        raise ValueError(f"period {bad[0]} does not exist (1-{n_periods}).")
    return sorted(out)


def parse_setup_form(form: Any, profile: ClassProfile) -> tuple[ClassProfile, list[str]]:
    """Return (updated copy, errors). The original profile is not modified."""
    p = profile.model_copy(deep=True)
    errors: list[str] = []

    # --- A. class information
    g = _int(_get(form, "grade"), "Grade", 1, 12, errors)
    if g is not None:
        p.grade = g
    p.section = _get(form, "section")[:10] or "A"
    n = _int(_get(form, "num_students"), "Number of students", 1, 200, errors)
    if n is not None:
        p.num_students = n
    # The closing time is normally derived from the timetable (see below); an explicit
    # end_time is still accepted for older forms and scripts.
    explicit_end = form.get("end_time") is not None
    for key, label in (("start_time", "School start time"), ("end_time", "School closing time")):
        if key == "end_time" and not explicit_end:
            continue
        val = _get(form, key)
        try:
            parse_hhmm(val)
            setattr(p, key, val if len(val) == 5 else f"0{val}")
        except ValueError:
            errors.append(f"{label}: use HH:MM, for example 08:30.")
    days = [d for d in ALL_DAYS if d in _list(form, "working_days")]
    if not days:
        errors.append("Select at least one working day.")
    else:
        p.working_days = days
    np_ = _int(_get(form, "num_periods"), "Number of periods", config.MIN_PERIODS_PER_DAY,
               config.MAX_PERIODS_PER_DAY, errors)
    if np_ is not None:
        p.num_periods = np_
    pm = _int(_get(form, "period_minutes"), "Period duration", 20, 120, errors)
    if pm is not None:
        p.period_minutes = pm

    # Every submitted break row is KEPT (even if invalid) so a rejected form can be shown back to
    # the teacher unchanged; break_errors() (via compute_schedule below) explains each problem.
    breaks = []
    for name, after, mins in zip(_list(form, "br_name"), _list(form, "br_after"), _list(form, "br_minutes")):
        if not (name or after or mins):
            continue
        breaks.append(BreakConfig(name=" ".join(name.split())[:30] or "Break",
                                  after_period=_lenient_int(after), minutes=_lenient_int(mins)))
    p.breaks = breaks
    if not explicit_end:
        end = school_day_end(p)
        if end is not None and end < 24 * 60:
            p.end_time = fmt(end)

    # --- B. subjects
    subjects, seen = [], set()
    rows = zip(_list(form, "subj_name"), _list(form, "subj_load"), _list(form, "subj_ppw"),
               _list(form, "subj_max"), _list(form, "subj_teacher"))
    for name, load, ppw, mx, teacher in rows:
        name = " ".join(name.split())[:40]
        if not name:
            continue
        if name.lower() in seen:
            errors.append(f"Subject '{name}' is listed twice.")
            continue
        seen.add(name.lower())
        if load not in ("high", "medium", "low"):
            errors.append(f"{name}: choose High, Medium or Low cognitive load.")
            continue
        w = _int(ppw, f"{name} - periods per week", 0, 40, errors)
        m = _int(mx, f"{name} - max per day", 1, 4, errors)
        if w is None or m is None:
            continue
        subjects.append(SubjectConfig(name=name, load=load, periods_per_week=w, max_per_day=m,
                                      teacher=" ".join(teacher.split())[:40]))
    if not subjects:
        errors.append("Add at least one subject.")
    p.subjects = subjects

    # --- C. attention pattern (teacher observations)
    windows = []
    try:
        count = int(_get(form, "att_count") or 0)
    except ValueError:
        count = 0
    for i in range(count):
        s, e, lvl = _get(form, f"att_start_{i}"), _get(form, f"att_end_{i}"), _get(form, f"att_level_{i}")
        try:
            parse_hhmm(s), parse_hhmm(e)
            level = int(lvl)
            if not 1 <= level <= 5:
                raise ValueError
        except ValueError:
            errors.append(f"Attention row {i + 1} has an invalid time or level.")
            continue
        windows.append(AttentionWindow(start=s, end=e, level=level))
    if (p.start_time, p.end_time) != (profile.start_time, profile.end_time) or not windows:
        windows = realign_windows(windows or profile.attention, p.start_time, p.end_time)
    p.attention = windows

    # --- teacher availability
    teachers = {s.teacher.lower() for s in subjects if s.teacher}
    unav = []
    for t, d, per in zip(_list(form, "un_teacher"), _list(form, "un_day"), _list(form, "un_periods")):
        if not (t or per):
            continue
        if not t or d not in ALL_DAYS:
            errors.append("Each unavailability row needs a teacher name and a day.")
            continue
        if t.lower() not in teachers:
            errors.append(f"Unavailability: no subject is taught by '{t}'. Use the same name as in "
                          "the subjects table.")
            continue
        try:
            unav.append(TeacherUnavailability(teacher=t, day=d, periods=parse_period_list(per, p.num_periods)))
        except ValueError as exc:
            errors.append(f"Unavailability for {t} on {d}: {exc}")
    p.unavailability = unav

    # --- fixed activities
    fixed, used = [], set()
    for d, per, label in zip(_list(form, "fx_day"), _list(form, "fx_period"), _list(form, "fx_label")):
        if not (label or per):
            continue
        k = _int(per, f"Fixed activity '{label}' - period", 1, p.num_periods, errors)
        if d not in ALL_DAYS or k is None or not label:
            if label and d not in ALL_DAYS:
                errors.append(f"Fixed activity '{label}': choose a day.")
            continue
        if d not in p.working_days:
            errors.append(f"Fixed activity '{label}' is on {d}, which is not a working day.")
            continue
        if (d, k) in used:
            errors.append(f"Two fixed activities are set for {d} period {k}.")
            continue
        used.add((d, k))
        fixed.append(FixedActivity(day=d, period=k, label=" ".join(label.split())[:40]))
    p.fixed = fixed

    _, _, timing_errors = compute_schedule(p)
    errors += timing_errors
    errors = list(dict.fromkeys(errors))
    if not errors:
        p.breaks.sort(key=lambda b: b.after_period)  # store in day order
    return p, errors


def parse_feedback_form(form: Any, profile: ClassProfile) -> tuple[FeedbackEntry | None, list[str]]:
    errors: list[str] = []
    day = _get(form, "day")
    if day not in profile.working_days:
        errors.append("Choose a working day.")
    period = _int(_get(form, "period"), "Period", 1, profile.num_periods, errors)
    subject = _get(form, "subject")
    if subject not in {s.name for s in profile.subjects}:
        errors.append("Choose one of this class's subjects.")
    scores = {}
    for key, label in (("understanding", "Understanding"), ("engagement", "Engagement"),
                       ("concentration", "Concentration")):
        scores[key] = _int(_get(form, key), label, 1, 5, errors)
    d = _get(form, "date") or date.today().isoformat()
    try:
        date.fromisoformat(d)
    except ValueError:
        errors.append("Date must look like 2026-11-25.")
    if errors:
        return None, errors
    return FeedbackEntry(date=d, day=day, period=period, subject=subject,
                         took_longer=_get(form, "took_longer") in ("1", "on", "true"),
                         successful_activity=_get(form, "successful_activity")[:300],
                         notes=_get(form, "notes")[:500], **scores), []
