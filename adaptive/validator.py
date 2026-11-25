"""Deterministic validation of a class timetable. This, not the AI, decides validity."""
from __future__ import annotations

from collections import Counter

from adaptive.models import FREE_LABEL, AdaptiveValidation, Cell, ClassProfile, attention_label
from adaptive.timing import PeriodTime, compute_schedule

Key = tuple[str, int]


def _blocks(profile: ClassProfile) -> dict[int, int]:
    after = {b.after_period for b in profile.breaks}
    block, out = 0, {}
    for p in range(1, profile.num_periods + 1):
        out[p] = block
        if p in after:
            block += 1
    return out


def validate_class_timetable(profile: ClassProfile, cells: list[Cell],
                             locked_ref: dict[Key, str] | None = None,
                             external_busy: set[tuple[str, str, int]] | None = None,
                             periods: list[PeriodTime] | None = None) -> AdaptiveValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if periods is None:
        periods, _, timing_errors = compute_schedule(profile)
        errors += timing_errors

    days = profile.working_days
    grid = {(d, p) for d in days for p in range(1, profile.num_periods + 1)}
    subjects = {s.name: s for s in profile.subjects if s.periods_per_week > 0}
    fixed = {(f.day, f.period): f.label for f in profile.fixed}

    # --- structure: one entry per teaching period, nothing outside the grid (breaks / after hours)
    seen = Counter((c.day, c.period) for c in cells)
    for key, n in seen.items():
        if n > 1:
            errors.append(f"Overlap: {n} entries share {key[0]} period {key[1]}.")
        if key not in grid:
            errors.append(f"{key[0]} period {key[1]} is not a teaching period (it would fall in a "
                          "break, outside working days or after the last period).")
    for key in sorted(grid - set(seen), key=lambda k: (days.index(k[0]), k[1])):
        errors.append(f"{key[0]} period {key[1]} is empty.")

    by_key = {(c.day, c.period): c for c in cells}

    # --- fixed activities and locks
    for key, label in fixed.items():
        if key in grid and by_key.get(key) and by_key[key].subject != label:
            errors.append(f"Fixed activity '{label}' on {key[0]} P{key[1]} was replaced by "
                          f"'{by_key[key].subject}'.")
    for key, label in (locked_ref or {}).items():
        if key in by_key and by_key[key].subject != label:
            errors.append(f"Locked period {key[0]} P{key[1]} must stay '{label}' but is "
                          f"'{by_key[key].subject}'.")

    # --- subjects: known, frequency, per-day limit
    counts = Counter(c.subject for c in cells if c.subject in subjects)
    for c in cells:
        if c.subject not in subjects and c.subject != FREE_LABEL and fixed.get((c.day, c.period)) != c.subject:
            errors.append(f"{c.day} P{c.period}: '{c.subject}' is not one of this class's subjects.")
    for name, s in subjects.items():
        n = counts.get(name, 0)
        if n < s.periods_per_week:
            errors.append(f"{name} has {n} of the required {s.periods_per_week} periods per week.")
        elif n > s.periods_per_week:
            errors.append(f"{name} appears {n} times but only {s.periods_per_week} periods per week "
                          "are required (unnecessary repetition).")
    per_day = Counter((c.subject, c.day) for c in cells if c.subject in subjects)
    for (name, day), n in per_day.items():
        if n > subjects[name].max_per_day:
            errors.append(f"{name} is scheduled {n} times on {day}; the limit is "
                          f"{subjects[name].max_per_day} per day.")

    # --- teacher availability
    unavailable = {(u.teacher.strip().lower(), u.day, p) for u in profile.unavailability for p in u.periods}
    for t, d, p in external_busy or set():
        unavailable.add((t.strip().lower(), d, p))
    for c in cells:
        s = subjects.get(c.subject)
        if s and s.teacher and (s.teacher.strip().lower(), c.day, c.period) in unavailable:
            errors.append(f"{c.day} P{c.period}: teacher {s.teacher} ({c.subject}) is not available then.")

    # --- quality (soft) checks
    blocks = _blocks(profile)
    att = {p.index: p.base_attention for p in periods}
    eff = {(c.day, c.period): c.attention for c in cells} if cells else {}
    consecutive = 0
    for d in days:
        row = [by_key.get((d, p)) for p in range(1, profile.num_periods + 1)]
        for a, b in zip(row, row[1:]):
            if a and b and a.load == "high" and b.load == "high" and blocks.get(a.period) == blocks.get(b.period):
                consecutive += 1
                warnings.append(f"{d}: {a.subject} (P{a.period}) and {b.subject} (P{b.period}) are both "
                                "high-load and back-to-back with no break.")

    highs = [c for c in cells if c.load == "high"]
    in_high = sum(1 for c in highs if eff.get((c.day, c.period), att.get(c.period, 3)) >= 3.5)
    for c in highs:
        a = eff.get((c.day, c.period), att.get(c.period, 3))
        if a > 2.5:
            continue
        s = subjects[c.subject]
        for o in cells:
            oa = eff.get((o.day, o.period), att.get(o.period, 3))
            if oa < 3.5 or o.load == "high" or o.kind == "fixed" or o.locked or c.locked:
                continue
            if s.teacher and (s.teacher.strip().lower(), o.day, o.period) in unavailable:
                continue
            if o.day != c.day and per_day.get((c.subject, o.day), 0) >= s.max_per_day:
                continue
            warnings.append(
                f"{c.subject} on {c.day} P{c.period} is in a {attention_label(a)} attention period while "
                f"{o.subject} has {o.day} P{o.period} ({attention_label(oa)}). Consider swapping them.")
            break

    metrics = {
        "high_load_periods": len(highs),
        "high_load_in_high_attention": in_high,
        "consecutive_high_pairs": consecutive,
        "free_periods": sum(1 for c in cells if c.subject == FREE_LABEL),
    }
    return AdaptiveValidation(ok=not errors, errors=errors, warnings=warnings, metrics=metrics)
