"""Clock arithmetic: period/break times, attention per period, time-of-day buckets."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import config
from adaptive.models import AttentionWindow, ClassProfile

_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def parse_hhmm(value: str) -> int:
    m = _HHMM.match((value or "").strip())
    if not m:
        raise ValueError(f"'{value}' is not a valid time (use HH:MM, e.g. 08:30).")
    return int(m.group(1)) * 60 + int(m.group(2))


def fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def fmt12(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    suffix = "AM" if h < 12 else "PM"
    return f"{(h - 1) % 12 + 1}:{m:02d} {suffix}"


@dataclass
class PeriodTime:
    index: int
    start: int
    end: int
    base_attention: float
    bucket: str
    attention_covered: bool = True

    @property
    def label(self) -> str:
        return f"{fmt(self.start)}-{fmt(self.end)}"


@dataclass
class BreakTime:
    name: str
    start: int
    end: int
    after_period: int

    @property
    def label(self) -> str:
        return f"{fmt(self.start)}-{fmt(self.end)}"


def attention_for(start: int, end: int, windows: list[AttentionWindow]) -> tuple[float, bool]:
    """Overlap-weighted average of teacher-entered levels; (3.0, False) if no window covers it."""
    total, weight = 0.0, 0
    for w in windows:
        try:
            ws, we = parse_hhmm(w.start), parse_hhmm(w.end)
        except ValueError:
            continue
        overlap = min(end, we) - max(start, ws)
        if overlap > 0:
            total += overlap * w.level
            weight += overlap
    if not weight:
        return 3.0, False
    return round(total / weight, 2), weight >= (end - start) * 0.5


def lunch_after(profile: ClassProfile) -> int | None:
    """Period after which lunch happens: a break named 'lunch', else the longest break >= 20 min."""
    named = [b for b in profile.breaks if "lunch" in b.name.lower()]
    if named:
        return named[0].after_period
    longest = max(profile.breaks, key=lambda b: b.minutes, default=None)
    return longest.after_period if longest and longest.minutes >= 20 else None


def bucket_for(period: int, profile: ClassProfile) -> str:
    lunch = lunch_after(profile)
    pre_lunch = lunch if lunch else profile.num_periods
    if lunch and period > lunch:
        return "afternoon"
    other_breaks = sorted(b.after_period for b in profile.breaks if b.after_period != lunch
                          and b.after_period < pre_lunch)
    early_end = other_breaks[0] if other_breaks else math.ceil(pre_lunch / 2)
    return "early_morning" if period <= early_end else "late_morning"


def valid_break_positions(num_periods: int) -> list[int]:
    """A break can follow any period except the last one (that period ends the day)."""
    return list(range(1, max(1, num_periods)))


def period_count_errors(num_periods: int) -> list[str]:
    lo, hi = config.MIN_PERIODS_PER_DAY, config.MAX_PERIODS_PER_DAY
    if not lo <= num_periods <= hi:
        return [f"The school day must have between {lo} and {hi} periods (currently {num_periods})."]
    return []


def break_errors(breaks, num_periods: int) -> list[str]:
    """Position, duplicate and duration checks for breaks (shared by forms, scheduler, validator)."""
    errors: list[str] = []
    allowed = valid_break_positions(num_periods)
    seen: dict[int, str] = {}
    for b in breaks:
        label = b.name or "Break"
        if b.after_period not in allowed:
            if not allowed:
                errors.append(f"'{label}' cannot be scheduled: with only {num_periods} period there is "
                              "no gap between periods. Remove the break or add periods.")
            else:
                errors.append(f"'{label}' is set after period {b.after_period}, but the day has "
                              f"{num_periods} periods. Breaks can only follow periods 1-{allowed[-1]}; "
                              "choose another position or remove this break.")
        elif b.after_period in seen:
            errors.append(f"'{label}' and '{seen[b.after_period]}' are both after period "
                          f"{b.after_period}. Only one break is allowed in each position.")
        else:
            seen[b.after_period] = label
        if not config.MIN_BREAK_MINUTES <= b.minutes <= config.MAX_BREAK_MINUTES:
            errors.append(f"'{label}' must last between {config.MIN_BREAK_MINUTES} and "
                          f"{config.MAX_BREAK_MINUTES} minutes.")
    if len(breaks) > len(allowed) and allowed:
        errors.append(f"{len(breaks)} breaks are set but a {num_periods}-period day has room for at most "
                      f"{len(allowed)}.")
    return errors


def school_day_end(profile: ClassProfile) -> int | None:
    """Minutes after midnight when the last period ends (valid breaks only), or None."""
    try:
        t = parse_hhmm(profile.start_time)
    except ValueError:
        return None
    allowed = set(valid_break_positions(profile.num_periods))
    t += profile.num_periods * profile.period_minutes
    t += sum(b.minutes for b in profile.breaks if b.after_period in allowed)
    return t


def compute_schedule(profile: ClassProfile) -> tuple[list[PeriodTime], list[BreakTime], list[str]]:
    """Lay out periods and breaks from the start time. Returns (periods, breaks, errors)."""
    errors: list[str] = []
    try:
        start = parse_hhmm(profile.start_time)
        close = parse_hhmm(profile.end_time)
    except ValueError as exc:
        return [], [], [str(exc)]
    if close <= start:
        errors.append("School closing time must be after the start time.")
    errors += period_count_errors(profile.num_periods)
    errors += break_errors(profile.breaks, profile.num_periods)

    periods: list[PeriodTime] = []
    breaks: list[BreakTime] = []
    t = start
    for p in range(1, profile.num_periods + 1):
        att, covered = attention_for(t, t + profile.period_minutes, profile.attention)
        periods.append(PeriodTime(p, t, t + profile.period_minutes, att, bucket_for(p, profile), covered))
        t += profile.period_minutes
        if p == profile.num_periods:
            break  # nothing is inserted after the final period: it ends the school day
        for b in sorted((b for b in profile.breaks if b.after_period == p), key=lambda b: b.name):
            breaks.append(BreakTime(b.name, t, t + max(0, b.minutes), p))
            t += max(0, b.minutes)

    if close > start and t > close:
        errors.append(
            f"{profile.num_periods} periods of {profile.period_minutes} min plus breaks end at "
            f"{fmt(t)}, after the closing time {profile.end_time}. Reduce periods, shorten "
            f"periods or breaks, or set a later closing time.")
    return periods, breaks, errors


def default_attention_windows(start_time: str, end_time: str, level: int = 3) -> list[AttentionWindow]:
    """Hourly windows covering the school day (teacher then sets the levels)."""
    try:
        s, e = parse_hhmm(start_time), parse_hhmm(end_time)
    except ValueError:
        return []
    out, t = [], s
    while t < e:
        nxt = min(e, (t // 60 + 1) * 60)
        if nxt - t < 15 and out:  # merge a tiny tail into the previous window
            out[-1].end = fmt(nxt)
        else:
            out.append(AttentionWindow(start=fmt(t), end=fmt(nxt), level=level))
        t = nxt
    return out


def realign_windows(old: list[AttentionWindow], start_time: str, end_time: str) -> list[AttentionWindow]:
    """New hourly windows for changed school hours, keeping the teacher's levels where they overlap."""
    fresh = default_attention_windows(start_time, end_time)
    for w in fresh:
        level, covered = attention_for(parse_hhmm(w.start), parse_hhmm(w.end), old)
        if covered:
            w.level = int(round(level))
    return fresh
