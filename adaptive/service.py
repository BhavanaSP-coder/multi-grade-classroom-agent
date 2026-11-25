"""Orchestration for the Adaptive Classroom Assistant.

Rules this module enforces:
* AI-generated timetables are always PROPOSALS. Only ``approve()`` makes one current.
* Approval is refused while the validator reports errors.
* Teacher edits are explicit and immediately re-validated; nothing changes silently.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import config
from adaptive import storage
from adaptive.ai import get_advice, summarize
from adaptive.explain import advice_honoured, diff, explain, recommendations
from adaptive.learning import Insights, analyse
from adaptive.models import (
    DEFAULT_LOADS,
    FREE_LABEL,
    BreakConfig,
    Cell,
    ClassProfile,
    FeedbackEntry,
    SubjectConfig,
    TimetableVersion,
)
from adaptive.scheduler import build_problem, generate, score_assignment, assignment_from_cells
from adaptive.timing import BreakTime, PeriodTime, compute_schedule, default_attention_windows
from adaptive.validator import validate_class_timetable
from chains.llm import friendly_llm_error

log = logging.getLogger(__name__)
Key = tuple[str, int]
HISTORY_LIMIT = 10


@dataclass
class Context:
    periods: list[PeriodTime]
    breaks: list[BreakTime]
    timing_errors: list[str]
    insights: Insights
    external_busy: set[tuple[str, str, int]] = field(default_factory=set)


def new_profile(grade: int = 4, section: str = "A") -> ClassProfile:
    """Editable starting point. Attention starts at Moderate everywhere: the teacher sets it."""
    if grade <= 2:
        plan = [("Mathematics", 6, 2), ("Environmental Studies", 4, 1), ("English", 5, 1),
                ("Kannada", 6, 2), ("Hindi", 3, 1), ("Art and Craft", 3, 1),
                ("Physical Education", 4, 1), ("Music", 2, 1)]
    else:
        plan = [("Mathematics", 6, 2), ("Science", 4, 1), ("Social Studies", 3, 1), ("English", 5, 1),
                ("Kannada", 5, 1), ("Hindi", 3, 1), ("Art and Craft", 2, 1),
                ("Physical Education", 3, 1), ("Music", 1, 1)]
    p = ClassProfile(
        grade=grade, section=section, start_time="09:00", end_time="15:30",
        num_periods=7, period_minutes=40,
        breaks=[BreakConfig(name="Short break", after_period=2, minutes=15),
                BreakConfig(name="Lunch", after_period=4, minutes=40)],
        subjects=[SubjectConfig(name=n, load=DEFAULT_LOADS.get(n, "medium"), periods_per_week=w,
                                max_per_day=m) for n, w, m in plan],
    )
    p.attention = default_attention_windows(p.start_time, p.end_time)
    return p


def external_busy_for(profile: ClassProfile) -> set[tuple[str, str, int]]:
    """Teachers already teaching another class at (day, period) in its APPROVED timetable."""
    busy = set()
    for other in storage.list_profiles():
        if other.id == profile.id or not other.current:
            continue
        teachers = {s.name: s.teacher for s in other.subjects if s.teacher}
        for c in other.current.cells:
            t = teachers.get(c.subject)
            if t:
                busy.add((t, c.day, c.period))
    return busy


def context_for(profile: ClassProfile, include_external: bool = True) -> Context:
    periods, breaks, errors = compute_schedule(profile)
    insights = analyse(profile, periods)
    ext = external_busy_for(profile) if include_external else set()
    return Context(periods, breaks, errors, insights, ext)


def attention_is_flat(profile: ClassProfile) -> bool:
    return len({w.level for w in profile.attention}) <= 1


def _refresh_cells(profile: ClassProfile, cells: list[Cell], ctx: Context) -> list[Cell]:
    """Recompute kind/load/attention after edits or settings changes."""
    subjects = {s.name: s for s in profile.subjects}
    fixed = {(f.day, f.period): f.label for f in profile.fixed}
    att = {p.index: ctx.insights.period_effective.get(p.index, p.base_attention) for p in ctx.periods}
    out = []
    for c in cells:
        c = c.model_copy()
        if (c.day, c.period) in fixed and c.subject == fixed[(c.day, c.period)]:
            c.kind, c.load = "fixed", None
        elif c.subject in subjects:
            c.kind, c.load = "subject", subjects[c.subject].load
        else:
            c.kind, c.load = "free", None
        c.attention = att.get(c.period, c.attention)
        out.append(c)
    return out


def _evaluate(profile: ClassProfile, version: TimetableVersion, ctx: Context,
              locked_ref: dict[Key, str] | None) -> TimetableVersion:
    version.cells = _refresh_cells(profile, version.cells, ctx)
    v = validate_class_timetable(profile, version.cells, locked_ref=locked_ref,
                                 external_busy=ctx.external_busy, periods=ctx.periods)
    v.errors = ctx.timing_errors + v.errors
    v.ok = not v.errors
    problem = build_problem(profile, ctx.periods, ctx.insights,
                            advice={a.subject: a.preferred_time for a in version.advice},
                            external_busy=ctx.external_busy)
    v.score, breakdown = score_assignment(
        {k: s for k, s in assignment_from_cells(version.cells).items() if k in problem.slots}, problem)
    v.metrics.update({f"score_{k}": round(val, 2) for k, val in breakdown.items()})
    version.validation, version.score = v, v.score
    version.advice = advice_honoured(version.advice, version.cells, ctx.periods)
    version.explanations = explain(profile, version.cells, ctx.periods, ctx.insights, version.advice)
    return version


def locked_from_current(profile: ClassProfile) -> dict[Key, str]:
    if not profile.current:
        return {}
    return {(c.day, c.period): c.subject for c in profile.current.cells if c.locked}


def generate_proposal(profile: ClassProfile, use_ai: bool | None = None,
                      regenerate: bool = False) -> TimetableVersion:
    """Create a new proposed timetable (saved on the profile; the current one is untouched)."""
    use_ai = config.ADAPTIVE_USE_AI if use_ai is None else use_ai
    ctx = context_for(profile)
    if ctx.timing_errors:
        raise ValueError("Fix the class timings first: " + " ".join(ctx.timing_errors))
    if not any(s.periods_per_week > 0 for s in profile.subjects):
        raise ValueError("Add at least one subject with periods per week before generating.")

    notes: list[str] = []
    advice = []
    if use_ai:
        try:
            advice, ai_notes = get_advice(profile, ctx.periods, ctx.insights)
            notes += ai_notes
        except Exception as exc:  # noqa: BLE001 - AI is optional, the optimiser is not
            log.warning("Adaptive advice failed: %s", exc)
            notes.append("AI advice unavailable (" + friendly_llm_error(exc) + "). "
                         "The timetable was optimised from your data only.")

    locked = locked_from_current(profile)
    problem = build_problem(profile, ctx.periods, ctx.insights,
                            advice={a.subject: a.preferred_time for a in advice},
                            locked=locked, external_busy=ctx.external_busy)
    seed = random.randint(1, 10_000) if regenerate else 7
    result = generate(problem, seed=seed)

    version = TimetableVersion(cells=result.cells, advice=advice,
                               alternatives_scores=result.alternatives,
                               notes=result.conflicts + notes)
    version = _evaluate(profile, version, ctx, locked)
    version.changes = diff(profile.current.cells if profile.current else None, version.cells,
                           ctx.insights, ctx.periods)
    if attention_is_flat(profile):
        version.notes.insert(0, "Every period has the same attention level, so the plan mainly "
                                "balances subjects. Set your class's attention pattern for better results.")

    if use_ai and version.validation.ok:
        try:
            version.ai_summary, version.ai_dropped = summarize(profile, version.cells, version.explanations)
        except Exception as exc:  # noqa: BLE001
            log.warning("Adaptive summary failed: %s", exc)
    profile.proposal = version
    storage.save_profile(profile)
    return version


def approve(profile: ClassProfile) -> None:
    if not profile.proposal:
        raise ValueError("There is no proposed timetable to approve.")
    ctx = context_for(profile)
    proposal = _evaluate(profile, profile.proposal, ctx, locked_from_current(profile))
    if not proposal.validation.ok:
        profile.proposal = proposal
        storage.save_profile(profile)
        raise ValueError("This proposal breaks constraints and cannot be approved: "
                         + " ".join(proposal.validation.errors[:3]))
    if profile.current:
        profile.history = ([profile.current] + profile.history)[:HISTORY_LIMIT]
    proposal.status = "approved"
    profile.current, profile.proposal = proposal, None
    storage.save_profile(profile)


def reject(profile: ClassProfile) -> None:
    if profile.proposal:
        profile.proposal.status = "rejected"
        profile.history = ([profile.proposal] + profile.history)[:HISTORY_LIMIT]
        profile.proposal = None
        storage.save_profile(profile)


def apply_edit(profile: ClassProfile, target: str, action: str, a: Key,
               b: Key | None = None, subject: str | None = None) -> TimetableVersion:
    """Teacher edit on 'current' or 'proposal': swap two periods, set a subject, or toggle a lock."""
    version = profile.current if target == "current" else profile.proposal
    if version is None:
        raise ValueError("There is no timetable to edit yet. Generate one first.")
    version = version.model_copy(deep=True)
    by_key = {(c.day, c.period): c for c in version.cells}
    if a not in by_key or (b is not None and b not in by_key):
        raise ValueError("That period does not exist in this timetable.")
    fixed = {(f.day, f.period) for f in profile.fixed}
    ca = by_key[a]

    if action == "swap":
        if b is None or b == a:
            raise ValueError("Choose two different periods to swap.")
        cb = by_key[b]
        if a in fixed or b in fixed:
            raise ValueError("Fixed activities cannot be swapped. Change them in Setup.")
        if ca.locked or cb.locked:
            raise ValueError("Unlock a period before swapping it.")
        ca.subject, cb.subject = cb.subject, ca.subject
    elif action == "set":
        valid = {s.name for s in profile.subjects} | {FREE_LABEL}
        if subject not in valid:
            raise ValueError("Choose one of this class's subjects.")
        if a in fixed:
            raise ValueError("This is a fixed activity. Change it in Setup.")
        if ca.locked:
            raise ValueError("Unlock this period before changing it.")
        ca.subject = subject
    elif action == "lock":
        if a in fixed:
            raise ValueError("Fixed activities are always kept; no lock needed.")
        ca.locked = not ca.locked
    else:
        raise ValueError("Unknown edit action.")

    ctx = context_for(profile)
    locked_ref = {k: c.subject for k, c in by_key.items() if c.locked} if action != "lock" else None
    version.source = "edited"
    version.ai_summary = []
    version.changes = []
    version = _evaluate(profile, version, ctx, locked_ref)
    if target == "current":
        profile.current = version
    else:
        profile.proposal = version
    storage.save_profile(profile)
    return version


def revalidate(profile: ClassProfile) -> None:
    """Re-check stored timetables after settings change (shows conflicts instead of hiding them)."""
    ctx = context_for(profile)
    for attr in ("current", "proposal"):
        v = getattr(profile, attr)
        if v:
            locked = {(c.day, c.period): c.subject for c in v.cells if c.locked}
            setattr(profile, attr, _evaluate(profile, v, ctx, locked))


def add_feedback(profile: ClassProfile, entry: FeedbackEntry) -> None:
    profile.feedback.append(entry)
    storage.save_profile(profile)


def delete_feedback(profile: ClassProfile, entry_id: str) -> bool:
    before = len(profile.feedback)
    profile.feedback = [f for f in profile.feedback if f.id != entry_id]
    if len(profile.feedback) != before:
        storage.save_profile(profile)
        return True
    return False


def dashboard_recommendations(profile: ClassProfile, ctx: Context) -> list[str]:
    return recommendations(profile, ctx.insights, ctx.periods)
