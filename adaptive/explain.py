"""Explanations derived from the ACTUAL placements and the teacher's data (no generic text)."""
from __future__ import annotations

from collections import defaultdict

from adaptive.learning import MIN_OBS, Insights
from adaptive.models import (
    BUCKET_LABELS,
    FREE_LABEL,
    LOAD_LABELS,
    AdviceItem,
    Cell,
    Change,
    ClassProfile,
    Explanation,
    attention_label,
)
from adaptive.timing import PeriodTime


def ref(c: Cell) -> str:
    return f"{c.day} P{c.period}"


def _blocks(profile: ClassProfile) -> dict[int, int]:
    after = {b.after_period for b in profile.breaks}
    block, out = 0, {}
    for p in range(1, profile.num_periods + 1):
        out[p] = block
        if p in after:
            block += 1
    return out


def explain(profile: ClassProfile, cells: list[Cell], periods: list[PeriodTime],
            insights: Insights, advice: list[AdviceItem] | None = None) -> list[Explanation]:
    out: list[Explanation] = []
    ptime = {p.index: p for p in periods}
    by_subject: dict[str, list[Cell]] = defaultdict(list)
    for c in cells:
        if c.kind == "subject":
            by_subject[c.subject].append(c)

    # Attention pattern the plan is based on
    top = sorted(periods, key=lambda p: -insights.period_effective.get(p.index, p.base_attention))[:2]
    if top:
        out.append(Explanation(kind="placement", text=(
            "Your attention pattern ranks "
            + " and ".join(f"period {p.index} ({p.label}, {attention_label(insights.period_effective.get(p.index, p.base_attention))})" for p in top)
            + " as the most attentive times of the day; demanding subjects were given priority there.")))

    high_slots = sum(1 for c in cells if c.attention >= 3.5 and c.kind != "fixed")
    for load in ("high", "medium", "low"):
        for s in [s for s in profile.subjects if s.load == load and by_subject.get(s.name)]:
            placed = sorted(by_subject[s.name], key=lambda c: (profile.working_days.index(c.day), c.period))
            refs = [ref(c) for c in placed]
            labels = [attention_label(c.attention) for c in placed]
            if load == "high":
                good = [c for c in placed if c.attention >= 3.5]
                text = (f"{s.name} ({LOAD_LABELS[load]} load): {len(good)} of {len(placed)} periods are in "
                        f"High/Very High attention periods ("
                        + ", ".join(f"{r} {l}" for r, l in zip(refs, labels)) + ").")
                weak = [c for c in placed if c.attention < 3.5]
                if weak:
                    total_high = sum(len(by_subject[x.name]) for x in profile.subjects if x.load == "high")
                    if total_high > high_slots:
                        why = (f"there are {total_high} high-load periods but only {high_slots} "
                               "high-attention periods in the week")
                    elif s.teacher and any(u.teacher.strip().lower() == s.teacher.strip().lower()
                                           for u in profile.unavailability):
                        why = f"teacher {s.teacher} is unavailable in some attentive periods"
                    else:
                        why = (f"of the limit of {s.max_per_day} per day and to avoid back-to-back "
                               "demanding periods")
                    text += f" {', '.join(ref(c) for c in weak)} could not be moved earlier because {why}."
                out.append(Explanation(kind="placement", text=text, cells=refs))
            elif load == "low":
                avg = sum(c.attention for c in placed) / len(placed)
                buckets = sorted({ptime[c.period].bucket for c in placed if c.period in ptime})
                out.append(Explanation(kind="placement", text=(
                    f"{s.name} ({LOAD_LABELS[load]} load, activity-oriented) is in {', '.join(refs)} - "
                    f"average attention {attention_label(avg)}, mostly "
                    f"{' / '.join(BUCKET_LABELS[b] for b in buckets)}. Hands-on subjects suit "
                    "lower-concentration periods and keep energy up."), cells=refs))
            else:
                avg = sum(c.attention for c in placed) / len(placed)
                out.append(Explanation(kind="placement", text=(
                    f"{s.name} ({LOAD_LABELS[load]} load) is in {', '.join(refs)} - average attention "
                    f"{attention_label(avg)}, between the demanding and the activity periods."), cells=refs))

    # Consecutive demanding periods
    blocks = _blocks(profile)
    by_key = {(c.day, c.period): c for c in cells}
    back_to_back, split_by_break = [], []
    for d in profile.working_days:
        for p in range(1, profile.num_periods):
            a, b = by_key.get((d, p)), by_key.get((d, p + 1))
            if a and b and a.load == "high" and b.load == "high":
                (back_to_back if blocks[p] == blocks[p + 1] else split_by_break).append((a, b))
    if back_to_back:
        out.append(Explanation(kind="warning", text=(
            f"{len(back_to_back)} pair(s) of high-load subjects are back-to-back without a break ("
            + "; ".join(f"{ref(a)} {a.subject} -> {ref(b)} {b.subject}" for a, b in back_to_back)
            + "). The constraints left no better option; consider a short activity break between them."),
            cells=[ref(x) for pair in back_to_back for x in pair]))
    elif any(c.load == "high" for c in cells):
        out.append(Explanation(kind="balance", text=(
            "No two high-load subjects are back-to-back without a break, to reduce cognitive fatigue.")))
    if split_by_break:
        out.append(Explanation(kind="balance", text=(
            "High-load subjects in neighbouring periods are separated by a break: "
            + "; ".join(f"{ref(a)} {a.subject} | break | {ref(b)} {b.subject}" for a, b in split_by_break[:4])
            + "."), cells=[ref(x) for pair in split_by_break[:4] for x in pair]))

    # Feedback that shaped the plan
    for (subject, bucket), (avg, n) in insights.affinity.items():
        if n < MIN_OBS or subject not in by_subject or not (avg <= 2.5 or avg >= 4):
            continue
        there = [c for c in by_subject[subject] if ptime.get(c.period) and ptime[c.period].bucket == bucket]
        verb = "avoided" if avg <= 2.5 else "favoured"
        out.append(Explanation(kind="feedback", text=(
            f"Feedback: {subject} {BUCKET_LABELS[bucket]} averaged {avg:.1f}/5 understanding over {n} "
            f"lessons, so that time was {verb}. It now has {len(there)} of {len(by_subject[subject])} "
            f"{subject} periods {BUCKET_LABELS[bucket]}."), cells=[ref(c) for c in there]))
    for p in periods:
        observed = insights.period_observed.get(p.index)
        if observed and observed[1] >= MIN_OBS and abs(observed[0] - p.base_attention) >= 1:
            out.append(Explanation(kind="feedback", text=(
                f"Feedback adjusted period {p.index} ({p.label}) from {attention_label(p.base_attention)} "
                f"to {attention_label(insights.period_effective[p.index])} "
                f"({observed[1]} observations of concentration and engagement).")))

    # AI advice that was actually honoured
    for a in advice or []:
        if a.honoured:
            out.append(Explanation(kind="ai_advice", text=(
                f"AI suggestion for {a.subject}: {BUCKET_LABELS[a.preferred_time]} - {a.reason} "
                f"(followed in {a.honoured}).")))

    locked = [c for c in cells if c.locked]
    if locked:
        out.append(Explanation(kind="constraint", text=(
            "Locked periods kept exactly as you set them: " + ", ".join(f"{ref(c)} {c.subject}" for c in locked) + "."),
            cells=[ref(c) for c in locked]))
    fixed = [c for c in cells if c.kind == "fixed"]
    if fixed:
        out.append(Explanation(kind="constraint", text=(
            "Fixed activities respected: " + ", ".join(f"{ref(c)} {c.subject}" for c in fixed) + "."),
            cells=[ref(c) for c in fixed]))
    free = [c for c in cells if c.subject == FREE_LABEL]
    if free:
        out.append(Explanation(kind="constraint", text=(
            f"{len(free)} period(s) are not needed by any subject's weekly count and are set as "
            f"'{FREE_LABEL}' ({', '.join(ref(c) for c in free[:8])}{'...' if len(free) > 8 else ''}). "
            "Edit them or raise a subject's weekly periods."), cells=[ref(c) for c in free]))
    return out


def advice_honoured(advice: list[AdviceItem], cells: list[Cell], periods: list[PeriodTime]) -> list[AdviceItem]:
    ptime = {p.index: p for p in periods}
    out = []
    for a in advice:
        placed = [c for c in cells if c.subject == a.subject]
        if not placed or a.preferred_time == "any":
            a.honoured = ""
        else:
            n = sum(1 for c in placed if ptime.get(c.period) and ptime[c.period].bucket == a.preferred_time)
            a.honoured = f"{n} of {len(placed)} periods" if n else ""
        out.append(a)
    return out


def diff(old: list[Cell] | None, new: list[Cell], insights: Insights,
         periods: list[PeriodTime]) -> list[Change]:
    if not old:
        return []
    ptime = {p.index: p for p in periods}
    before = {(c.day, c.period): c for c in old}
    changes = []
    for c in new:
        prev = before.get((c.day, c.period))
        if prev and prev.subject == c.subject:
            continue
        bucket = ptime[c.period].bucket if c.period in ptime else "any"
        reason = f"{c.subject}"
        if c.load:
            reason += f" ({LOAD_LABELS[c.load]} load)"
        reason += f" fits this {attention_label(c.attention)} attention period"
        aff = insights.affinity_score(c.subject, bucket)
        if aff > 0:
            reason += f"; feedback shows good understanding {BUCKET_LABELS[bucket]}"
        if prev and prev.subject in {k[0] for k in insights.affinity} and insights.affinity_score(prev.subject, bucket) < 0:
            reason += f"; feedback shows {prev.subject} struggles {BUCKET_LABELS[bucket]}"
        if prev and prev.attention != c.attention:
            reason += f" (attention estimate changed {prev.attention:.1f} -> {c.attention:.1f})"
        changes.append(Change(slot=f"{c.day} P{c.period}", before=prev.subject if prev else "-",
                              after=c.subject, reason=reason + "."))
    return changes


def recommendations(profile: ClassProfile, insights: Insights, periods: list[PeriodTime]) -> list[str]:
    """Suggestions for the CURRENT timetable based on feedback. Never applied automatically."""
    if not profile.current:
        return []
    ptime = {p.index: p for p in periods}
    recs = []
    for (subject, bucket), (avg, n) in insights.affinity.items():
        if n < MIN_OBS or avg > 2.5:
            continue
        there = [c for c in profile.current.cells if c.subject == subject
                 and ptime.get(c.period) and ptime[c.period].bucket == bucket]
        if there:
            recs.append(f"{subject} is still scheduled {BUCKET_LABELS[bucket]} in "
                        f"{', '.join(ref(c) for c in there)}, where understanding averaged {avg:.1f}/5. "
                        "Generate an updated proposal to move it to a more attentive period.")
    for p in periods:
        obs = insights.period_observed.get(p.index)
        if not obs or obs[1] < MIN_OBS or obs[0] > 2:
            continue
        heavy = [c for c in profile.current.cells if c.period == p.index and c.load == "high"]
        if heavy:
            recs.append(f"Period {p.index} ({p.label}) had low observed focus ({obs[0]:.1f}/5) but holds "
                        f"high-load subjects ({', '.join(ref(c) + ' ' + c.subject for c in heavy[:4])}).")
    return recs
