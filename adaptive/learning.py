"""Learning from teacher feedback.

Feedback is treated as evidence that refines (not replaces) the teacher's attention pattern:
* period attention = 60 % teacher-entered level + 40 % observed focus, once a period has
  at least MIN_OBS feedback entries;
* subject x time-of-day affinity = average 'understanding' for that subject in that part of
  the day, which the scheduler uses as a bonus / penalty.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from adaptive.models import BUCKET_LABELS, ClassProfile, attention_label
from adaptive.timing import PeriodTime

MIN_OBS = 2
TEACHER_WEIGHT = 0.6


@dataclass
class Insights:
    period_observed: dict[int, tuple[float, int]] = field(default_factory=dict)
    period_effective: dict[int, float] = field(default_factory=dict)
    affinity: dict[tuple[str, str], tuple[float, int]] = field(default_factory=dict)
    took_longer: dict[str, int] = field(default_factory=dict)
    successes: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    n_entries: int = 0

    def affinity_score(self, subject: str, bucket: str) -> float:
        """(avg understanding - 3) when there is enough evidence, else 0."""
        avg, n = self.affinity.get((subject, bucket), (3.0, 0))
        return (avg - 3.0) if n >= MIN_OBS else 0.0


def analyse(profile: ClassProfile, periods: list[PeriodTime]) -> Insights:
    ins = Insights(n_entries=len(profile.feedback))
    by_index = {p.index: p for p in periods}
    focus: dict[int, list[float]] = defaultdict(list)
    understanding: dict[tuple[str, str], list[int]] = defaultdict(list)
    longer: dict[str, int] = defaultdict(int)

    for fb in profile.feedback:
        p = by_index.get(fb.period)
        if not p:
            continue
        focus[fb.period].append((fb.concentration + fb.engagement) / 2)
        understanding[(fb.subject, p.bucket)].append(fb.understanding)
        if fb.took_longer:
            longer[fb.subject] += 1
        if fb.successful_activity.strip():
            ins.successes.append(f"{fb.subject} ({fb.day} P{fb.period}): {fb.successful_activity.strip()}")

    for p in periods:
        vals = focus.get(p.index, [])
        if vals:
            ins.period_observed[p.index] = (round(sum(vals) / len(vals), 2), len(vals))
        if len(vals) >= MIN_OBS:
            obs = sum(vals) / len(vals)
            eff = round(TEACHER_WEIGHT * p.base_attention + (1 - TEACHER_WEIGHT) * obs, 2)
            ins.period_effective[p.index] = eff
            if abs(obs - p.base_attention) >= 1:
                ins.messages.append(
                    f"Period {p.index} ({p.label}): you set {attention_label(p.base_attention)}, but "
                    f"{len(vals)} feedback entries average {obs:.1f}/5 ({attention_label(obs)}). "
                    f"The assistant now treats it as {eff:.1f} ({attention_label(eff)}).")
        else:
            ins.period_effective[p.index] = p.base_attention

    for key, vals in understanding.items():
        avg = round(sum(vals) / len(vals), 2)
        ins.affinity[key] = (avg, len(vals))
        subject, bucket = key
        if len(vals) >= MIN_OBS and avg <= 2.5:
            ins.messages.append(
                f"{subject} {BUCKET_LABELS[bucket]}: understanding averaged {avg:.1f}/5 over "
                f"{len(vals)} lessons, so the assistant will try to avoid that time for {subject}.")
        elif len(vals) >= MIN_OBS and avg >= 4:
            ins.messages.append(
                f"{subject} {BUCKET_LABELS[bucket]}: understanding averaged {avg:.1f}/5 over "
                f"{len(vals)} lessons, so the assistant will favour that time for {subject}.")

    ins.took_longer = dict(longer)
    for subject, n in longer.items():
        if n >= MIN_OBS:
            ins.messages.append(
                f"{subject} took longer than expected {n} times. Consider adding a period per "
                f"week or reducing content per lesson (the assistant will not change frequencies "
                f"without you).")
    return ins
