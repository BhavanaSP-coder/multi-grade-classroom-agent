"""Groq assistance for the adaptive timetable (advice + plain-language summary).

The AI never decides validity: advice becomes small soft bonuses in the Python optimiser,
and every AI summary sentence is checked against the real timetable. Sentences that cite
a slot ("Mon P1") whose subject is not named in that sentence are dropped.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from adaptive.learning import Insights
from adaptive.models import AdviceItem, Cell, ClassProfile, Explanation, attention_label
from adaptive.timing import PeriodTime
from chains.llm import get_llm, structured

log = logging.getLogger(__name__)


class SubjectPreference(BaseModel):
    subject: str = Field(description="Exact subject name from the list")
    preferred_time: Literal["early_morning", "late_morning", "afternoon", "any"] = Field(
        description="Best part of the day for this subject for THIS class")
    reason: str = Field(description="One sentence citing the teacher's attention data or feedback")


class AdaptiveAdvice(BaseModel):
    preferences: list[SubjectPreference] = Field(description="One entry per subject")
    notes: list[str] = Field(description="0-3 short, practical scheduling notes for the teacher")


class TimetableSummary(BaseModel):
    bullets: list[str] = Field(description="4-6 short sentences explaining WHY, not listing periods")


ADVICE_SYSTEM = """You are an experienced school timetable planner and learning-science adviser.
Recommend the best part of the day for each subject for ONE specific class, using ONLY the teacher's data below.
- The attention levels are the teacher's own observations, not measurements; refer to them as observations.
- Demanding subjects suit periods the teacher observed as attentive; activity subjects suit lower-attention periods.
- Use feedback evidence when present (it outweighs general rules).
- Consider grade level: younger children usually need demanding work earlier and shorter.
- Use the exact subject names given. Do not invent data."""

ADVICE_HUMAN = """Class: Grade {grade}, section {section}, {students} students.
Parts of the day: early_morning = before the first break, late_morning = before lunch, afternoon = after lunch.

Periods (time, part of day, attention observed by the teacher, adjusted by feedback):
{periods}

Subjects (name, cognitive load set by teacher, periods per week):
{subjects}

Feedback insights:
{insights}

Successful activities reported:
{successes}"""

SUMMARY_SYSTEM = """You explain WHY a school timetable was arranged this way, for a busy teacher.
Write 4-6 short sentences (max 30 words each). Explain the reasoning: attention observations, cognitive load, breaks between demanding subjects, activity subjects in low-focus periods, feedback and constraints.
Do NOT list the timetable period by period. Cite at most two example periods per sentence.
Use ONLY the facts provided; do not add placements, reasons or data that are not in the facts.
When you mention a specific period, write it exactly as 'Mon P1' (day + P + number) and name the subject in that period in the same sentence.
Treat attention levels as the teacher's observations, never as measurements."""

SUMMARY_HUMAN = """Timetable (day period: subject [load, attention]):
{grid}

Verified facts:
{facts}"""

_SLOT = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat)\s*P(\d{1,2})\b")


def _period_lines(periods: list[PeriodTime], insights: Insights) -> str:
    return "\n".join(
        f"- P{p.index} {p.label} ({p.bucket}): teacher {attention_label(p.base_attention)}"
        f", effective {insights.period_effective.get(p.index, p.base_attention):.1f}/5"
        for p in periods)


def get_advice(profile: ClassProfile, periods: list[PeriodTime],
               insights: Insights) -> tuple[list[AdviceItem], list[str]]:
    chain = ChatPromptTemplate.from_messages([("system", ADVICE_SYSTEM), ("human", ADVICE_HUMAN)]) \
        | structured(AdaptiveAdvice)
    out: AdaptiveAdvice | None = chain.invoke({
        "grade": profile.grade, "section": profile.section, "students": profile.num_students,
        "periods": _period_lines(periods, insights),
        "subjects": "\n".join(f"- {s.name}: {s.load}, {s.periods_per_week}/week"
                              for s in profile.subjects if s.periods_per_week > 0),
        "insights": "\n".join(f"- {m}" for m in insights.messages) or "- No feedback yet.",
        "successes": "\n".join(f"- {s}" for s in insights.successes[-8:]) or "- None recorded.",
    })
    if out is None:
        return [], ["The AI did not return advice; the optimiser used your data only."]
    names = {s.name.lower(): s.name for s in profile.subjects}
    items, seen = [], set()
    for pref in out.preferences:
        name = names.get(pref.subject.strip().lower())
        if not name or name in seen:
            continue
        seen.add(name)
        items.append(AdviceItem(subject=name, preferred_time=pref.preferred_time, reason=pref.reason.strip()))
    return items, ["AI tip (general advice; the grid shows what was actually scheduled): " + n.strip()
                   for n in out.notes if n and n.strip()][:3]


def verify_bullets(bullets: list[str], cells: list[Cell], subjects: list[str]) -> tuple[list[str], int]:
    """Keep only sentences consistent with the timetable."""
    by_ref = {f"{c.day} P{c.period}": c.subject for c in cells}
    kept, dropped = [], 0
    for b in bullets:
        text = b.strip()
        if not text:
            continue
        mentioned, rest = set(), text.lower()
        for s in sorted(subjects, key=len, reverse=True):  # "Computer Science" before "Science"
            if s.lower() in rest:
                mentioned.add(s)
                rest = rest.replace(s.lower(), " ")
        ok = True
        for day, p in _SLOT.findall(text):
            actual = by_ref.get(f"{day} P{int(p)}")
            if actual is None or (mentioned and actual not in mentioned):
                ok = False
                break
        if ok:
            kept.append(text)
        else:
            dropped += 1
    return kept, dropped


def summarize(profile: ClassProfile, cells: list[Cell],
              explanations: list[Explanation]) -> tuple[list[str], int]:
    chain = ChatPromptTemplate.from_messages([("system", SUMMARY_SYSTEM), ("human", SUMMARY_HUMAN)]) \
        | structured(TimetableSummary)
    grid = "\n".join(f"{c.day} P{c.period}: {c.subject} [{c.load or c.kind}, {attention_label(c.attention)}]"
                     for c in cells)
    out = chain.invoke({"grid": grid, "facts": "\n".join(f"- {e.text}" for e in explanations)})
    if out is None:
        return [], 0
    names = [s.name for s in profile.subjects] + [c.subject for c in cells if c.kind != "subject"]
    return verify_bullets(out.bullets[:7], cells, sorted(set(names), key=len, reverse=True))
