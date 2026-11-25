"""Data models for the AI Adaptive Classroom Assistant (subject timetable for one class).

Attention levels are TEACHER OBSERVATIONS, not measurements: the app never claims to
measure students' cognitive capacity. Feedback entries refine those observations.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Load = Literal["high", "medium", "low"]
TimeBucket = Literal["early_morning", "late_morning", "afternoon", "any"]

ALL_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DAY_NAMES = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
             "Fri": "Friday", "Sat": "Saturday"}
ATTENTION_LABELS = {5: "Very High", 4: "High", 3: "Moderate", 2: "Low", 1: "Very Low"}
LOAD_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}
BUCKET_LABELS = {"early_morning": "early morning (before the first break)",
                 "late_morning": "late morning (before lunch)",
                 "afternoon": "after lunch", "any": "any time"}

# Sensible starting classifications; teachers can change every one of them.
DEFAULT_LOADS: dict[str, Load] = {
    "Mathematics": "high", "Science": "high", "Computer Science": "medium",
    "English": "medium", "Kannada": "medium", "Hindi": "medium",
    "Social Studies": "medium", "Environmental Studies": "medium",
    "Art and Craft": "low", "Physical Education": "low", "Music": "low",
}
SUBJECT_CHOICES = list(DEFAULT_LOADS)

# Map assistant subject names to the lesson planner's subject list.
LESSON_SUBJECT_MAP = {"Mathematics": "Maths", "Environmental Studies": "EVS"}


def attention_label(level: float) -> str:
    return ATTENTION_LABELS[min(5, max(1, int(round(level))))]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class BreakConfig(BaseModel):
    name: str = "Break"
    after_period: int  # break happens after this period number (1-based)
    minutes: int


class SubjectConfig(BaseModel):
    name: str
    load: Load = "medium"
    periods_per_week: int = 5
    max_per_day: int = 1
    teacher: str = ""


class AttentionWindow(BaseModel):
    start: str  # "HH:MM"
    end: str
    level: int = 3  # 1..5


class TeacherUnavailability(BaseModel):
    teacher: str
    day: str
    periods: list[int]


class FixedActivity(BaseModel):
    day: str
    period: int
    label: str


class FeedbackEntry(BaseModel):
    id: str = Field(default_factory=new_id)
    date: str = ""  # YYYY-MM-DD
    day: str
    period: int
    subject: str
    understanding: int = 3  # 1..5
    engagement: int = 3
    concentration: int = 3
    took_longer: bool = False
    successful_activity: str = ""
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class Cell(BaseModel):
    day: str
    period: int
    subject: str  # subject name, fixed-activity label, or FREE_LABEL
    kind: Literal["subject", "fixed", "free"] = "subject"
    load: Load | None = None
    attention: float = 3.0  # effective attention used when scheduling
    locked: bool = False


class Explanation(BaseModel):
    kind: Literal["placement", "constraint", "feedback", "ai_advice", "balance", "warning"]
    text: str
    cells: list[str] = Field(default_factory=list)  # "Mon P1" references backing the claim


class Change(BaseModel):
    slot: str  # "Mon P1"
    before: str
    after: str
    reason: str


class AdviceItem(BaseModel):
    subject: str
    preferred_time: TimeBucket
    reason: str
    honoured: str = ""  # filled in after scheduling, e.g. "4 of 5 periods"


class AdaptiveValidation(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score: float = 0.0
    metrics: dict[str, float | int | str] = Field(default_factory=dict)


class TimetableVersion(BaseModel):
    id: str = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=datetime.now)
    source: Literal["generated", "edited"] = "generated"
    status: Literal["proposed", "approved", "rejected"] = "proposed"
    cells: list[Cell] = Field(default_factory=list)
    score: float = 0.0
    alternatives_scores: list[float] = Field(default_factory=list)
    validation: AdaptiveValidation = Field(default_factory=lambda: AdaptiveValidation(ok=False))
    explanations: list[Explanation] = Field(default_factory=list)
    ai_summary: list[str] = Field(default_factory=list)
    ai_dropped: int = 0  # AI sentences removed because they contradicted the timetable
    advice: list[AdviceItem] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def cell(self, day: str, period: int) -> Cell | None:
        return next((c for c in self.cells if c.day == day and c.period == period), None)


class ClassProfile(BaseModel):
    id: str = Field(default_factory=new_id)
    grade: int = 4
    section: str = "A"
    num_students: int = 30
    start_time: str = "09:00"
    end_time: str = "15:30"
    working_days: list[str] = Field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri"])
    num_periods: int = 7
    period_minutes: int = 40
    breaks: list[BreakConfig] = Field(default_factory=list)
    subjects: list[SubjectConfig] = Field(default_factory=list)
    attention: list[AttentionWindow] = Field(default_factory=list)
    unavailability: list[TeacherUnavailability] = Field(default_factory=list)
    fixed: list[FixedActivity] = Field(default_factory=list)
    feedback: list[FeedbackEntry] = Field(default_factory=list)
    current: TimetableVersion | None = None
    proposal: TimetableVersion | None = None
    history: list[TimetableVersion] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def name(self) -> str:
        return f"Grade {self.grade} - {self.section}"

    def subject(self, name: str) -> SubjectConfig | None:
        return next((s for s in self.subjects if s.name == name), None)


FREE_LABEL = "Revision / Reading time"
