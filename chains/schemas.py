"""Pydantic models shared by RAG, the LangChain chains, the validator, the UI and the PDF.

Models marked "LLM output" are passed to ``llm.with_structured_output(...)``; keep their
field descriptions clear because the model reads them as instructions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------------------


ACCESSIBILITY_OPTIONS: list[tuple[str, str, str]] = [
    ("large_font", "Larger font size", "Body text about 15 pt instead of 11 pt."),
    ("extra_spacing", "Increased line and paragraph spacing", "More space between lines and blocks."),
    ("simple_instructions", "Simple and clear instructions", "Instructions broken into small numbered steps."),
    ("short_sentences", "Shorter sentences and simpler vocabulary", "Same maths, easier words to read."),
    ("one_question_per_section", "One question per section", "Each question in its own clearly separated block."),
    ("reduced_clutter", "Reduced visual clutter", "No boxes, colours or decoration except what helps."),
    ("more_answer_space", "More space for writing answers", "Extra, widely spaced answer lines."),
    ("visual_hints", "Visual hints and illustrations", "A short picture or drawing hint for each question."),
    ("highlight_keywords", "Highlight important keywords", "Key words in bold with a soft highlight."),
    ("oral_alternatives", "Oral-response alternatives", "Every question can also be answered out loud."),
    ("reduce_questions", "Reduce number of questions when appropriate",
     "Fewer questions, but every learning objective is still covered."),
]
# The recommended preset leaves question count unchanged so expectations are not lowered.
DYSLEXIA_PRESET = {k for k, _, _ in ACCESSIBILITY_OPTIONS} - {"reduce_questions"}


class AccessibilitySettings(BaseModel):
    """Teacher-selected worksheet adaptations (not LLM output)."""

    preset: Literal["off", "dyslexia", "custom"] = "off"
    large_font: bool = False
    extra_spacing: bool = False
    simple_instructions: bool = False
    short_sentences: bool = False
    one_question_per_section: bool = False
    reduced_clutter: bool = False
    more_answer_space: bool = False
    visual_hints: bool = False
    highlight_keywords: bool = False
    oral_alternatives: bool = False
    reduce_questions: bool = False

    @property
    def enabled(self) -> list[str]:
        return [k for k, _, _ in ACCESSIBILITY_OPTIONS if getattr(self, k)]

    @property
    def any_enabled(self) -> bool:
        return bool(self.enabled)

    @property
    def enabled_labels(self) -> list[str]:
        return [label for k, label, _ in ACCESSIBILITY_OPTIONS if getattr(self, k)]

    @classmethod
    def dyslexia_preset(cls) -> "AccessibilitySettings":
        return cls(preset="dyslexia", **{k: True for k in DYSLEXIA_PRESET})


class LessonRequest(BaseModel):
    topic: str
    subject: str
    grades: list[int]
    duration: int  # minutes
    language: str = "English"
    accessibility: AccessibilitySettings = Field(default_factory=AccessibilitySettings)
    # Optional hint from the Adaptive Classroom Assistant ("Very High" ... "Very Low")
    attention_hint: str = ""


# --------------------------------------------------------------------------------------
# RAG results (not LLM output)
# --------------------------------------------------------------------------------------


class SourceSnippet(BaseModel):
    label: str  # e.g. "S1" - used for citations in prompts and UI
    chunk_id: str
    source: str  # file name
    grade: int = 0  # 0 = not grade-specific
    subject: str = "General"
    page: int = 0  # 1-based page for PDFs, 0 for text files
    score: float = 0.0  # cosine relevance 0..1
    text: str
    is_demo: bool = False

    @property
    def grade_label(self) -> str:
        return f"Grade {self.grade}" if self.grade else "All grades"


class RetrievalResult(BaseModel):
    query: str
    sources: list[SourceSnippet] = Field(default_factory=list)
    by_grade: dict[int, list[str]] = Field(default_factory=dict)  # grade -> source labels
    store_empty: bool = False
    error: str | None = None

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)

    def labels_for_grade(self, grade: int) -> list[str]:
        return list(self.by_grade.get(grade, []))

    def snippets_for_grade(self, grade: int) -> list[SourceSnippet]:
        wanted = set(self.labels_for_grade(grade))
        return [s for s in self.sources if s.label in wanted]

    def grades_without_sources(self) -> list[int]:
        return [g for g, labels in sorted(self.by_grade.items()) if not labels]

    @staticmethod
    def _format(snippets: list[SourceSnippet], max_chars: int) -> str:
        parts = []
        for s in snippets:
            text = " ".join(s.text.split())
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
            demo = " DEMO MATERIAL" if s.is_demo else ""
            page = f", page {s.page}" if s.page else ""
            parts.append(f"[{s.label}] ({s.grade_label}, {s.source}{page}{demo})\n{text}")
        return "\n\n".join(parts)

    def context_for_grade(self, grade: int, max_chars: int = 700) -> str:
        snippets = self.snippets_for_grade(grade)
        if not snippets:
            return (
                f"NO CURRICULUM SOURCE WAS RETRIEVED FOR GRADE {grade}. "
                "Use general, age-appropriate practice and do not cite any source."
            )
        return self._format(snippets, max_chars)

    def context_for_all(self, max_chars: int = 500) -> str:
        if not self.sources:
            return (
                "NO CURRICULUM SOURCE WAS RETRIEVED. Use general, age-appropriate "
                "practice and do not cite any source."
            )
        return self._format(self.sources, max_chars)


# --------------------------------------------------------------------------------------
# LLM output: lesson intake / objectives
# --------------------------------------------------------------------------------------


class GradeObjectives(BaseModel):
    """Learning objectives for one grade."""

    grade: int = Field(description="Grade number, 1-5")
    objectives: list[str] = Field(
        description="2-3 measurable learning objectives that start with a verb"
    )
    success_criteria: str = Field(
        description="One sentence: what the teacher will see/hear when a child has learned it"
    )
    key_vocabulary: list[str] = Field(description="3-5 key words for this grade")
    curriculum_alignment: str = Field(
        description=(
            "How the objectives follow the retrieved curriculum, citing source labels such as "
            "[S1]. If no source was retrieved for this grade, say so plainly."
        )
    )


class LessonPlanOutline(BaseModel):
    """Shared multi-grade lesson outline."""

    lesson_title: str = Field(description="Short title for the shared lesson")
    big_idea: str = Field(description="One sentence big idea that links all grades")
    whole_class_opening: str = Field(
        description="A 3-5 minute whole-class warm-up all grades do together"
    )
    whole_class_closing: str = Field(
        description="A 3-5 minute whole-class closing / sharing activity"
    )
    grades: list[GradeObjectives] = Field(description="One entry per selected grade")

    def objectives_for(self, grade: int) -> GradeObjectives | None:
        return next((g for g in self.grades if g.grade == grade), None)


# --------------------------------------------------------------------------------------
# LLM output: grade content
# --------------------------------------------------------------------------------------


class WorksheetTask(BaseModel):
    task: str = Field(description="One task the child does alone on slate or notebook")
    answer: str = Field(description="Correct answer or what a good answer looks like")


class QuizQuestion(BaseModel):
    question: str = Field(description="Short quiz question")
    options: list[str] = Field(
        default_factory=list,
        description="2-4 answer choices for multiple choice, or [] for open/oral questions",
    )
    answer: str = Field(description="Correct answer")


class GradeContent(BaseModel):
    """Activity, worksheet and quiz for one grade."""

    grade: int = Field(description="Grade number, 1-5")
    activity_title: str = Field(description="Name of the teacher-led activity")
    activity_type: str = Field(description="hands-on, blackboard, or oral game")
    materials: list[str] = Field(
        description="Free or low-cost local materials (stones, sticks, leaves, paper, chalk)"
    )
    activity_steps: list[str] = Field(description="4-6 short, numbered-in-order steps")
    worksheet_title: str = Field(description="Title of the independent worksheet")
    worksheet_instructions: str = Field(
        description="One simple instruction line the child can follow without the teacher"
    )
    worksheet_tasks: list[WorksheetTask] = Field(description="3-4 independent tasks")
    quiz: list[QuizQuestion] = Field(description="Exactly 3 quiz questions with answers")
    teacher_tip: str = Field(
        description="One practical tip for this grade (misconception, peer help, support)"
    )
    source_labels: list[str] = Field(
        default_factory=list,
        description="Curriculum source labels used, e.g. ['S1','S3']; [] if none were given",
    )


# --------------------------------------------------------------------------------------
# Timetable
# --------------------------------------------------------------------------------------

Mode = Literal["with_teacher", "independent", "whole_class"]


class GradeAssignment(BaseModel):
    grade: int
    mode: Mode
    task: str


class TimeBlock(BaseModel):
    start_min: int
    end_min: int
    teacher_focus: list[int] = Field(default_factory=list)  # [] = whole class
    title: str = ""
    assignments: list[GradeAssignment] = Field(default_factory=list)

    @property
    def minutes(self) -> int:
        return self.end_min - self.start_min

    @property
    def time_label(self) -> str:
        return f"{self.start_min}-{self.end_min} min"

    @property
    def teacher_label(self) -> str:
        if not self.teacher_focus:
            return "Whole class"
        return "Grade " + " & ".join(str(g) for g in self.teacher_focus)

    def assignment_for(self, grade: int) -> GradeAssignment | None:
        return next((a for a in self.assignments if a.grade == grade), None)


class Timetable(BaseModel):
    blocks: list[TimeBlock] = Field(default_factory=list)
    teacher_tips: list[str] = Field(default_factory=list)


class BlockTaskText(BaseModel):
    grade: int = Field(description="Grade number")
    task: str = Field(description="One concrete sentence: what this grade does in this block")


class BlockText(BaseModel):
    block_index: int = Field(description="Index of the block, starting at 0")
    tasks: list[BlockTaskText] = Field(description="One entry per grade in this block")


class TimetableText(BaseModel):
    """LLM output: task wording for a fixed rotation skeleton."""

    blocks: list[BlockText] = Field(description="One entry per block of the skeleton")
    teacher_tips: list[str] = Field(
        description="2-3 tips for managing the multi-grade rotation (signals, peer helpers)"
    )


class ValidationReport(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Full result (saved as JSON, rendered in UI and PDF)
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# LLM output: accessible (dyslexia-friendly) worksheet
# --------------------------------------------------------------------------------------


class WorkedExample(BaseModel):
    problem: str = Field(description="A solved example similar to the harder questions")
    steps: list[str] = Field(description="2-4 very short steps showing how to solve it")
    answer: str = Field(description="The answer to the example")


class AccessibleQuestion(BaseModel):
    heading: str = Field(description="Short heading, e.g. 'Question 1: Find one half'")
    steps: list[str] = Field(description="The instruction broken into 1-3 short steps, one action each")
    question: str = Field(description="The question itself, short and clear")
    keywords: list[str] = Field(
        description="1-3 important words copied exactly from the steps or question, to highlight"
    )
    visual_hint: str = Field(
        description="A simple drawing or object hint the child can use, or '' if not helpful"
    )
    other_ways: str = Field(
        description="Another way to show understanding: draw it, use objects, or tell the teacher"
    )
    answer: str = Field(description="Correct answer, for the teacher's answer key")


class AccessibleWorksheet(BaseModel):
    """Dyslexia-friendly version of a grade's worksheet."""

    grade: int = Field(description="Grade number, 1-5")
    title: str = Field(description="Short worksheet title")
    goal: str = Field(description="One short 'I can ...' sentence stating the learning goal")
    how_to_use: list[str] = Field(description="2-3 very short tips for working through the sheet")
    example: WorkedExample = Field(description="A worked example shown before the questions")
    questions: list[AccessibleQuestion] = Field(description="The worksheet questions")
    teacher_note: str = Field(
        description="One sentence for the teacher: how this sheet keeps the same learning objectives"
    )


class LessonResult(BaseModel):
    lesson_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    request: LessonRequest
    retrieval: RetrievalResult
    outline: LessonPlanOutline
    contents: list[GradeContent]
    timetable: Timetable
    validation: ValidationReport
    warnings: list[str] = Field(default_factory=list)
    model: str = ""
    elapsed_s: float = 0.0
    accessible: list[AccessibleWorksheet] = Field(default_factory=list)
    edited_at: datetime | None = None

    def content_for(self, grade: int) -> GradeContent | None:
        return next((c for c in self.contents if c.grade == grade), None)

    def accessible_for(self, grade: int) -> AccessibleWorksheet | None:
        return next((a for a in self.accessible if a.grade == grade), None)
