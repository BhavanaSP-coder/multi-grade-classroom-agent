"""Printable PDF packet (ReportLab): cover + sources, timetable, student worksheets, answer keys.

Fonts / languages
-----------------
* English uses the built-in Helvetica.
* Kannada / Hindi need a Unicode TrueType font. The generator looks, in order, for:
    fonts/NotoSansKannada-Regular.ttf (+ -Bold.ttf)      for Kannada
    fonts/NotoSansDevanagari-Regular.ttf (+ -Bold.ttf)   for Hindi
    Windows system fonts: Nirmala.ttc / Nirmala.ttf (Kannada + Devanagari), Tunga, Mangal
    Common Linux / macOS Noto locations
  Download the Noto fonts free from https://fonts.google.com/noto and put the .ttf files in fonts/.
* Limitation: ReportLab does not fully "shape" Indic scripts, so some conjunct letters
  (ಕ್ಷ, क्ष, matras) may look split. The packet notes this; for perfect script rendering
  use the "Print / Save as PDF" button on the results page (the browser shapes text correctly).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    HRFlowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

import config
from chains.schemas import LessonResult

log = logging.getLogger(__name__)

# LLMs often emit typographic characters that common PDF fonts lack (they print as boxes).
_PDF_CHAR_MAP = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "⁄": "/",
})


def clean_text(text) -> str:
    return str(text or "").translate(_PDF_CHAR_MAP)


def escape(text) -> str:
    """XML-escape for ReportLab paragraphs after normalising unsupported typography."""
    return _xml_escape(clean_text(text))

WIN_FONTS = Path("C:/Windows/Fonts")
FONT_CANDIDATES: dict[str, list[tuple[Path, Path | None]]] = {
    "Kannada": [
        (config.FONTS_DIR / "NotoSansKannada-Regular.ttf", config.FONTS_DIR / "NotoSansKannada-Bold.ttf"),
        (WIN_FONTS / "Nirmala.ttc", None),
        (WIN_FONTS / "Nirmala.ttf", WIN_FONTS / "NirmalaB.ttf"),
        (WIN_FONTS / "tunga.ttf", WIN_FONTS / "tungab.ttf"),
        (Path("/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf"),
         Path("/usr/share/fonts/truetype/noto/NotoSansKannada-Bold.ttf")),
    ],
    "Hindi": [
        (config.FONTS_DIR / "NotoSansDevanagari-Regular.ttf", config.FONTS_DIR / "NotoSansDevanagari-Bold.ttf"),
        (WIN_FONTS / "Nirmala.ttc", None),
        (WIN_FONTS / "Nirmala.ttf", WIN_FONTS / "NirmalaB.ttf"),
        (WIN_FONTS / "mangal.ttf", WIN_FONTS / "mangalb.ttf"),
        (Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
         Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf")),
    ],
}

MODE_COLORS = {
    "with_teacher": colors.HexColor("#DCFCE7"),
    "independent": colors.HexColor("#EFF6FF"),
    "whole_class": colors.HexColor("#FEF9C3"),
}
MODE_TEXT = {"with_teacher": "WITH TEACHER", "independent": "ON OWN", "whole_class": "ALL TOGETHER"}


def _load_ttf(name: str, path: Path) -> bool:
    try:
        if path.suffix.lower() == ".ttc":
            pdfmetrics.registerFont(TTFont(name, str(path), subfontIndex=0))
        else:
            pdfmetrics.registerFont(TTFont(name, str(path)))
        return True
    except Exception as exc:  # bad / CFF-based font
        log.warning("Could not load font %s: %s", path, exc)
        return False


@lru_cache(maxsize=None)
def resolve_fonts(language: str) -> tuple[str, str, str | None]:
    """Return (regular_font, bold_font, warning_or_None) for the language."""
    if language not in FONT_CANDIDATES:
        return "Helvetica", "Helvetica-Bold", None
    reg_name, bold_name = f"Indic-{language}", f"Indic-{language}-Bold"
    for regular, bold in FONT_CANDIDATES[language]:
        if regular.exists() and _load_ttf(reg_name, regular):
            if not (bold and bold.exists() and _load_ttf(bold_name, bold)):
                bold_name = reg_name
            pdfmetrics.registerFontFamily(reg_name, normal=reg_name, bold=bold_name,
                                          italic=reg_name, boldItalic=bold_name)
            log.info("Using %s for %s PDF", regular.name, language)
            return reg_name, bold_name, None
    return ("Helvetica", "Helvetica-Bold",
            f"No {language} font found: {language} text may print as empty boxes. "
            f"Add the Noto Sans font for {language} to the fonts/ folder (see README).")


def _p(text, style) -> Paragraph:
    safe = escape(str(text or "")).replace("\n", "<br/>")
    return Paragraph(safe, style)


def _styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName=bold, fontSize=20, leading=26),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=bold, fontSize=16, leading=22,
                             spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#1E3A8A")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=bold, fontSize=13, leading=18,
                             spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontName=regular, fontSize=10.5, leading=15),
        "small": ParagraphStyle("s", parent=base["BodyText"], fontName=regular, fontSize=8.5, leading=11),
        "cell": ParagraphStyle("c", parent=base["BodyText"], fontName=regular, fontSize=8.5, leading=11),
        "cellb": ParagraphStyle("cb", parent=base["BodyText"], fontName=bold, fontSize=8.5, leading=11),
        "task": ParagraphStyle("k", parent=base["BodyText"], fontName=regular, fontSize=12, leading=18),
        "center": ParagraphStyle("ctr", parent=base["BodyText"], fontName=regular, fontSize=9,
                                 alignment=TA_CENTER, textColor=colors.grey),
        "warn": ParagraphStyle("w", parent=base["BodyText"], fontName=regular, fontSize=9.5, leading=13,
                               textColor=colors.HexColor("#92400E"), backColor=colors.HexColor("#FEF3C7"),
                               borderPadding=5, spaceAfter=8),
    }
    return s


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    w, _ = canvas._pagesize
    canvas.drawString(1.5 * cm, 1 * cm, "Multi-Grade Classroom Agent - AI-generated; teacher to review before use")
    canvas.drawRightString(w - 1.5 * cm, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _grid(data, col_widths, header=True, extra=None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def build_lesson_pdf(result: LessonResult, out_path: Path | None = None,
                     only_accessible: bool = False) -> Path:
    """Full packet, or (only_accessible=True) just the dyslexia-friendly worksheets + key."""
    req = result.request
    suffix = "_accessible" if only_accessible else ""
    out_path = Path(out_path or config.PDF_DIR / f"lesson_{result.lesson_id}{suffix}.pdf")
    regular, bold, font_warning = resolve_fonts(req.language)
    st = _styles(regular, bold)

    doc = BaseDocTemplate(str(out_path), pagesize=A4, title=result.outline.lesson_title,
                          author="Multi-Grade Classroom Agent",
                          leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm,
                          bottomMargin=1.6 * cm)
    pw, ph = A4
    lw, lh = landscape(A4)
    m = 1.5 * cm
    doc.addPageTemplates([
        PageTemplate(id="portrait", pagesize=A4, onPage=_footer,
                     frames=[Frame(m, 1.6 * cm, pw - 2 * m, ph - 3.1 * cm, id="pf")]),
        PageTemplate(id="landscape", pagesize=landscape(A4), onPage=_footer,
                     frames=[Frame(m, 1.6 * cm, lw - 2 * m, lh - 3.1 * cm, id="lf")]),
    ])

    story: list = []
    grades = req.grades

    if only_accessible:
        if not result.accessible:
            raise ValueError("This lesson has no accessible worksheets yet. Enable the "
                             "Worksheet Accessibility Settings and generate them first.")
        for i, sheet in enumerate(result.accessible):
            if i:
                story.append(PageBreak())
            story += accessible_worksheet_story(sheet, req.accessibility, req.language)
        story += [PageBreak(), _p("Answer key: accessible worksheets", st["h1"])]
        story += _accessible_answer_tables(result, st)
        doc.build(story)
        return out_path

    # ---------------- Cover ----------------
    story.append(_p(result.outline.lesson_title, st["title"]))
    meta = (f"Topic: {req.topic}  |  Subject: {req.subject}  |  Grades: "
            f"{', '.join(map(str, grades))}  |  {req.duration} minutes  |  {req.language}")
    story += [_p(meta, st["center"]), Spacer(1, 10)]
    if font_warning:
        story.append(_p(font_warning, st["warn"]))
    if req.language in ("Kannada", "Hindi") and not font_warning:
        story.append(_p("Note: some Indic letter combinations may print imperfectly in this PDF. "
                        "For exact script rendering use the browser Print button on the results page.",
                        st["warn"]))
    story += [_p("Big idea", st["h2"]), _p(result.outline.big_idea, st["body"]),
              _p("Whole-class opening", st["h2"]), _p(result.outline.whole_class_opening, st["body"]),
              _p("Whole-class closing", st["h2"]), _p(result.outline.whole_class_closing, st["body"]),
              _p("Learning objectives", st["h2"])]
    rows = [[_p("Grade", st["cellb"]), _p("Objectives", st["cellb"]), _p("Curriculum alignment", st["cellb"])]]
    for g in grades:
        o = result.outline.objectives_for(g)
        rows.append([
            _p(f"Grade {g}", st["cellb"]),
            _p("\n".join(f"• {x}" for x in (o.objectives if o else [])), st["cell"]),
            _p(o.curriculum_alignment if o else "-", st["cell"]),
        ])
    story.append(_grid(rows, [2 * cm, 9 * cm, 7 * cm]))

    story.append(_p("Curriculum sources used (RAG)", st["h2"]))
    if result.retrieval.sources:
        rows = [[_p(h, st["cellb"]) for h in ("Label", "File", "Grade", "Relevance", "Note")]]
        for s in result.retrieval.sources:
            rows.append([_p(s.label, st["cell"]), _p(s.source + (f" p.{s.page}" if s.page else ""), st["cell"]),
                         _p(s.grade_label, st["cell"]), _p(f"{s.score:.2f}", st["cell"]),
                         _p("DEMO material - not official" if s.is_demo else "", st["cell"])])
        story.append(_grid(rows, [1.4 * cm, 7.6 * cm, 2.4 * cm, 2 * cm, 4.6 * cm]))
    else:
        story.append(_p("No relevant curriculum source was retrieved. Content is based on general "
                        "grade-level practice and should be checked against your syllabus.", st["warn"]))

    # ---------------- Timetable (landscape) ----------------
    story += [NextPageTemplate("landscape"), PageBreak(), _p("Teacher rotation timetable", st["h1"])]
    usable = lw - 2 * m
    fixed = 2.2 * cm + 2.6 * cm
    gcol = (usable - fixed) / max(1, len(grades))
    rows = [[_p("Time", st["cellb"]), _p("Teacher is with", st["cellb"])] +
            [_p(f"Grade {g}", st["cellb"]) for g in grades]]
    extra = []
    for r, b in enumerate(result.timetable.blocks, start=1):
        row = [_p(f"{b.time_label}\n({b.minutes} min)", st["cellb"]), _p(b.teacher_label, st["cellb"])]
        for c, g in enumerate(grades, start=2):
            a = b.assignment_for(g)
            if a:
                row.append(Paragraph(f"<b>{MODE_TEXT[a.mode]}</b><br/>{escape(a.task)}", st["cell"]))
                extra.append(("BACKGROUND", (c, r), (c, r), MODE_COLORS[a.mode]))
            else:
                row.append(_p("-", st["cell"]))
        rows.append(row)
    story.append(_grid(rows, [2.2 * cm, 2.6 * cm] + [gcol] * len(grades), extra=extra))
    if result.timetable.teacher_tips:
        story.append(_p("Tips for managing the rotation", st["h2"]))
        for tip in result.timetable.teacher_tips:
            story.append(_p(f"• {tip}", st["body"]))

    # ---------------- Student worksheets ----------------
    story.append(NextPageTemplate("portrait"))
    for g in grades:
        c = result.content_for(g)
        story.append(PageBreak())
        if not c:
            story.append(_p(f"Grade {g}: no content generated.", st["warn"]))
            continue
        story += [
            _p(f"Grade {g} - {c.worksheet_title}", st["h1"]),
            _p("Name: ______________________    Date: ____________", st["body"]),
            Spacer(1, 6),
            _p(c.worksheet_instructions, st["body"]),
            Spacer(1, 6),
        ]
        for i, t in enumerate(c.worksheet_tasks, start=1):
            story.append(KeepTogether([
                _p(f"{i}. {t.task}", st["task"]),
                _p("_" * 70, st["small"]), _p("_" * 70, st["small"]), Spacer(1, 8),
            ]))
        story.append(_p(f"Quiz - Grade {g}", st["h2"]))
        for i, q in enumerate(c.quiz, start=1):
            block = [_p(f"Q{i}. {q.question}", st["task"])]
            if q.options:
                block.append(_p("    ".join(f"({chr(97 + j)}) {o}" for j, o in enumerate(q.options)), st["body"]))
            else:
                block.append(_p("_" * 70, st["small"]))
            block.append(Spacer(1, 8))
            story.append(KeepTogether(block))
        sheet = result.accessible_for(g)
        if sheet:
            story.append(PageBreak())
            story += accessible_worksheet_story(sheet, req.accessibility, req.language)

    # ---------------- Teacher section: activities + answer keys ----------------
    story += [PageBreak(), _p("Teacher section: activities and answer keys", st["h1"])]
    for g in grades:
        c = result.content_for(g)
        if not c:
            continue
        part = [_p(f"Grade {g} - {c.activity_title} ({c.activity_type})", st["h2"]),
                _p("Materials: " + ", ".join(c.materials), st["body"])]
        part += [_p(f"{i}. {s}", st["body"]) for i, s in enumerate(c.activity_steps, start=1)]
        part.append(_p(f"Tip: {c.teacher_tip}", st["small"]))
        if c.source_labels:
            part.append(_p("Sources: " + ", ".join(c.source_labels), st["small"]))
        story.append(KeepTogether(part))
        rows = [[_p("#", st["cellb"]), _p("Worksheet task", st["cellb"]), _p("Answer", st["cellb"])]]
        rows += [[_p(str(i), st["cell"]), _p(t.task, st["cell"]), _p(t.answer, st["cell"])]
                 for i, t in enumerate(c.worksheet_tasks, start=1)]
        rows += [[_p(f"Q{i}", st["cell"]), _p(q.question, st["cell"]), _p(q.answer, st["cell"])]
                 for i, q in enumerate(c.quiz, start=1)]
        story += [Spacer(1, 4), _grid(rows, [1.2 * cm, 10.3 * cm, 6.5 * cm]), Spacer(1, 12)]
    if result.accessible:
        story += [_p("Answer key: accessible worksheets", st["h2"])]
        story += _accessible_answer_tables(result, st)

    doc.build(story)
    return out_path


def _accessible_answer_tables(result: LessonResult, st) -> list:
    out: list = []
    for sheet in result.accessible:
        rows = [[_p("#", st["cellb"]), _p(f"Grade {sheet.grade} - {sheet.title}", st["cellb"]),
                 _p("Answer", st["cellb"])]]
        rows.append([_p("Ex.", st["cell"]), _p(sheet.example.problem, st["cell"]),
                     _p(sheet.example.answer, st["cell"])])
        rows += [[_p(str(i), st["cell"]), _p(q.question, st["cell"]), _p(q.answer, st["cell"])]
                 for i, q in enumerate(sheet.questions, start=1)]
        out += [_grid(rows, [1.2 * cm, 10.3 * cm, 6.5 * cm]),
                _p(f"Teacher note: {sheet.teacher_note}", st["small"]), Spacer(1, 10)]
    return out


# --------------------------------------------------------------------------------------
# Accessible (dyslexia-friendly) worksheet layout
# --------------------------------------------------------------------------------------

ACCESS_FONT_FILES = [
    (WIN_FONTS / "verdana.ttf", WIN_FONTS / "verdanab.ttf"),
    (WIN_FONTS / "arial.ttf", WIN_FONTS / "arialbd.ttf"),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    (Path("/Library/Fonts/Verdana.ttf"), Path("/Library/Fonts/Verdana Bold.ttf")),
]
HIGHLIGHT = "#FFF1A8"


@lru_cache(maxsize=None)
def accessible_fonts(language: str) -> tuple[str, str]:
    """Readable sans-serif: Verdana > Arial > DejaVu Sans > Helvetica (English);
    the Indic font for Kannada / Hindi."""
    if language in FONT_CANDIDATES:
        regular, bold, _ = resolve_fonts(language)
        return regular, bold
    for regular, bold in ACCESS_FONT_FILES:
        if regular.exists() and _load_ttf("Access", regular):
            bold_name = "Access-Bold" if bold.exists() and _load_ttf("Access-Bold", bold) else "Access"
            # Needed so <b> inside paragraphs switches to the bold face.
            pdfmetrics.registerFontFamily("Access", normal="Access", bold=bold_name,
                                          italic="Access", boldItalic=bold_name)
            return "Access", bold_name
    return "Helvetica", "Helvetica-Bold"


def highlight_markup(text: str, keywords: list[str], enabled: bool) -> str:
    """Escape text for ReportLab and wrap keywords in bold + soft highlight."""
    safe = escape(str(text or ""))
    if not enabled or not keywords:
        return safe.replace("\n", "<br/>")
    import re as _re
    kws = sorted({escape(k) for k in keywords if k and k.strip()}, key=len, reverse=True)
    pattern = _re.compile("|".join(_re.escape(k) for k in kws), _re.IGNORECASE)
    marked = pattern.sub(lambda m: f'<font backColor="{HIGHLIGHT}"><b>{m.group(0)}</b></font>', safe)
    return marked.replace("\n", "<br/>")


def accessible_styles(settings, language: str) -> dict[str, ParagraphStyle]:
    regular, bold = accessible_fonts(language)
    size = 15 if settings.large_font else 12.5
    lead = 1.75 if settings.extra_spacing else 1.4
    gap = 14 if settings.extra_spacing else 8
    base = getSampleStyleSheet()["BodyText"]

    def mk(name, font, sz, after=gap, before=0, color=colors.black):
        return ParagraphStyle(name, parent=base, fontName=font, fontSize=sz, leading=sz * lead,
                              spaceAfter=after, spaceBefore=before, textColor=color, alignment=0)
    return {
        "title": mk("a_title", bold, size + 6, after=gap + 4),
        "h": mk("a_h", bold, size + 2, before=gap),
        "body": mk("a_body", regular, size),
        "step": mk("a_step", regular, size, after=gap / 2),
        "q": mk("a_q", bold, size),
        "hint": mk("a_hint", regular, size - 1, color=colors.HexColor("#334155")),
        "small": mk("a_small", regular, size - 2.5, color=colors.HexColor("#475569")),
    }


def _answer_lines(n: int, width: float, height: float) -> Table:
    t = Table([[""]] * n, colWidths=[width], rowHeights=[height] * n)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.8, colors.HexColor("#64748B"))]))
    return t


def accessible_worksheet_story(sheet, settings, language: str) -> list:
    """Flowables for one accessible worksheet (A4 portrait frame)."""
    st = accessible_styles(settings, language)
    width = A4[0] - 3 * cm
    hl = settings.highlight_keywords
    n_lines = 5 if settings.more_answer_space else 3
    line_h = (34 if settings.large_font else 28) if settings.more_answer_space else 24
    boxed = not settings.reduced_clutter

    story: list = [
        Paragraph(escape(f"Grade {sheet.grade}: {sheet.title}"), st["title"]),
        Paragraph(escape("Name: ____________________     Date: __________"), st["body"]),
        Paragraph("<b>" + escape(sheet.goal) + "</b>", st["body"]),
    ]
    for tip in sheet.how_to_use:
        story.append(Paragraph(escape(f"- {tip}"), st["step"]))

    ex = [Paragraph(escape("Example"), st["h"]), Paragraph(escape(sheet.example.problem), st["body"])]
    ex += [Paragraph(escape(f"Step {i}: {s}"), st["step"]) for i, s in enumerate(sheet.example.steps, 1)]
    ex.append(Paragraph("<b>" + escape(f"Answer: {sheet.example.answer}") + "</b>", st["body"]))
    story.append(KeepTogether(ex))

    for i, q in enumerate(sheet.questions, start=1):
        block: list = []
        if settings.one_question_per_section:
            block.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94A3B8"),
                                    spaceBefore=6, spaceAfter=6))
        block.append(Paragraph(escape(q.heading or f"Question {i}"), st["h"]))
        if len(q.steps) > 1 or settings.simple_instructions:
            block += [Paragraph(highlight_markup(f"{j}. {s}", q.keywords, hl), st["step"])
                      for j, s in enumerate(q.steps, start=1)]
        elif q.steps:
            block.append(Paragraph(highlight_markup(q.steps[0], q.keywords, hl), st["step"]))
        block.append(Paragraph(highlight_markup(q.question, q.keywords, hl), st["q"]))
        if q.visual_hint and settings.visual_hints:
            hint = Paragraph(escape(f"Hint: {q.visual_hint}"), st["hint"])
            if boxed:
                box = Table([[hint]], colWidths=[width])
                box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
                block.append(box)
            else:
                block.append(hint)
        block.append(_answer_lines(n_lines, width, line_h))
        if q.other_ways:
            block.append(Paragraph(escape(f"Or: {q.other_ways}"), st["small"]))
        block.append(Spacer(1, 10))
        story.append(KeepTogether(block))
    return story
