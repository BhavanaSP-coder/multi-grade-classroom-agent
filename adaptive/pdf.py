"""Printable PDF for an adaptive class timetable (reuses the lesson PDF helpers)."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

import config
from adaptive.models import DAY_NAMES, LOAD_LABELS, ClassProfile, TimetableVersion, attention_label
from adaptive.timing import compute_schedule
from pdf_generator import _footer, _grid, _p, _styles, escape

LOAD_COLORS = {"high": colors.HexColor("#FDE2E2"), "medium": colors.HexColor("#FEF3C7"),
               "low": colors.HexColor("#DCFCE7"), "fixed": colors.HexColor("#E2E8F0"),
               "free": colors.HexColor("#F1F5F9")}


def build_timetable_pdf(profile: ClassProfile, version: TimetableVersion, out_path: Path | None = None) -> Path:
    out_path = Path(out_path or config.PDF_DIR / f"timetable_{profile.id}_{version.id}.pdf")
    st = _styles("Helvetica", "Helvetica-Bold")
    doc = SimpleDocTemplate(str(out_path), pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.6 * cm,
                            title=f"Timetable {profile.name}", author="Multi-Grade Classroom Agent")
    periods, breaks, _ = compute_schedule(profile)
    status = {"approved": "Approved by teacher", "proposed": "PROPOSAL - not yet approved",
              "rejected": "Rejected"}[version.status]
    story = [
        _p(f"Weekly timetable - {profile.name}", st["title"]),
        _p(f"{profile.num_students} students | {profile.start_time}-{profile.end_time} | "
           f"{status} | {'Valid' if version.validation.ok else 'HAS CONSTRAINT ERRORS'}", st["center"]),
        Spacer(1, 8),
    ]

    days = profile.working_days
    width = landscape(A4)[0] - 2.4 * cm
    first = [2.6 * cm, 2.2 * cm]
    dcol = (width - sum(first)) / max(1, len(days))
    rows = [[_p("Time", st["cellb"]), _p("Attention*", st["cellb"])] + [_p(DAY_NAMES[d], st["cellb"]) for d in days]]
    extra = []
    by_key = {(c.day, c.period): c for c in version.cells}
    timeline = sorted([(p.start, "p", p) for p in periods] + [(b.start, "b", b) for b in breaks],
                      key=lambda x: x[0])
    for start, kind, item in timeline:
        r = len(rows)
        if kind == "b":
            rows.append([_p(item.label, st["cell"]), _p("", st["cell"])] + [_p(item.name, st["cellb"])] + [""] * (len(days) - 1))
            extra += [("SPAN", (2, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F8FAFC")),
                      ("ALIGN", (2, r), (-1, r), "CENTER")]
            continue
        any_cell = next((by_key[(d, item.index)] for d in days if (d, item.index) in by_key), None)
        att = any_cell.attention if any_cell else item.base_attention
        row = [_p(f"P{item.index}\n{item.label}", st["cellb"]), _p(attention_label(att), st["cell"])]
        for col, d in enumerate(days, start=2):
            c = by_key.get((d, item.index))
            if not c:
                row.append(_p("-", st["cell"]))
                continue
            sub = LOAD_LABELS.get(c.load, "Fixed" if c.kind == "fixed" else "")
            lock = " [locked]" if c.locked else ""
            row.append(Paragraph(f"<b>{escape(c.subject)}</b><br/>{escape(sub)}{lock}", st["cell"]))
            extra.append(("BACKGROUND", (col, r), (col, r), LOAD_COLORS[c.load or c.kind]))
        rows.append(row)
    story.append(_grid(rows, first + [dcol] * len(days), extra=extra))
    story.append(_p("* Attention levels are the teacher's observations (adjusted by recorded feedback), "
                    "not measurements. Colours: red = high cognitive load, amber = medium, green = low / "
                    "activity, grey = fixed or revision.", st["small"]))

    if not version.validation.ok:
        story.append(_p("Constraint errors", st["h2"]))
        story += [_p(f"- {e}", st["body"]) for e in version.validation.errors]
    story.append(_p("Why this timetable?", st["h2"]))
    story += [_p(f"- {e.text}", st["body"]) for e in version.explanations]
    if version.ai_summary:
        story.append(_p("AI summary (checked against the timetable)", st["h2"]))
        story += [_p(f"- {b}", st["body"]) for b in version.ai_summary]
    if version.notes:
        story.append(_p("Notes", st["h2"]))
        story += [_p(f"- {n}", st["body"]) for n in version.notes]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path
