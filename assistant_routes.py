"""Flask routes for the AI Adaptive Classroom Assistant (one dashboard page + actions)."""
from __future__ import annotations

import logging
from urllib.parse import urlencode

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, url_for

import config
from adaptive import service, storage
from adaptive.forms import parse_feedback_form, parse_setup_form
from adaptive.models import (
    ALL_DAYS,
    ATTENTION_LABELS,
    BUCKET_LABELS,
    DAY_NAMES,
    FREE_LABEL,
    LESSON_SUBJECT_MAP,
    LOAD_LABELS,
    SUBJECT_CHOICES,
    attention_label,
)
from adaptive.timing import fmt12, parse_hhmm, school_day_end, valid_break_positions

log = logging.getLogger(__name__)
bp = Blueprint("assistant", __name__, url_prefix="/assistant")


def _profile_or_404(cid: str):
    profile = storage.load_profile(cid)
    if profile is None:
        abort(404)
    return profile


def _key(text: str) -> tuple[str, int]:
    day, _, period = (text or "").partition("-")
    if day not in ALL_DAYS or not period.isdigit():
        raise ValueError("Pick a period in the timetable first.")
    return day, int(period)


def lesson_link(profile, cell) -> str | None:
    """Link a timetable cell to the lesson planner, pre-filled for this period."""
    subject = LESSON_SUBJECT_MAP.get(cell.subject, cell.subject)
    if cell.kind != "subject" or subject not in config.SUBJECTS:
        return None
    params = [("subject", subject), ("duration", str(max(config.MIN_DURATION, profile.period_minutes))),
              ("attention_hint", attention_label(cell.attention))]
    if profile.grade in config.SUPPORTED_GRADES:
        params.append(("grades", str(profile.grade)))
    return url_for("planner") + "?" + urlencode(params)


def _safe_hhmm(value: str) -> int:
    try:
        return parse_hhmm(value)
    except ValueError:
        return 24 * 60


def _render_dashboard(profile, classes, tab, setup_errors=None, status=200):
    ctx = service.context_for(profile) if profile else None
    return render_template(
        "assistant/dashboard.html", classes=classes, p=profile, ctx=ctx, tab=tab,
        setup_errors=setup_errors or [], draft=bool(setup_errors),
        max_periods=config.MAX_PERIODS_PER_DAY, min_periods=config.MIN_PERIODS_PER_DAY,
        min_break=config.MIN_BREAK_MINUTES, max_break=config.MAX_BREAK_MINUTES,
        all_days=ALL_DAYS, day_names=DAY_NAMES, attention_labels=ATTENTION_LABELS,
        load_labels=LOAD_LABELS, bucket_labels=BUCKET_LABELS, subject_choices=SUBJECT_CHOICES,
        free_label=FREE_LABEL, attention_label=attention_label, fmt12=fmt12, parse_hhmm=_safe_hhmm,
        school_day_end=school_day_end, valid_break_positions=valid_break_positions,
        recs=service.dashboard_recommendations(profile, ctx) if profile else [],
        flat_attention=service.attention_is_flat(profile) if profile else False,
        lesson_link=lesson_link, use_ai=config.ADAPTIVE_USE_AI and bool(config.GROQ_API_KEY),
    ), status


@bp.get("/")
def dashboard():
    classes = storage.list_profiles()
    cid = request.args.get("class") or (classes[0].id if classes else None)
    profile = next((c for c in classes if c.id == cid), None) if cid else None
    tab = request.args.get("tab") or ("timetable" if profile and (profile.current or profile.proposal) else "setup")
    return _render_dashboard(profile, classes, tab)


@bp.post("/new")
def new_class():
    try:
        grade = int(request.form.get("grade", "4"))
    except ValueError:
        grade = 4
    if not 1 <= grade <= 12:
        flash("Grade must be between 1 and 12.", "error")
        return redirect(url_for("assistant.dashboard"))
    section = (request.form.get("section") or "A").strip()[:10] or "A"
    profile = service.new_profile(grade, section)
    storage.save_profile(profile)
    flash(f"Created {profile.name}. Review the setup, especially the attention pattern, then generate.",
          "success")
    return redirect(url_for("assistant.dashboard", **{"class": profile.id, "tab": "setup"}))


@bp.post("/<cid>/setup")
def save_setup(cid: str):
    profile = _profile_or_404(cid)
    updated, errors = parse_setup_form(request.form, profile)
    if errors:
        # Show the teacher's unsaved values again (nothing is discarded) with the problems listed.
        classes = [updated if c.id == cid else c for c in storage.list_profiles()]
        return _render_dashboard(updated, classes, "setup", setup_errors=errors, status=400)
    service.revalidate(updated)
    storage.save_profile(updated)
    msg = "Settings saved."
    if updated.current and not updated.current.validation.ok:
        msg += " The approved timetable no longer meets the new settings - generate an updated proposal."
    flash(msg, "success")
    return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "setup"}))


@bp.post("/<cid>/generate")
def generate(cid: str):
    profile = _profile_or_404(cid)
    try:
        version = service.generate_proposal(profile, regenerate=request.form.get("mode") == "regenerate")
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "setup"}))
    except Exception as exc:  # noqa: BLE001
        log.exception("Timetable generation failed")
        flash(f"Timetable generation failed: {exc}", "error")
        return redirect(url_for("assistant.dashboard", **{"class": cid}))
    if version.validation.ok:
        flash("New timetable proposal ready. Review it, then approve or reject.", "success")
    else:
        flash("A proposal was generated but it breaks some constraints - see the conflicts and "
              "suggestions below. It cannot be approved until they are resolved.", "error")
    return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "timetable"}))


@bp.post("/<cid>/approve")
def approve(cid: str):
    profile = _profile_or_404(cid)
    try:
        service.approve(profile)
        flash("Timetable approved and saved as the class's current timetable.", "success")
    except ValueError as exc:
        flash(str(exc), "error")
    return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "timetable"}))


@bp.post("/<cid>/reject")
def reject(cid: str):
    profile = _profile_or_404(cid)
    service.reject(profile)
    flash("Proposal rejected. The current timetable is unchanged.", "success")
    return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "timetable"}))


@bp.post("/<cid>/edit")
def edit(cid: str):
    profile = _profile_or_404(cid)
    target = request.form.get("target", "current")
    action = request.form.get("action", "")
    before = (profile.current if target == "current" else profile.proposal)
    old_score = before.score if before else None
    try:
        a = _key(request.form.get("cell_a", ""))
        b = _key(request.form["cell_b"]) if action == "swap" else None
        version = service.apply_edit(profile, target, action, a, b, request.form.get("subject"))
    except (ValueError, KeyError) as exc:
        flash(str(exc) if str(exc) else "Choose a second period to swap with.", "error")
    else:
        delta = "" if old_score is None or action == "lock" else             f" Score {old_score:.1f} -> {version.score:.1f}."
        if version.validation.ok and version.score < (old_score or 0) - 0.01 and action != "lock":
            flash("Change saved. Hard constraints are still met, but the schedule now fits your class's "
                  f"attention pattern less well.{delta} See 'Things to consider' below.", "success")
        elif version.validation.ok:
            flash(f"Change saved. All constraints are still met.{delta}", "success")
        else:
            flash("Change saved, but it breaks constraints: " + " ".join(version.validation.errors[:3]),
                  "error")
    return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "timetable"}))


@bp.post("/<cid>/feedback")
def feedback(cid: str):
    profile = _profile_or_404(cid)
    entry, errors = parse_feedback_form(request.form, profile)
    if errors:
        for e in errors:
            flash(e, "error")
    else:
        service.add_feedback(profile, entry)
        flash("Feedback saved. It will be used the next time you generate a proposal.", "success")
    return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "feedback"}))


@bp.post("/<cid>/feedback/<fid>/delete")
def delete_feedback(cid: str, fid: str):
    profile = _profile_or_404(cid)
    flash("Feedback entry deleted." if service.delete_feedback(profile, fid) else "Entry not found.",
          "success")
    return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "feedback"}))


@bp.get("/<cid>/pdf")
def pdf(cid: str):
    from adaptive.pdf import build_timetable_pdf

    profile = _profile_or_404(cid)
    which = request.args.get("version", "current")
    version = profile.proposal if which == "proposal" else profile.current
    if version is None:
        flash("There is no timetable to export yet.", "error")
        return redirect(url_for("assistant.dashboard", **{"class": cid}))
    try:
        path = build_timetable_pdf(profile, version)
    except Exception as exc:  # noqa: BLE001
        log.exception("Timetable PDF failed")
        flash(f"Could not create the PDF: {exc}", "error")
        return redirect(url_for("assistant.dashboard", **{"class": cid, "tab": "timetable"}))
    name = f"timetable_grade{profile.grade}{profile.section}_{which}.pdf"
    return send_file(path, mimetype="application/pdf", as_attachment=True, download_name=name)


@bp.post("/<cid>/delete")
def delete_class(cid: str):
    _profile_or_404(cid)
    storage.delete_profile(cid)
    flash("Class deleted.", "success")
    return redirect(url_for("assistant.dashboard"))
