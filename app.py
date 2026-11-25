"""Flask web app for the Multi-Grade Classroom Agent.

Pages:  /            Dashboard
        /planner     Lesson Planner           /lesson/<id>   Lesson results
        /assistant/  Adaptive Classroom Assistant (blueprint in assistant_routes.py)
        /curriculum  Curriculum Library

Run:  python app.py   then open http://127.0.0.1:5000
"""
from __future__ import annotations

import logging
import re
import shutil
import threading

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from markupsafe import Markup, escape
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import config
import rag
from chains.llm import friendly_llm_error
from chains.schemas import ACCESSIBILITY_OPTIONS, DYSLEXIA_PRESET, AccessibilitySettings
from validators import ATTENTION_HINTS, parse_accessibility, validate_lesson_form, validate_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
)

SAMPLE_LESSON = {
    "topic": "Fractions",
    "subject": "Maths",
    "grades": ["1", "2", "3", "4", "5"],
    "duration": "45",
    "language": "English",
}
DEFAULT_FORM = {"topic": "", "subject": "Maths", "grades": [1, 2, 3, 4, 5],
                "duration": "45", "language": "English",
                "accessibility": AccessibilitySettings(), "attention_hint": "", "field_errors": {}}
PLANNER_QUERY_KEYS = {"subject", "topic", "grades", "duration", "attention_hint", "language"}


@app.template_filter("highlight")
def highlight_filter(text, keywords=None, enabled=True):
    """Escape text, then wrap keywords in <mark> (used by the accessible worksheet preview)."""
    safe = str(escape(text or ""))
    kws = sorted({str(escape(k)) for k in (keywords or []) if k and k.strip()}, key=len, reverse=True)
    if not enabled or not kws:
        return Markup(safe)
    pattern = re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE)
    return Markup(pattern.sub(lambda m: f'<mark class="kw">{m.group(0)}</mark>', safe))


@app.context_processor
def _globals():
    return {"acc_options": ACCESSIBILITY_OPTIONS, "dyslexia_preset": sorted(DYSLEXIA_PRESET),
            "cfg": config, "api_key_set": bool(config.GROQ_API_KEY)}


class _SampleForm(dict):
    """dict with getlist() so validators treat it like request.form."""

    def getlist(self, key):
        val = self.get(key, [])
        return val if isinstance(val, list) else [val]


def _library_status() -> dict:
    try:
        return rag.index_status()
    except Exception as exc:  # noqa: BLE001
        log.exception("index_status failed")
        return {"files": [], "total_chunks": 0, "needs_sync": True, "error": str(exc)}


def _render_planner(form=None, errors=None, status_code=200):
    form = form or DEFAULT_FORM
    return render_template(
        "lesson/planner.html",
        form=form,
        errors=errors or [],
        field_errors=form.get("field_errors", {}),
        library=_library_status(),
        subjects=config.SUBJECTS,
        languages=config.LANGUAGES,
        grades=config.SUPPORTED_GRADES,
    ), status_code


def _run_generation(form):
    req, errors, values = validate_lesson_form(form)
    if errors:
        return _render_planner(values, errors, 400)
    from chains.pipeline import generate_lesson  # lazy: keeps app start fast

    try:
        result = generate_lesson(req)
    except Exception as exc:  # noqa: BLE001
        log.exception("Lesson generation failed")
        return _render_planner(values, [friendly_llm_error(exc)], 502)
    return redirect(url_for("lesson", lesson_id=result.lesson_id))


# --------------------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------------------


@app.get("/")
def dashboard():
    # Older links (e.g. from the Adaptive Assistant) pre-filled the planner at "/".
    if PLANNER_QUERY_KEYS & set(request.args):
        return redirect(url_for("planner", **request.args.to_dict(flat=False)))

    from adaptive import storage
    from chains.pipeline import count_lessons, list_lessons

    library = _library_status()
    classes = storage.list_profiles()
    stats = {
        "lessons": count_lessons(),
        "curriculum_files": len(library["files"]),
        "curriculum_chunks": library["total_chunks"],
        "classes": len(classes),
        "approved": sum(1 for c in classes if c.current),
        "pending": sum(1 for c in classes if c.proposal),
    }
    return render_template("dashboard.html", stats=stats, recent=list_lessons(limit=5),
                           classes=classes[:5], library=library)


# --------------------------------------------------------------------------------------
# Lesson planner and results
# --------------------------------------------------------------------------------------


@app.get("/planner")
def planner():
    """Lesson form. Query parameters (from the Adaptive Assistant) can pre-fill it."""
    q = request.args
    form = dict(DEFAULT_FORM)
    if q.get("subject") in config.SUBJECTS:
        form["subject"] = q["subject"]
    if q.get("duration", "").isdigit():
        form["duration"] = q["duration"]
    if q.get("topic"):
        form["topic"] = q["topic"][: config.TOPIC_MAX_LEN]
    grades = [int(g) for g in q.getlist("grades") if g.isdigit() and int(g) in config.SUPPORTED_GRADES]
    if grades:
        form["grades"] = grades
    if q.get("attention_hint") in ATTENTION_HINTS:
        form["attention_hint"] = q["attention_hint"]
    if q.get("language") in config.LANGUAGES:
        form["language"] = q["language"]
    return _render_planner(form)[0]


@app.post("/generate")
def generate():
    return _run_generation(request.form)


@app.post("/sample")
def sample():
    return _run_generation(_SampleForm(SAMPLE_LESSON))


@app.get("/lesson/<lesson_id>")
def lesson(lesson_id: str):
    from chains.pipeline import load_lesson

    result = load_lesson(lesson_id)
    if result is None:
        flash("That lesson was not found. It may have been deleted - please generate it again.", "error")
        return redirect(url_for("planner"))
    return render_template("lesson/results.html", r=result, languages=config.LANGUAGES,
                           min_relevance=config.MIN_RELEVANCE)


@app.post("/lesson/<lesson_id>/accessible")
def lesson_accessible(lesson_id: str):
    """(Re)generate dyslexia-friendly worksheets for an existing lesson with new settings."""
    from chains.pipeline import load_lesson, regenerate_accessible

    result = load_lesson(lesson_id)
    if result is None:
        abort(404)
    settings = parse_accessibility(request.form)
    try:
        regenerate_accessible(result, settings)
    except Exception as exc:  # noqa: BLE001
        log.exception("Accessible regeneration failed")
        flash(friendly_llm_error(exc), "error")
    else:
        flash("Accessible worksheets updated - review the preview below." if settings.any_enabled
              else "Accessibility settings turned off; accessible worksheets removed.", "success")
    return redirect(url_for("lesson", lesson_id=lesson_id) + "#grades")


@app.post("/lesson/<lesson_id>/edit")
def lesson_edit(lesson_id: str):
    """Save inline teacher edits: JSON {"edits": {"contents.0.activity_title": "..."}}."""
    from chains.pipeline import apply_edits, load_lesson

    result = load_lesson(lesson_id)
    if result is None:
        return jsonify({"ok": False, "errors": ["Lesson not found."]}), 404
    payload = request.get_json(silent=True) or {}
    edits = payload.get("edits")
    if not isinstance(edits, dict) or not edits:
        return jsonify({"ok": False, "errors": ["No changes were sent."]}), 400
    if len(edits) > 300:
        return jsonify({"ok": False, "errors": ["Too many changes at once."]}), 400
    try:
        updated, errors = apply_edits(result, edits)
    except Exception as exc:  # noqa: BLE001
        log.exception("Edit failed")
        return jsonify({"ok": False, "errors": [f"Could not save: {exc}"]}), 400
    return jsonify({"ok": not errors, "errors": errors,
                    "saved": updated.edited_at is not None and updated is not result})


@app.get("/lesson/<lesson_id>/pdf")
def lesson_pdf(lesson_id: str):
    from chains.pipeline import load_lesson
    from pdf_generator import build_lesson_pdf

    result = load_lesson(lesson_id)
    if result is None:
        abort(404)
    only_accessible = request.args.get("only") == "accessible"
    try:
        path = build_lesson_pdf(result, only_accessible=only_accessible)
    except Exception as exc:  # noqa: BLE001
        log.exception("PDF generation failed")
        flash(f"Could not create the PDF: {exc}", "error")
        return redirect(url_for("lesson", lesson_id=lesson_id))
    safe_topic = secure_filename(result.request.topic) or "lesson"
    kind = "accessible_worksheets" if only_accessible else "grades"
    return send_file(path, mimetype="application/pdf", as_attachment=True,
                     download_name=f"{safe_topic}_{kind}_{'-'.join(map(str, result.request.grades))}.pdf")


# --------------------------------------------------------------------------------------
# Curriculum library
# --------------------------------------------------------------------------------------


@app.get("/curriculum")
def curriculum():
    return render_template("curriculum/library.html", library=_library_status(),
                           grades=config.SUPPORTED_GRADES)


@app.post("/upload")
def upload():
    back = redirect(url_for("curriculum"))
    file = request.files.get("file")
    if file is None or not file.filename:
        flash("Please choose a .pdf or .txt file to upload.", "error")
        return back

    filename = secure_filename(file.filename)
    if not filename or "." not in filename:
        flash("That file name is not valid. Rename it (e.g. maths_grade3_fractions.pdf).", "error")
        return back

    grade = request.form.get("grade", "auto")
    if grade in {str(g) for g in config.SUPPORTED_GRADES} and rag.detect_grade(filename) == 0:
        filename = f"grade{grade}_{filename}"

    data = file.read()
    errors = validate_upload(filename, data)
    if errors:
        for e in errors:
            flash(e, "error")
        return back

    staged = config.UPLOAD_DIR / filename
    target = config.CURRICULUM_DIR / filename
    try:
        staged.write_bytes(data)
        rag.load_file(staged)  # checks the content (damaged, password-protected, scanned...)
        replaced = target.exists()
        shutil.move(str(staged), str(target))
        chunks = rag.index_file(target)
    except rag.CurriculumFileError as exc:
        staged.unlink(missing_ok=True)
        flash(str(exc), "error")
        return back
    except Exception as exc:  # noqa: BLE001
        log.exception("Upload failed")
        staged.unlink(missing_ok=True)
        flash(f"Could not add '{filename}': {exc}", "error")
        return back

    flash(f"{'Updated' if replaced else 'Added'} '{filename}' ({chunks} chunks indexed).", "success")
    return back


@app.post("/curriculum/delete")
def delete_curriculum_file():
    name = request.form.get("filename", "")
    try:
        removed = rag.delete_curriculum_file(name)
    except rag.CurriculumFileError as exc:
        flash(str(exc), "error")
    except Exception as exc:  # noqa: BLE001
        log.exception("Delete failed")
        flash(f"Could not delete '{name}': {exc}", "error")
    else:
        flash(f"Deleted '{name}' and its {removed} indexed chunks.", "success")
    return redirect(url_for("curriculum"))


@app.post("/rebuild-index")
def rebuild_index():
    try:
        summary = rag.sync_index(force=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("Rebuild failed")
        flash(f"Rebuild failed: {exc}", "error")
    else:
        n = len(summary["added"]) + len(summary["updated"])
        msg = f"Vector store rebuilt from {n} file(s)."
        if summary["failed"]:
            msg += " Not indexed: " + " ".join(summary["failed"])
        flash(msg, "error" if summary["failed"] else "success")
    return redirect(url_for("curriculum"))


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True, "model": config.GROQ_MODEL, "api_key_set": bool(config.GROQ_API_KEY)}


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    flash(f"File is too large. The maximum upload size is {config.MAX_UPLOAD_MB} MB.", "error")
    return redirect(url_for("curriculum"))


@app.errorhandler(404)
def not_found(_exc):
    return render_template("not_found.html"), 404


from assistant_routes import bp as assistant_bp  # noqa: E402

app.register_blueprint(assistant_bp)


def _warm_up():
    """Load the embedding model and sync the index in the background at startup."""
    try:
        summary = rag.sync_index()
        log.info("Curriculum index ready: %s", {k: len(v) for k, v in summary.items()})
    except Exception:  # noqa: BLE001
        log.exception("Index warm-up failed (will retry on first lesson)")


if __name__ == "__main__":
    if not config.GROQ_API_KEY:
        log.warning("GROQ_API_KEY is not set - add it to .env before generating lessons.")
    threading.Thread(target=_warm_up, daemon=True).start()
    app.run(host="127.0.0.1", port=config.PORT, debug=config.FLASK_DEBUG,
            use_reloader=False, threaded=True)
