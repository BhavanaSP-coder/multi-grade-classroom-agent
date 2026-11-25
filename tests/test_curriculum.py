"""Curriculum ingestion with pypdf, ChromaDB indexing, library routes and the new frontend pages.

Embeddings are replaced with a deterministic fake so the tests run offline and fast;
ChromaDB itself is real (persisted in a temp folder).
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pypdf import PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import config
import rag

ROOT = Path(__file__).resolve().parent.parent


# ---------------- PDF fixtures (built on the fly) ----------------

def text_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(60, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def blank_pdf() -> bytes:  # like a scanned page: no text layer
    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def encrypted_pdf(user_password: str) -> bytes:
    from pypdf import PdfReader

    w = PdfWriter()
    for page in PdfReader(io.BytesIO(text_pdf(["Grade 4 fractions: halves and quarters."]))).pages:
        w.add_page(page)
    w.encrypt(user_password=user_password, owner_password="owner-secret")
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Empty curriculum folder + fresh vector store using fake embeddings."""
    cur, vec, up = tmp_path / "curriculum", tmp_path / "vectorstore", tmp_path / "uploads"
    for d in (cur, vec, up):
        d.mkdir()
    monkeypatch.setattr(config, "CURRICULUM_DIR", cur)
    monkeypatch.setattr(config, "VECTORSTORE_DIR", vec)
    monkeypatch.setattr(config, "UPLOAD_DIR", up)
    monkeypatch.setattr(config, "COLLECTION_NAME", f"test_{tmp_path.name[-12:]}")
    monkeypatch.setattr(rag, "MANIFEST_PATH", vec / "manifest.json")
    monkeypatch.setattr(rag, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(rag, "_vectorstore", None)
    monkeypatch.setattr(rag, "_file_errors", {})
    yield cur
    rag._vectorstore = None


# ---------------- pypdf extraction ----------------

def test_pdf_text_is_extracted_per_page(tmp_path):
    p = tmp_path / "maths_grade3_fractions.pdf"
    p.write_bytes(text_pdf(["Subject: Mathematics\nGrade 3 fractions: one half", "Page two: quarters"]))
    pages = rag.read_pdf_pages(p)
    assert [n for n, _ in pages] == [1, 2]
    assert "one half" in pages[0][1] and "quarters" in pages[1][1]
    docs = rag.load_file(p)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert docs[0].metadata["grade"] == 3 and docs[0].metadata["source"] == p.name
    assert docs[0].metadata["subject"] == "Maths"


def test_pdf_without_page_text_is_skipped_but_others_kept(tmp_path):
    from pypdf import PdfReader

    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    for page in PdfReader(io.BytesIO(text_pdf(["Grade 2 halves"]))).pages:
        w.add_page(page)
    buf = io.BytesIO()
    w.write(buf)
    p = tmp_path / "mixed.pdf"
    p.write_bytes(buf.getvalue())
    assert [n for n, _ in rag.read_pdf_pages(p)] == [2]


@pytest.mark.parametrize("data,message", [
    (b"%PDF-1.7\nthis is not really a pdf", "damaged"),
    (b"", "damaged"),
    (blank_pdf(), "no selectable text"),
    (encrypted_pdf("secret"), "password-protected"),
])
def test_unusable_pdfs_give_teacher_friendly_errors(tmp_path, data, message):
    p = tmp_path / "bad.pdf"
    p.write_bytes(data)
    with pytest.raises(rag.CurriculumFileError, match=message):
        rag.load_file(p)


def test_pdf_with_only_owner_password_is_readable(tmp_path):
    p = tmp_path / "restricted.pdf"
    p.write_bytes(encrypted_pdf(""))
    assert "halves" in rag.load_file(p)[0].page_content


def test_txt_files_still_supported(tmp_path):
    p = tmp_path / "evs_class2_water.txt"
    p.write_text("[DEMO CURRICULUM]\nSubject: EVS\nWater around us.", encoding="utf-8")
    d = rag.load_file(p)[0]
    assert d.metadata["grade"] == 2 and d.metadata["is_demo"] and d.metadata["page"] == 0
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    with pytest.raises(rag.CurriculumFileError, match="empty"):
        rag.load_file(tmp_path / "empty.txt")


def test_no_pypdfloader_or_langchain_community_left():
    src = "\n".join(f.read_text(encoding="utf-8") for f in ROOT.glob("*.py"))
    src += "\n".join(f.read_text(encoding="utf-8") for f in (ROOT / "chains").glob("*.py"))
    assert "PyPDFLoader" not in src and "langchain_community" not in src
    assert "langchain-community" not in (ROOT / "requirements.txt").read_text(encoding="utf-8")


# ---------------- ingestion + ChromaDB ----------------

@pytest.mark.filterwarnings("ignore:Relevance scores must be between")
def test_ingest_index_retrieve_and_delete(library):
    (library / "maths_grade3_fractions.pdf").write_bytes(text_pdf(["Grade 3 fractions: halves, thirds and quarters."] * 2))
    (library / "maths_grade4_fractions.txt").write_text("Grade 4: equivalent fractions on a number line. " * 30, encoding="utf-8")
    (library / "scanned.pdf").write_bytes(blank_pdf())

    summary = rag.sync_index()
    assert sorted(summary["added"]) == ["maths_grade3_fractions.pdf", "maths_grade4_fractions.txt"]
    assert len(summary["failed"]) == 1 and "no selectable text" in summary["failed"][0]

    status = {f["name"]: f for f in rag.index_status()["files"]}
    assert status["maths_grade3_fractions.pdf"]["indexed"] and status["maths_grade3_fractions.pdf"]["type"] == "PDF"
    assert status["maths_grade4_fractions.txt"]["chunks"] > 1  # split by RecursiveCharacterTextSplitter
    assert "no selectable text" in status["scanned.pdf"]["error"]

    vs = rag.get_vectorstore()
    stored = vs.get(where={"source": "maths_grade3_fractions.pdf"}, include=["metadatas"])
    assert stored["ids"] and all(m["grade"] == 3 and m["page"] in (1, 2) for m in stored["metadatas"])

    config.MIN_RELEVANCE = -10.0  # fake embeddings have no meaningful similarity
    try:
        res = rag.retrieve("fractions", "Maths", [3, 4])
    finally:
        config.MIN_RELEVANCE = 0.30
    assert res.by_grade[3] and res.by_grade[4]
    assert {s.source for s in res.snippets_for_grade(3)} == {"maths_grade3_fractions.pdf"}

    assert rag.sync_index()["unchanged"] and not rag.sync_index()["added"]  # incremental
    removed = rag.delete_curriculum_file("maths_grade3_fractions.pdf")
    assert removed > 0 and not (library / "maths_grade3_fractions.pdf").exists()
    assert not vs.get(where={"source": "maths_grade3_fractions.pdf"}, include=[])["ids"]
    with pytest.raises(rag.CurriculumFileError):
        rag.delete_curriculum_file("../app.py")


def test_broken_reupload_keeps_old_chunks(library):
    good = library / "notes_grade5.pdf"
    good.write_bytes(text_pdf(["Grade 5 adding fractions with like denominators."]))
    assert rag.index_file(good) >= 1
    good.write_bytes(b"%PDF-1.4 truncated")
    with pytest.raises(rag.CurriculumFileError):
        rag.index_file(good)
    assert rag.get_vectorstore().get(where={"source": "notes_grade5.pdf"}, include=[])["ids"]


# ---------------- library routes ----------------

def test_upload_validation_and_index(client, library):
    r = client.post("/upload", data={"grade": "4", "file": (io.BytesIO(text_pdf(["Fractions on a number line."])), "unit.pdf")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"Added &#39;grade4_unit.pdf&#39;" in r.data and (library / "grade4_unit.pdf").exists()

    for name, data, msg in [("scan.pdf", blank_pdf(), b"no selectable text"),
                            ("locked.pdf", encrypted_pdf("pw"), b"password-protected"),
                            ("fake.pdf", b"%PDF-1.4 broken", b"damaged"),
                            ("notes.docx", b"hello", b"not supported")]:
        r = client.post("/upload", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data",
                        follow_redirects=True)
        assert msg in r.data, name
        assert not (library / name).exists()

    page = client.get("/curriculum").data.decode()
    assert "grade4_unit.pdf" in page and "Upload and index" in page and "PDF or TXT" in page

    r = client.post("/curriculum/delete", data={"filename": "grade4_unit.pdf"}, follow_redirects=True)
    assert b"Deleted" in r.data and not (library / "grade4_unit.pdf").exists()
    r = client.post("/curriculum/delete", data={"filename": "nope.pdf"}, follow_redirects=True)
    assert b"not in the curriculum library" in r.data


def test_rebuild_route(client, library):
    (library / "a_grade1.txt").write_text("Grade 1 sharing equally with stones. " * 5, encoding="utf-8")
    r = client.post("/rebuild-index", follow_redirects=True)
    assert b"Vector store rebuilt from 1 file(s)." in r.data


# ---------------- frontend: pages, navigation, no Tailwind ----------------

@pytest.mark.parametrize("url,active", [("/", "Dashboard"), ("/planner", "Lesson Planner"),
                                        ("/assistant/", "Adaptive Classroom Assistant"),
                                        ("/curriculum", "Curriculum Library")])
def test_pages_render_with_active_navigation(client, library, url, active):
    html = client.get(url).data.decode()
    assert "<main" in html and "<nav" in html and "<header" in html
    current = re.search(r'aria-current="page"><svg.*?</svg><span>(.*?)</span>', html, re.S)
    assert current and current.group(1) == active
    assert "tailwind" not in html.lower()


def test_no_tailwind_or_inline_styles_in_templates():
    for f in (ROOT / "templates").rglob("*.html"):
        text = f.read_text(encoding="utf-8")
        assert "tailwind" not in text.lower(), f
        assert 'style="' not in text, f
    for name in ["style", "layout", "dashboard", "lesson", "assistant", "curriculum", "responsive"]:
        assert (ROOT / "static" / "css" / f"{name}.css").exists()
    for name in ["app", "lesson", "assistant", "curriculum"]:
        assert (ROOT / "static" / "js" / f"{name}.js").exists()


def test_dashboard_statistics_come_from_real_data(client, library, sample_lesson):
    from adaptive import service, storage
    from chains.pipeline import save_lesson

    html = client.get("/").data.decode()
    assert "No lessons yet" in html and "No classes yet" in html
    save_lesson(sample_lesson)
    storage.save_profile(service.new_profile(3, "B"))
    (library / "x_grade2.txt").write_text("Grade 2 halves and quarters. " * 3, encoding="utf-8")
    html = client.get("/").data.decode()
    stats = dict(re.findall(r"<dt>(.*?)</dt>\s*<dd>(\d+)</dd>", html))
    assert stats == {"Lessons generated": "1", "Curriculum files": "1", "Classes": "1",
                     "Timetables awaiting approval": "0"}
    assert sample_lesson.outline.lesson_title in html and "Grade 3 - B" in html


def test_planner_shows_field_errors_next_to_inputs(client):
    r = client.post("/generate", data={"topic": "", "subject": "Maths", "duration": "5", "language": "English"})
    html = r.data.decode()
    assert r.status_code == 400
    assert re.search(r'data-error-for="topic">Please enter a topic', html)
    assert re.search(r'data-error-for="grades">Select at least one grade', html)
    assert re.search(r'data-error-for="duration">Class duration must be between', html)


def test_closing_time_is_derived_from_periods_and_breaks(client):
    from adaptive import service, storage
    from werkzeug.datastructures import MultiDict

    p = service.new_profile(4, "A")
    storage.save_profile(p)
    form = MultiDict([("grade", "4"), ("section", "A"), ("num_students", "30"), ("start_time", "08:00"),
                      ("num_periods", "6"), ("period_minutes", "45"), ("att_count", "0"), ("working_days", "Mon"),
                      ("br_name", "Lunch"), ("br_after", "3"), ("br_minutes", "30"),
                      ("subj_name", "Mathematics"), ("subj_load", "high"), ("subj_ppw", "1"), ("subj_max", "1"),
                      ("subj_teacher", "")])
    r = client.post(f"/assistant/{p.id}/setup", data=form)
    assert r.status_code == 302
    saved = storage.load_profile(p.id)
    assert saved.end_time == "13:00"  # 08:00 + 6 x 45 min + 30 min lunch
    assert saved.attention[0].start == "08:00" and saved.attention[-1].end == "13:00"


def test_oral_alternatives_option():
    from chains.accessible_chain import build_rules
    from chains.schemas import AccessibilitySettings, DYSLEXIA_PRESET

    assert "oral_alternatives" in DYSLEXIA_PRESET
    assert "ORAL option" in build_rules(AccessibilitySettings(oral_alternatives=True))
    assert "ORAL option" not in build_rules(AccessibilitySettings())


@pytest.mark.parametrize("name,subject", [
    ("evs_grade3_water.txt", "EVS"), ("science_grade5_x.txt", "Science"), ("english_grade1_x.txt", "English"),
    ("kannada_grade2_x.txt", "Kannada"), ("hindi_grade4_x.txt", "Hindi"), ("social_studies_grade5_x.txt", "Social Studies"),
    ("art_craft_grade3_x.txt", "Art and Craft"), ("music_grade1_x.txt", "Music"),
    ("physical_education_grade4_x.txt", "Physical Education"), ("pe_grade2_x.txt", "Physical Education"),
    ("computer_science_grade5_x.txt", "Computer Science"), ("maths_grade1_x.txt", "Maths"), ("notes.txt", "General")])
def test_subject_detected_from_file_name(name, subject):
    assert rag.detect_subject(name) == subject


def test_demo_library_covers_every_subject_and_grade():
    """Every lesson-planner subject has a DEMO file for each grade 1-5."""
    files = list((ROOT / "data" / "curriculum").glob("*.txt"))
    covered = {(rag.detect_subject(f.name), rag.detect_grade(f.name)) for f in files}
    subject_names = {"Maths": "Maths", "EVS": "EVS"}
    for s in config.SUBJECTS:
        for g in config.SUPPORTED_GRADES:
            assert (subject_names.get(s, s), g) in covered, (s, g)
    assert all("DEMO CURRICULUM" in f.read_text(encoding="utf-8")[:200] for f in files)
