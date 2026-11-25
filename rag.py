"""Curriculum RAG: load .txt/.pdf files, chunk, embed locally, persist in ChromaDB, retrieve.

Index maintenance
-----------------
* ``sync_index()``      - incremental: (re)indexes new/changed files, removes deleted ones.
                          Uses ``vectorstore/manifest.json`` (file SHA-256 hashes).
* ``sync_index(force=True)`` - wipes the collection and rebuilds everything.
* ``index_file(path)``  - (re)index a single file (used after uploads).

CLI: ``python scripts/ingest.py`` / ``python scripts/ingest.py --rebuild``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from chains.schemas import RetrievalResult, SourceSnippet

log = logging.getLogger(__name__)

MANIFEST_PATH = config.VECTORSTORE_DIR / "manifest.json"
DEMO_MARKER = "DEMO CURRICULUM"

_lock = threading.RLock()
_vectorstore = None
_file_errors: dict[str, str] = {}  # last indexing problem per file, shown in the library

_GRADE_RE = re.compile(r"(?:grade|class|std|standard)[\s_\-:]*([1-5])(?!\d)", re.IGNORECASE)
_SUBJECTS = {
    "math": "Maths", "maths": "Maths", "mathematics": "Maths",
    "evs": "EVS", "science": "Science", "english": "English",
    "kannada": "Kannada", "hindi": "Hindi", "social": "Social Studies",
    "art": "Art and Craft", "music": "Music", "pe": "Physical Education",
    "physical": "Physical Education", "computer": "Computer Science", "computers": "Computer Science",
}


# --------------------------------------------------------------------------------------
# Embeddings + vector store
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embeddings():
    """Local sentence-transformers embeddings (downloaded once, no API key)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore():
    global _vectorstore
    with _lock:
        if _vectorstore is None:
            from langchain_chroma import Chroma

            _vectorstore = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=str(config.VECTORSTORE_DIR),
                collection_metadata={"hnsw:space": "cosine"},
            )
        return _vectorstore


def _store_is_empty(vs) -> bool:
    try:
        return not vs.get(limit=1, include=[])["ids"]
    except Exception:  # pragma: no cover - defensive
        return True


# --------------------------------------------------------------------------------------
# Manifest helpers
# --------------------------------------------------------------------------------------


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Manifest unreadable; it will be rebuilt.")
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Loading + splitting
# --------------------------------------------------------------------------------------


def list_curriculum_files() -> list[Path]:
    return sorted(
        p for p in config.CURRICULUM_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in config.ALLOWED_EXTENSIONS
    )


def detect_grade(filename: str, text: str = "") -> int:
    """Grade from file name (preferred) or the first lines of text; 0 = all grades."""
    m = _GRADE_RE.search(Path(filename).stem.replace("_", " "))
    if not m and text:
        m = _GRADE_RE.search(text[:600])
    return int(m.group(1)) if m else 0


def detect_subject(filename: str, text: str = "") -> str:
    first = re.split(r"[_\-\s]+", Path(filename).stem.lower())[0]
    if first in _SUBJECTS:
        return _SUBJECTS[first]
    m = re.search(r"subject\s*:\s*([A-Za-z ]+)", text[:400], re.IGNORECASE)
    if m:
        return _SUBJECTS.get(m.group(1).strip().lower().split()[0], m.group(1).strip().title())
    return "General"


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    encodings = ("utf-16",) if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1252")
    for enc in encodings:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


class CurriculumFileError(ValueError):
    """A curriculum file cannot be used; the message is written for the teacher."""


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for pages with text, using pypdf.

    Raises CurriculumFileError for corrupted, password-protected, empty or scanned PDFs."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            # Many PDFs are "encrypted" with an empty user password (print/copy restrictions only).
            try:
                unlocked = reader.decrypt("")
            except Exception:  # noqa: BLE001 - unsupported encryption also lands here
                unlocked = 0
            if not unlocked:
                raise CurriculumFileError(
                    f"'{path.name}' is password-protected. Remove the password and upload it again.")
        page_count = len(reader.pages)
    except CurriculumFileError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError, TypeError) as exc:
        raise CurriculumFileError(f"'{path.name}' could not be read - the PDF looks damaged or "
                                  f"incomplete ({type(exc).__name__}).") from exc
    if page_count == 0:
        raise CurriculumFileError(f"'{path.name}' has no pages.")

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one broken page should not hide the others
            log.warning("Could not read page %s of %s: %s", number, path.name, exc)
            continue
        if text.strip():
            pages.append((number, text))
    if not pages:
        raise CurriculumFileError(
            f"'{path.name}' has no selectable text (it is probably a scanned image). "
            "Run OCR on it or upload a text version.")
    return pages


def load_file(path: Path) -> list[Document]:
    """Load one curriculum file into page-level Documents with metadata.

    PDFs are read with pypdf (one Document per page); .txt files become one Document.
    Raises CurriculumFileError when the file has no usable text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        docs = [Document(page_content=text, metadata={"page": number})
                for number, text in read_pdf_pages(path)]
        full_head = docs[0].page_content
    else:
        text = _read_text_file(path)
        if not text.strip():
            raise CurriculumFileError(f"'{path.name}' is empty.")
        docs = [Document(page_content=text, metadata={"page": 0})]
        full_head = text

    grade = detect_grade(path.name, full_head)
    subject = detect_subject(path.name, full_head)
    is_demo = DEMO_MARKER in full_head[:400].upper()
    for d in docs:
        d.metadata.update(
            {"source": path.name, "grade": grade, "subject": subject, "is_demo": is_demo}
        )
    return docs


def split_documents(docs: list[Document], file_hash: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{file_hash[:16]}-{i}"
        c.metadata["chunk_index"] = i
    return chunks


# --------------------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------------------


def remove_source(filename: str) -> int:
    """Delete all chunks of one source file from the store."""
    with _lock:
        vs = get_vectorstore()
        ids = vs.get(where={"source": filename}, include=[])["ids"]
        if ids:
            vs.delete(ids=ids)
        manifest = _load_manifest()
        if manifest.pop(filename, None) is not None:
            _save_manifest(manifest)
        return len(ids)


def index_file(path: Path) -> int:
    """(Re)index one file. Returns number of chunks stored."""
    path = Path(path)
    with _lock:
        vs = get_vectorstore()
        sha = file_sha256(path)
        try:
            # Read first: a broken new version must not wipe the chunks of the old one.
            chunks = split_documents(load_file(path), sha)
        except CurriculumFileError as exc:
            _file_errors[path.name] = str(exc)
            raise
        _file_errors.pop(path.name, None)
        old_ids = vs.get(where={"source": path.name}, include=[])["ids"]
        if old_ids:
            vs.delete(ids=old_ids)
        if chunks:
            vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])

        manifest = _load_manifest()
        manifest[path.name] = {
            "sha256": sha,
            "chunks": len(chunks),
            "grade": chunks[0].metadata["grade"] if chunks else detect_grade(path.name),
            "subject": chunks[0].metadata["subject"] if chunks else detect_subject(path.name),
            "is_demo": bool(chunks and chunks[0].metadata["is_demo"]),
        }
        _save_manifest(manifest)
        log.info("Indexed %s (%d chunks)", path.name, len(chunks))
        return len(chunks)


def _reset_collection(vs) -> None:
    global _vectorstore
    try:
        vs.reset_collection()
    except Exception:
        vs.delete_collection()
        _vectorstore = None


def sync_index(force: bool = False) -> dict:
    """Bring the vector store in line with data/curriculum/. Returns a summary dict."""
    with _lock:
        vs = get_vectorstore()
        manifest = _load_manifest()
        if force or (manifest and _store_is_empty(vs)):
            _reset_collection(vs)
            manifest = {}
            _save_manifest(manifest)

        files = {p.name: p for p in list_curriculum_files()}
        added, updated, removed, unchanged, failed = [], [], [], [], []

        for name in list(manifest):
            if name not in files:
                remove_source(name)
                removed.append(name)

        for name, path in files.items():
            entry = manifest.get(name)
            try:
                if entry and entry.get("sha256") == file_sha256(path):
                    unchanged.append(name)
                    continue
                index_file(path)
                (updated if entry else added).append(name)
            except Exception as exc:  # keep going if one file is broken
                log.warning("Failed to index %s: %s", name, exc)
                failed.append(str(exc) if isinstance(exc, CurriculumFileError) else f"{name}: {exc}")

        return {
            "added": added, "updated": updated, "removed": removed,
            "unchanged": unchanged, "failed": failed,
        }


def index_status() -> dict:
    """Cheap status for the UI (reads the manifest only; no model loading)."""
    manifest = _load_manifest()
    files = list_curriculum_files()
    rows = []
    for p in files:
        entry = manifest.get(p.name)
        rows.append({
            "name": p.name,
            "type": p.suffix.lower().lstrip(".").upper(),
            "error": _file_errors.get(p.name, ""),
            "size_kb": round(p.stat().st_size / 1024, 1),
            "indexed": bool(entry) and entry.get("sha256") == file_sha256(p),
            "chunks": entry.get("chunks", 0) if entry else 0,
            "grade": entry.get("grade", detect_grade(p.name)) if entry else detect_grade(p.name),
            "is_demo": entry.get("is_demo", False) if entry else False,
        })
    return {
        "files": rows,
        "total_chunks": sum(r["chunks"] for r in rows if r["indexed"]),
        "needs_sync": any(not r["indexed"] for r in rows) or len(manifest) != len(files),
    }


def delete_curriculum_file(filename: str) -> int:
    """Delete a file from data/curriculum/ and its chunks from the vector store.
    Only names that are currently listed in the library are accepted (no path tricks)."""
    match = next((p for p in list_curriculum_files() if p.name == filename), None)
    if match is None:
        raise CurriculumFileError(f"'{filename}' is not in the curriculum library.")
    with _lock:
        match.unlink()
        _file_errors.pop(filename, None)
        return remove_source(filename)


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------


def retrieve(topic: str, subject: str, grades: list[int],
             k_per_grade: int = config.RETRIEVAL_K_PER_GRADE) -> RetrievalResult:
    """Retrieve relevant curriculum chunks for each grade.

    Each grade is searched separately (filtered to that grade + grade-agnostic files),
    so every grade gets its own grounding. Low-relevance chunks are dropped so the UI
    can honestly show "no relevant source".
    """
    query = f"{subject}: {topic}"
    result = RetrievalResult(query=query, by_grade={g: [] for g in grades})
    try:
        summary = sync_index()
        if summary["failed"]:
            log.warning("Some files failed to index: %s", summary["failed"])
        vs = get_vectorstore()
    except Exception as exc:
        log.exception("Vector store unavailable")
        result.error = f"Curriculum search unavailable: {exc}"
        return result

    if _store_is_empty(vs):
        result.store_empty = True
        return result

    seen: dict[str, SourceSnippet] = {}
    for g in grades:
        q = f"{subject} {topic} - Grade {g} learning outcomes, activities and assessment"
        try:
            hits = vs.similarity_search_with_relevance_scores(
                q, k=k_per_grade, filter={"grade": {"$in": [g, 0]}}
            )
        except Exception as exc:
            log.exception("Search failed for grade %s", g)
            result.error = f"Search failed for Grade {g}: {exc}"
            continue

        for doc, score in hits:
            if score < config.MIN_RELEVANCE:
                continue
            md = doc.metadata or {}
            cid = md.get("chunk_id") or getattr(doc, "id", None) or hashlib.md5(
                doc.page_content.encode("utf-8")).hexdigest()
            if cid not in seen:
                seen[cid] = SourceSnippet(
                    label=f"S{len(seen) + 1}",
                    chunk_id=cid,
                    source=md.get("source", "unknown"),
                    grade=int(md.get("grade", 0) or 0),
                    subject=md.get("subject", "General"),
                    page=int(md.get("page", 0) or 0),
                    score=round(float(score), 3),
                    text=doc.page_content.strip(),
                    is_demo=bool(md.get("is_demo", False)),
                )
            label = seen[cid].label
            if label not in result.by_grade[g]:
                result.by_grade[g].append(label)

    result.sources = list(seen.values())
    return result
