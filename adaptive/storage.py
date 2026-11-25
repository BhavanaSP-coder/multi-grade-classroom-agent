"""JSON persistence for class profiles (data/adaptive/<id>.json)."""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path

import config
from adaptive.models import ClassProfile

log = logging.getLogger(__name__)
_ID = re.compile(r"^[0-9a-f]{12}$")
_lock = threading.RLock()


def _path(profile_id: str, base: Path | None = None) -> Path:
    if not _ID.match(profile_id or ""):
        raise ValueError("Invalid class id.")
    return (base or config.ADAPTIVE_DIR) / f"{profile_id}.json"


def save_profile(profile: ClassProfile, base: Path | None = None) -> None:
    profile.updated_at = datetime.now()
    path = _path(profile.id, base)
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)


def load_profile(profile_id: str, base: Path | None = None) -> ClassProfile | None:
    try:
        path = _path(profile_id, base)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return ClassProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("Could not read class profile %s", profile_id)
        return None


def list_profiles(base: Path | None = None) -> list[ClassProfile]:
    folder = base or config.ADAPTIVE_DIR
    out = []
    for p in sorted(folder.glob("*.json")):
        prof = load_profile(p.stem, base)
        if prof:
            out.append(prof)
    return sorted(out, key=lambda c: (c.grade, c.section))


def delete_profile(profile_id: str, base: Path | None = None) -> bool:
    try:
        path = _path(profile_id, base)
    except ValueError:
        return False
    with _lock:
        if path.exists():
            path.unlink()
            return True
    return False
