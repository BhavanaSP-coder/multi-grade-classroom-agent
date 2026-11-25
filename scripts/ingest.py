"""Build or update the curriculum vector store from data/curriculum/.

Usage (from the project folder, with the virtual environment active):
    python scripts/ingest.py            # incremental: index new/changed files only
    python scripts/ingest.py --rebuild  # wipe and rebuild everything
    python scripts/ingest.py --status   # show what is indexed
    python scripts/ingest.py --query "fractions" --grades 3 4   # test retrieval
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the curriculum vector store.")
    parser.add_argument("--rebuild", action="store_true", help="wipe and rebuild the index")
    parser.add_argument("--status", action="store_true", help="show index status only")
    parser.add_argument("--query", help="run a test retrieval for this topic")
    parser.add_argument("--subject", default="Maths")
    parser.add_argument("--grades", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.status:
        st = rag.index_status()
        for f in st["files"]:
            mark = "OK " if f["indexed"] else "-- "
            demo = " [DEMO]" if f["is_demo"] else ""
            print(f"{mark}{f['name']:<50} grade={f['grade'] or 'all':<4} chunks={f['chunks']}{demo}")
        print(f"Total indexed chunks: {st['total_chunks']}  needs_sync={st['needs_sync']}")
        return 0

    if args.query:
        res = rag.retrieve(args.query, args.subject, args.grades)
        if res.error:
            print("ERROR:", res.error)
        if not res.sources:
            print("No relevant source retrieved.")
        for s in res.sources:
            print(f"[{s.label}] {s.source} ({s.grade_label}) score={s.score}")
            print("    " + " ".join(s.text.split())[:200] + " ...")
        print("By grade:", res.by_grade)
        return 0

    summary = rag.sync_index(force=args.rebuild)
    for key in ("added", "updated", "removed", "unchanged", "failed"):
        print(f"{key:>9}: {len(summary[key])}  {', '.join(summary[key])}")
    print(f"Total indexed chunks: {rag.index_status()['total_chunks']}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
