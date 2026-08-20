#!/usr/bin/env python3
"""
04_add_subject.py – add one subject to a course.

    python scripts/04_add_subject.py --list
    python scripts/04_add_subject.py --course 99B --code AVN101 --name "Avionics Fundamentals"

Interim tool. Subject management moves to the Phase 2 Admin UI under the
TRAINING_CONTROL role; this stays useful afterwards for scripted / bulk setup.

Idempotent: re-adding an existing (course, subject_code) pair is a no-op.
Subject codes are unique PER COURSE, so the same code may be used in 99B and
in 100B - those are two independent subjects with independent history.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import get_settings
from backend.database import check_connection

settings = get_settings()

# ── Colours ───────────────────────────────────────────────────────────────────
_G, _R, _Y, _C, _B, _RS = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"

def ok(m):   print(f"  {_G}✓{_RS} {m}")
def err(m):  print(f"  {_R}✗{_RS} {m}"); sys.exit(1)
def warn(m): print(f"  {_Y}!{_RS} {m}")
def info(m): print(f"  {_C}▸{_RS} {m}")

# Reserved for the pre-migration history bucket created by 03_migrate_multicourse.
# It is deliberately is_active=FALSE and must never become a selectable subject.
RESERVED_CODES = {"LEGACY-CS"}


async def show_catalog(conn) -> None:
    """Print every course with its subjects, archived ones marked."""
    rows = (await conn.execute(text("""
        SELECT c.code, c.name, c.is_active,
               s.subject_code, s.subject_name, s.is_active
        FROM courses c
        LEFT JOIN subjects s ON s.course_id = c.id
        ORDER BY c.id, s.subject_code
    """))).all()

    current = None
    for ccode, cname, cactive, scode, sname, sactive in rows:
        if ccode != current:
            current = ccode
            flag = "" if cactive else "  (inactive)"
            print(f"\n  {_B}{ccode}{_RS}  {cname}{flag}")
        if scode is None:
            info("(no subjects yet)")
        elif sactive:
            ok(f"{scode:<12} {sname}")
        else:
            warn(f"{scode:<12} {sname}   [archived - not selectable]")
    print()


async def add_subject(conn, course_code: str, subject_code: str, subject_name: str) -> None:
    course = (await conn.execute(text(
        "SELECT id, name FROM courses WHERE code = :c"), {"c": course_code})).first()
    if course is None:
        valid = (await conn.execute(text(
            "SELECT string_agg(code, ', ' ORDER BY id) FROM courses"))).scalar()
        err(f"No course with code {course_code!r}.\n    Valid codes: {valid}")

    course_id, course_name = course

    r = await conn.execute(text("""
        INSERT INTO subjects (course_id, subject_code, subject_name, is_active)
        VALUES (:cid, :scode, :sname, TRUE)
        ON CONFLICT (course_id, subject_code) DO NOTHING
        RETURNING id
    """), {"cid": course_id, "scode": subject_code, "sname": subject_name})
    new_id = r.scalar()

    if new_id is None:
        existing = (await conn.execute(text("""
            SELECT id, subject_name, is_active FROM subjects
            WHERE course_id = :cid AND subject_code = :scode
        """), {"cid": course_id, "scode": subject_code})).first()
        warn(
            f"{course_code} / {subject_code} already exists "
            f"(id={existing[0]}, name={existing[1]!r}, active={existing[2]}) - nothing changed"
        )
    else:
        ok(f"Added {course_code} / {subject_code} — {subject_name}  (id={new_id})")
        info(f"Course: {course_name}")
        info("Selectable in the live monitor's Subject dropdown within ~10s")


async def main(args) -> None:
    print(f"\n{_B}{_C}════════════════════════════════════════{_RS}")
    print(f"{_B}{_C}   Course / subject catalogue            {_RS}")
    print(f"{_B}{_C}════════════════════════════════════════{_RS}")

    if not await check_connection():
        err(f"Cannot reach database. Check DATABASE_URL in .env\n    {settings.database_url}")

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    try:
        async with engine.begin() as conn:
            if args.list:
                await show_catalog(conn)
                return

            code = args.code.strip().upper()
            if code in RESERVED_CODES:
                err(
                    f"{code!r} is reserved for pre-migration history and must stay archived.\n"
                    f"    Choose a different subject code."
                )

            await add_subject(conn, args.course.strip().upper(), code, args.name.strip())
            print()
            await show_catalog(conn)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add one subject to a course")
    ap.add_argument("--list", action="store_true",
                    help="show all courses and their subjects, then exit")
    ap.add_argument("--course", help="course code, e.g. 99B")
    ap.add_argument("--code",   help="subject code, e.g. AVN101")
    ap.add_argument("--name",   help="subject name, e.g. \"Avionics Fundamentals\"")
    parsed = ap.parse_args()

    if not parsed.list and not (parsed.course and parsed.code and parsed.name):
        ap.error("--course, --code and --name are all required (or use --list)")

    asyncio.run(main(parsed))
