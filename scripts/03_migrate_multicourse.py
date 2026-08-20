#!/usr/bin/env python3
"""
03_migrate_multicourse.py – Phase 1: multi-course / multi-subject / four-role
schema migration.

Run from the project root:
    python scripts/03_migrate_multicourse.py --dry-run   # rehearse, then roll back
    python scripts/03_migrate_multicourse.py             # apply

What it does
────────────
1. Pre-flight: connectivity, current-state report, HOD_SEED_PASSWORD present
2. Creates user_role_enum + courses, subjects, users, instructor_assignments
3. Adds students.course_id and sessions.subject_id (nullable at first)
4. Seeds courses 99B-104B (99B populated; 100B-104B valid, active, empty)
5. Creates the 99B LEGACY-CS subject that existing session history maps onto
6. Backfills students.course_id and sessions.subject_id
7. Asserts the backfill is complete, then SET NOT NULL + indexes
8. Seeds one HOD login and prints a verification summary

Everything from step 2 onward runs inside ONE transaction. Postgres has
transactional DDL, so any failure rolls the database back to its pre-migration
state - there is no half-migrated outcome. --dry-run performs every step and
then deliberately rolls back.

Non-destructive: no row is deleted, no column is dropped. Only students.course_id
and sessions.subject_id are written, on 15 and 65 rows respectively.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# passlib 1.7.4 probes bcrypt.__about__, removed in bcrypt 4.1+. The failure is
# trapped internally and hashing works correctly; silence the alarming traceback.
logging.getLogger("passlib.handlers.bcrypt").setLevel(logging.ERROR)

from passlib.context import CryptContext
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import get_settings
from backend.database import check_connection

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Colours ───────────────────────────────────────────────────────────────────
_G, _R, _Y, _C, _B, _RS = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"

def ok(m):   print(f"  {_G}✓{_RS} {m}")
def err(m):  print(f"  {_R}✗{_RS} {m}"); sys.exit(1)
def warn(m): print(f"  {_Y}!{_RS} {m}")
def info(m): print(f"  {_C}▸{_RS} {m}")

# ── Seed constants ────────────────────────────────────────────────────────────

COURSES = [
    ("99B",  "CAE Avionics 99B"),
    ("100B", "CAE Avionics 100B"),
    ("101B", "CAE Avionics 101B"),
    ("102B", "CAE Avionics 102B"),
    ("103B", "CAE Avionics 103B"),
    ("104B", "CAE Avionics 104B"),
]

# Real subjects are entered later through the Admin UI, not hardcoded here.
# This one exists only so the 65 pre-migration sessions have somewhere truthful
# to live: they were free-text 'Computer Science', not any real avionics
# subject, and folding them into a real one would corrupt its statistics.
# is_active=FALSE keeps it out of pickers while history stays fully queryable.
LEGACY_SUBJECT_CODE = "LEGACY-CS"
LEGACY_SUBJECT_NAME = "Computer Science (pre-migration history)"
LEGACY_COURSE_CODE  = "99B"

# The classroom whose roster and session history belong to 99B.
LEGACY_CLASSROOM_ID = 1

HOD_USERNAME  = "hod"
HOD_FULL_NAME = "Head of Department"


# ── Step 1 – pre-flight ───────────────────────────────────────────────────────

async def preflight(conn) -> str:
    print(f"\n{_B}[1/8] Pre-flight checks …{_RS}")

    hod_password = os.getenv("HOD_SEED_PASSWORD", "").strip()
    if not hod_password:
        err(
            "HOD_SEED_PASSWORD is not set.\n"
            "    The migration seeds a working HOD login and will not invent a\n"
            "    password for it. Add a line to .env, then re-run:\n\n"
            "        HOD_SEED_PASSWORD=your-chosen-password\n"
        )
    ok("HOD_SEED_PASSWORD is set")

    for table, expected in (("students", 15), ("sessions", 65)):
        n = (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar()
        note = "" if n == expected else f"  (expected {expected} at design time)"
        info(f"{table:<22} {n:>7} rows{note}")

    for table in ("attendance_records", "behavior_events", "face_detections"):
        n = (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar()
        info(f"{table:<22} {n:>7} rows  (untouched by this migration)")

    # Any student outside the legacy classroom has no defensible course mapping.
    stray = (await conn.execute(text(
        "SELECT count(*) FROM students WHERE classroom_id IS DISTINCT FROM :cid"
    ), {"cid": LEGACY_CLASSROOM_ID})).scalar()
    if stray:
        err(
            f"{stray} student(s) are not in classroom {LEGACY_CLASSROOM_ID}.\n"
            f"    This migration maps only classroom {LEGACY_CLASSROOM_ID}'s roster onto course 99B.\n"
            f"    Decide which course those students belong to before re-running."
        )
    ok(f"all students are in classroom {LEGACY_CLASSROOM_ID} → mappable to 99B")

    stray_s = (await conn.execute(text(
        "SELECT count(*) FROM sessions WHERE classroom_id IS DISTINCT FROM :cid"
    ), {"cid": LEGACY_CLASSROOM_ID})).scalar()
    if stray_s:
        err(f"{stray_s} session(s) are not in classroom {LEGACY_CLASSROOM_ID}; mapping is ambiguous.")
    ok(f"all sessions are in classroom {LEGACY_CLASSROOM_ID} → mappable to 99B/{LEGACY_SUBJECT_CODE}")

    return hod_password


# ── Step 2 – new type + new tables ────────────────────────────────────────────

async def create_new_tables(conn) -> None:
    print(f"\n{_B}[2/8] Creating enum type and new tables …{_RS}")

    # CREATE TYPE has no IF NOT EXISTS; guard so re-runs are safe.
    await conn.execute(text("""
        DO $$ BEGIN
            CREATE TYPE user_role_enum AS ENUM
                ('HOD', 'INSTRUCTOR', 'STUDENT', 'TRAINING_CONTROL');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """))
    ok("user_role_enum  ('HOD','INSTRUCTOR','STUDENT','TRAINING_CONTROL')")

    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS courses (
            id          SERIAL       PRIMARY KEY,
            code        VARCHAR(20)  NOT NULL,
            name        VARCHAR(200) NOT NULL,
            is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
            created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
            CONSTRAINT uq_courses_code UNIQUE (code)
        );
    """))
    ok("courses")

    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS subjects (
            id            SERIAL       PRIMARY KEY,
            course_id     INTEGER      NOT NULL
                          REFERENCES courses(id) ON DELETE CASCADE,
            subject_code  VARCHAR(40)  NOT NULL,
            subject_name  VARCHAR(200) NOT NULL,
            is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
            created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
            -- unique PER COURSE: the same subject taught to 99B and 100B is two
            -- independent rows, which is what lets an empty course be filled in
            -- later with no code changes.
            CONSTRAINT uq_subjects_course_code UNIQUE (course_id, subject_code),
            -- target for the composite FK from instructor_assignments
            CONSTRAINT uq_subjects_id_course   UNIQUE (id, course_id)
        );
    """))
    # asyncpg rejects multiple commands in one prepared statement, so every
    # statement in this script is executed on its own.
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_subjects_course_id ON subjects (course_id);"
    ))
    ok("subjects")

    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            id                 SERIAL         PRIMARY KEY,
            role               user_role_enum NOT NULL,
            username           VARCHAR(80)    NOT NULL,
            password_hash      VARCHAR(255)   NOT NULL,
            full_name          VARCHAR(200),
            linked_student_id  INTEGER
                               REFERENCES students(id) ON DELETE SET NULL,
            is_active          BOOLEAN        NOT NULL DEFAULT TRUE,
            last_login_at      TIMESTAMPTZ,
            created_at         TIMESTAMPTZ    NOT NULL DEFAULT now(),
            updated_at         TIMESTAMPTZ    NOT NULL DEFAULT now(),
            CONSTRAINT uq_users_username   UNIQUE (username),
            CONSTRAINT uq_users_linked_stu UNIQUE (linked_student_id),
            -- Closed form over the complement, not a list of non-student roles,
            -- so any role added later inherits the restrictive branch.
            CONSTRAINT ck_users_student_link CHECK (
                (role =  'STUDENT' AND linked_student_id IS NOT NULL) OR
                (role <> 'STUDENT' AND linked_student_id IS NULL)
            )
        );
    """))
    ok("users")

    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS instructor_assignments (
            id          SERIAL      PRIMARY KEY,
            user_id     INTEGER     NOT NULL
                        REFERENCES users(id) ON DELETE CASCADE,
            course_id   INTEGER     NOT NULL
                        REFERENCES courses(id) ON DELETE CASCADE,
            subject_id  INTEGER     NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_instr_assign UNIQUE (user_id, subject_id),
            -- Composite FK: course_id is pinned to the subject's own course, so
            -- a row cannot claim a course its subject does not belong to.
            CONSTRAINT fk_instr_assign_subject
                FOREIGN KEY (subject_id, course_id)
                REFERENCES subjects (id, course_id) ON DELETE CASCADE
        );
    """))
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_instr_assign_user "
        "ON instructor_assignments (user_id);"
    ))
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_instr_assign_subject "
        "ON instructor_assignments (subject_id);"
    ))
    ok("instructor_assignments")
    warn("KNOWN PHASE 1 GAP (accepted): instructor_assignments.user_id accepts any")
    warn("role. Instructor-only integrity is enforced in app code until Phase 2.")


# ── Step 3 – new columns, nullable for now ────────────────────────────────────

async def add_columns(conn) -> None:
    print(f"\n{_B}[3/8] Adding columns to students / sessions …{_RS}")

    # Added nullable so existing rows survive; tightened in step 7 after backfill.
    await conn.execute(text(
        "ALTER TABLE students ADD COLUMN IF NOT EXISTS course_id INTEGER;"
    ))
    await conn.execute(text("""
        DO $$ BEGIN
            ALTER TABLE students ADD CONSTRAINT students_course_id_fkey
                FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE RESTRICT;
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """))
    ok("students.course_id  (nullable, FK → courses ON DELETE RESTRICT)")

    await conn.execute(text(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS subject_id INTEGER;"
    ))
    await conn.execute(text("""
        DO $$ BEGIN
            ALTER TABLE sessions ADD CONSTRAINT sessions_subject_id_fkey
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE RESTRICT;
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """))
    ok("sessions.subject_id  (nullable, FK → subjects ON DELETE RESTRICT)")
    info("sessions.subject (free-text) retained and deprecated - not dropped")
    info("students.classroom_id retained - physical room, orthogonal to course")


# ── Step 4 – seed courses ─────────────────────────────────────────────────────

async def seed_courses(conn) -> None:
    print(f"\n{_B}[4/8] Seeding courses …{_RS}")
    for code, name in COURSES:
        await conn.execute(text("""
            INSERT INTO courses (code, name, is_active)
            VALUES (:code, :name, TRUE)
            ON CONFLICT (code) DO NOTHING
        """), {"code": code, "name": name})
    rows = (await conn.execute(text(
        "SELECT code, name FROM courses ORDER BY id"
    ))).all()
    for code, name in rows:
        marker = "real roster" if code == LEGACY_COURSE_CODE else "empty, selectable"
        ok(f"{code:<5} {name:<22} ({marker})")


# ── Step 5 – legacy history subject ───────────────────────────────────────────

async def seed_legacy_subject(conn) -> None:
    print(f"\n{_B}[5/8] Creating legacy-history subject …{_RS}")
    await conn.execute(text("""
        INSERT INTO subjects (course_id, subject_code, subject_name, is_active)
        SELECT id, :scode, :sname, FALSE FROM courses WHERE code = :ccode
        ON CONFLICT (course_id, subject_code) DO NOTHING
    """), {"scode": LEGACY_SUBJECT_CODE, "sname": LEGACY_SUBJECT_NAME,
           "ccode": LEGACY_COURSE_CODE})
    ok(f"{LEGACY_COURSE_CODE} / {LEGACY_SUBJECT_CODE}  is_active=FALSE")
    info("Real subjects are added later via the Admin UI, not seeded here.")
    info("All other courses intentionally have zero subjects.")


# ── Step 6 – backfill ─────────────────────────────────────────────────────────

async def backfill(conn) -> None:
    print(f"\n{_B}[6/8] Backfilling existing data …{_RS}")

    r = await conn.execute(text("""
        UPDATE students SET course_id = (SELECT id FROM courses WHERE code = :ccode)
        WHERE course_id IS NULL AND classroom_id = :cid
    """), {"ccode": LEGACY_COURSE_CODE, "cid": LEGACY_CLASSROOM_ID})
    ok(f"students   → course {LEGACY_COURSE_CODE}: {r.rowcount} row(s) updated")

    r = await conn.execute(text("""
        UPDATE sessions SET subject_id = (
            SELECT s.id FROM subjects s JOIN courses c ON c.id = s.course_id
            WHERE c.code = :ccode AND s.subject_code = :scode
        )
        WHERE subject_id IS NULL
    """), {"ccode": LEGACY_COURSE_CODE, "scode": LEGACY_SUBJECT_CODE})
    ok(f"sessions   → {LEGACY_COURSE_CODE}/{LEGACY_SUBJECT_CODE}: {r.rowcount} row(s) updated")

    info("attendance_records / behavior_events / face_detections: 0 rows touched")
    info("  (they reference sessions.id and students.id, neither of which changed)")


# ── Step 7 – assert, then tighten ─────────────────────────────────────────────

async def tighten(conn) -> None:
    print(f"\n{_B}[7/8] Verifying backfill, then enforcing NOT NULL …{_RS}")

    # Belt-and-braces: SET NOT NULL would fail anyway, but this names the table
    # and count, and aborts the whole transaction before any constraint churn.
    orphan_students = (await conn.execute(text(
        "SELECT count(*) FROM students WHERE course_id IS NULL"))).scalar()
    orphan_sessions = (await conn.execute(text(
        "SELECT count(*) FROM sessions WHERE subject_id IS NULL"))).scalar()
    if orphan_students or orphan_sessions:
        err(
            f"Backfill incomplete - aborting (transaction rolls back):\n"
            f"      students  with NULL course_id : {orphan_students}\n"
            f"      sessions  with NULL subject_id: {orphan_sessions}"
        )
    ok("no NULL course_id / subject_id remain")

    await conn.execute(text("ALTER TABLE students ALTER COLUMN course_id  SET NOT NULL;"))
    await conn.execute(text("ALTER TABLE sessions ALTER COLUMN subject_id SET NOT NULL;"))
    ok("students.course_id  NOT NULL")
    ok("sessions.subject_id NOT NULL")

    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_students_course_id ON students (course_id);"))
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_sessions_subject_started "
        "ON sessions (subject_id, started_at);"))
    ok("ix_students_course_id, ix_sessions_subject_started")


# ── Step 8 – seed HOD + summary ───────────────────────────────────────────────

async def seed_hod(conn, password: str) -> None:
    print(f"\n{_B}[8/8] Seeding HOD login + verification …{_RS}")

    existing = (await conn.execute(text(
        "SELECT id FROM users WHERE username = :u"), {"u": HOD_USERNAME})).scalar()
    if existing:
        info(f"user {HOD_USERNAME!r} already exists (id={existing}) - password left unchanged")
    else:
        await conn.execute(text("""
            INSERT INTO users (role, username, password_hash, full_name, is_active)
            VALUES ('HOD', :u, :h, :n, TRUE)
        """), {"u": HOD_USERNAME, "h": pwd_context.hash(password), "n": HOD_FULL_NAME})
        ok(f"HOD login {HOD_USERNAME!r} ({HOD_FULL_NAME}) - bcrypt, from HOD_SEED_PASSWORD")
        info("Auth endpoints do not exist yet; this account is ready for Phase 2.")

    print()
    for table in ("courses", "subjects", "users", "instructor_assignments",
                  "classrooms", "students", "sessions",
                  "attendance_records", "behavior_events", "face_detections"):
        n = (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar()
        info(f"{table:<24} {n:>7} rows")

    print()
    rows = (await conn.execute(text("""
        SELECT c.code, count(DISTINCT st.id) AS students, count(DISTINCT se.id) AS sessions
        FROM courses c
        LEFT JOIN students st ON st.course_id = c.id
        LEFT JOIN subjects  su ON su.course_id = c.id
        LEFT JOIN sessions  se ON se.subject_id = su.id
        GROUP BY c.code, c.id ORDER BY c.id
    """))).all()
    info(f"{'course':<8}{'students':>10}{'sessions':>10}")
    for code, nstu, nses in rows:
        info(f"{code:<8}{nstu:>10}{nses:>10}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(dry_run: bool) -> None:
    print(f"\n{_B}{_C}══════════════════════════════════════════════════════{_RS}")
    print(f"{_B}{_C}  Phase 1 – multi-course / multi-subject / four-role   {_RS}")
    if dry_run:
        print(f"{_B}{_Y}  DRY RUN – every step runs, then rolls back           {_RS}")
    print(f"{_B}{_C}══════════════════════════════════════════════════════{_RS}")

    if not await check_connection():
        err(f"Cannot reach database. Check DATABASE_URL in .env\n    {settings.database_url}")

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            try:
                hod_password = await preflight(conn)
                await create_new_tables(conn)
                await add_columns(conn)
                await seed_courses(conn)
                await seed_legacy_subject(conn)
                await backfill(conn)
                await tighten(conn)
                await seed_hod(conn, hod_password)

                if dry_run:
                    await trans.rollback()
                    print(f"\n{_Y}{_B}✓ Dry run complete – all changes rolled back.{_RS}")
                    print(f"  Re-run without --dry-run to apply.\n")
                else:
                    await trans.commit()
                    print(f"\n{_G}{_B}✓ Migration applied.{_RS}")
                    print(f"\nNext:")
                    print(f"  1. {_C}python -m pytest backend/tests -q{_RS}")
                    print(f"  2. Phase 1 remainder: crud.py + stream.py roster-gating fix")
                    print(f"  3. Phase 2: auth endpoints, Admin UI for courses/subjects\n")
            except Exception:
                await trans.rollback()
                raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1 multi-course schema migration")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every step inside a transaction, then roll back")
    asyncio.run(main(ap.parse_args().dry_run))
