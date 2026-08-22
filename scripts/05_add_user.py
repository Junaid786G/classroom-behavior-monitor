#!/usr/bin/env python3
"""
05_add_user.py – create one login, and optionally assign an instructor to
course+subject pairs.

    python scripts/05_add_user.py --list
    python scripts/05_add_user.py --username ahmed --role instructor \
           --full-name "Dr. Ahmed" --assign 99B:AV-423 --assign 99B:AV-482
    python scripts/05_add_user.py --username ahmed --assign 99B:IE-413   # top up

Interim tool. Creating instructor accounts and assigning them to course+
subject pairs is a TRAINING_CONTROL job that moves into the Phase 2 Admin UI;
this stays useful afterwards for scripted / bulk setup and for bootstrapping
the first account, which no UI can do for itself.

THE PASSWORD NEVER COMES FROM THE COMMAND LINE
──────────────────────────────────────────────
argv is visible in shell history and to every process on the box via `ps`.
So the password is read from an environment variable (default
USER_SEED_PASSWORD, see --password-env), exactly as 03_migrate_multicourse.py
reads HOD_SEED_PASSWORD; failing that, it is prompted for twice, unechoed.

Idempotent: an existing username is left completely alone - password, role and
full name unchanged - but any missing --assign pairs are still added, so this
can be re-run to top up an account's assignments. --dry-run runs everything
inside a transaction and rolls it back.
"""

import argparse
import asyncio
import os
import sys
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import get_settings
from backend.database import check_connection
from backend.models import UserRole
# The one supported way to turn a password into a users.password_hash value:
# see the module docstring in backend/utils/security.py. 03_migrate_multicourse
# predates it and calls pwd_context.hash directly; new code should not.
from backend.utils.security import hash_password

settings = get_settings()

# ── Colours ───────────────────────────────────────────────────────────────────
_G, _R, _Y, _C, _B, _RS = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"

def ok(m):   print(f"  {_G}✓{_RS} {m}")
def err(m):  print(f"  {_R}✗{_RS} {m}"); sys.exit(1)
def warn(m): print(f"  {_Y}!{_RS} {m}")
def info(m): print(f"  {_C}▸{_RS} {m}")

DEFAULT_PASSWORD_ENV = "USER_SEED_PASSWORD"

# Reserved by 03_migrate_multicourse for the pre-migration history bucket. It is
# is_active=FALSE and must never be assigned to anyone. Same list as 04.
RESERVED_SUBJECT_CODES = {"LEGACY-CS"}

# The API speaks these; the DB stores the enum *name* (uppercase). Sourced from
# UserRole so a role added later shows up here without editing this script.
ROLE_VALUES = [r.value for r in UserRole]


# ── Reads ─────────────────────────────────────────────────────────────────────

async def show_users(conn) -> None:
    """Every login with its assignments, so --list is enough to see the state."""
    rows = (await conn.execute(text("""
        SELECT u.id, u.role, u.username, u.full_name, u.is_active,
               u.linked_student_id, c.code, s.subject_code
        FROM users u
        LEFT JOIN instructor_assignments a ON a.user_id = u.id
        LEFT JOIN courses  c ON c.id = a.course_id
        LEFT JOIN subjects s ON s.id = a.subject_id
        ORDER BY u.id, c.code, s.subject_code
    """))).all()

    if not rows:
        info("no users yet")
        return

    current = None
    for uid, role, username, full_name, active, student_id, ccode, scode in rows:
        if uid != current:
            current = uid
            flag = "" if active else "  (inactive)"
            link = f"  student_id={student_id}" if student_id else ""
            print(f"\n  {_B}{username}{_RS}  id={uid}  {role}  "
                  f"{full_name or '—'}{link}{flag}")
        if ccode:
            ok(f"{ccode} / {scode}")
    print()


async def _resolve_pair(conn, course_code: str, subject_code: str):
    """(course_id, subject_id) for a COURSE:SUBJECT pair, or exit with why not."""
    course = (await conn.execute(text(
        "SELECT id FROM courses WHERE code = :c"), {"c": course_code})).first()
    if course is None:
        valid = (await conn.execute(text(
            "SELECT string_agg(code, ', ' ORDER BY id) FROM courses"))).scalar()
        err(f"No course with code {course_code!r}.\n    Valid codes: {valid}")

    subject = (await conn.execute(text("""
        SELECT id, subject_name, is_active FROM subjects
        WHERE course_id = :cid AND subject_code = :s
    """), {"cid": course[0], "s": subject_code})).first()
    if subject is None:
        valid = (await conn.execute(text("""
            SELECT string_agg(subject_code, ', ' ORDER BY subject_code)
            FROM subjects WHERE course_id = :cid AND is_active
        """), {"cid": course[0]})).scalar()
        err(f"Course {course_code} has no subject {subject_code!r}.\n"
            f"    Active subjects: {valid or '(none)'}")

    if not subject[2]:
        err(f"{course_code} / {subject_code} is archived (is_active=FALSE) and "
            f"cannot be assigned.\n    Archived subjects stay queryable as "
            f"history but are never selectable.")

    return course[0], subject[0], subject[1]


# ── Writes ────────────────────────────────────────────────────────────────────

async def create_user(conn, username, role, full_name, password, student_id):
    """Insert the login, or report the existing one. Returns (user_id, created)."""
    existing = (await conn.execute(text(
        "SELECT id, role FROM users WHERE username = :u"), {"u": username})).first()
    if existing:
        warn(f"user {username!r} already exists (id={existing[0]}, "
             f"role={existing[1]}) - password, role and name left unchanged")
        return existing[0], False

    user_id = (await conn.execute(text("""
        INSERT INTO users (role, username, password_hash, full_name,
                           linked_student_id, is_active)
        VALUES (:r, :u, :h, :n, :sid, TRUE)
        RETURNING id
    """), {
        "r": role.name,          # DB stores the enum NAME, e.g. 'INSTRUCTOR'
        "u": username,
        "h": hash_password(password),
        "n": full_name,
        "sid": student_id,
    })).scalar()

    ok(f"Created {role.value} login {username!r} ({full_name or '—'})  (id={user_id})")
    info("bcrypt via backend.utils.security.hash_password")
    return user_id, True


async def assign(conn, user_id: int, username: str, pairs) -> None:
    """Add each course+subject pair the user does not already hold."""
    for course_code, subject_code in pairs:
        course_id, subject_id, subject_name = await _resolve_pair(
            conn, course_code, subject_code)

        new_id = (await conn.execute(text("""
            INSERT INTO instructor_assignments (user_id, course_id, subject_id)
            VALUES (:u, :c, :s)
            ON CONFLICT ON CONSTRAINT uq_instr_assign DO NOTHING
            RETURNING id
        """), {"u": user_id, "c": course_id, "s": subject_id})).scalar()

        if new_id is None:
            warn(f"{username} already assigned to {course_code} / {subject_code}"
                 f" - nothing changed")
        else:
            ok(f"Assigned {username} → {course_code} / {subject_code} — {subject_name}")


# ── Input handling ────────────────────────────────────────────────────────────

def parse_pairs(raw_pairs):
    """['99B:AV-423'] -> [('99B', 'AV-423')], normalised and validated."""
    pairs = []
    for raw in raw_pairs or []:
        if raw.count(":") != 1:
            err(f"--assign {raw!r} is not COURSE:SUBJECT, e.g. 99B:AV-423")
        course_code, subject_code = (p.strip().upper() for p in raw.split(":"))
        if not course_code or not subject_code:
            err(f"--assign {raw!r} is missing the course or the subject half")
        if subject_code in RESERVED_SUBJECT_CODES:
            err(f"{subject_code!r} is the reserved pre-migration history bucket "
                f"and must never be assigned.")
        pairs.append((course_code, subject_code))
    return pairs


def read_password(env_name: str) -> str:
    """From the environment, else prompted twice. Never from argv."""
    password = os.getenv(env_name, "").strip()
    if password:
        info(f"password read from {env_name}")
        return password

    if not sys.stdin.isatty():
        err(f"{env_name} is not set and there is no terminal to prompt on.\n"
            f"    Set it in .env (or use --password-env NAME), then re-run.")

    info(f"{env_name} is not set - prompting instead")
    first = getpass("  Password: ")
    if not first.strip():
        err("Empty password.")
    if first != getpass("  Repeat:   "):
        err("The two passwords do not match.")
    return first


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args) -> None:
    print(f"\n{_B}{_C}════════════════════════════════════════{_RS}")
    print(f"{_B}{_C}   Users / instructor assignments        {_RS}")
    if args.dry_run:
        print(f"{_B}{_Y}   DRY RUN - everything rolls back        {_RS}")
    print(f"{_B}{_C}════════════════════════════════════════{_RS}")

    if not await check_connection():
        err(f"Cannot reach database. Check DATABASE_URL in .env\n    {settings.database_url}")

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            try:
                if args.list:
                    await show_users(conn)
                    await trans.rollback()
                    return

                username = args.username.strip()
                pairs = parse_pairs(args.assign)

                existing_role = (await conn.execute(text(
                    "SELECT role FROM users WHERE username = :u"),
                    {"u": username})).scalar()

                if existing_role:
                    # Top-up run: --role is not needed, and the stored role is
                    # what the instructor-only check below must judge.
                    role = UserRole[existing_role]
                    user_id, _ = await create_user(
                        conn, username, role, None, None, None)
                else:
                    if not args.role:
                        err(f"user {username!r} does not exist yet, so --role is "
                            f"required.\n    One of: {', '.join(ROLE_VALUES)}")
                    role = UserRole(args.role)
                    # ck_users_student_link: a STUDENT row must point at a
                    # roster row, and no other role may. Checked here so the
                    # failure is a sentence rather than a constraint traceback.
                    if role is UserRole.STUDENT and args.student_id is None:
                        err("role 'student' requires --student-id (the students.id "
                            "row this login shows records for).")
                    if role is not UserRole.STUDENT and args.student_id is not None:
                        err(f"--student-id is only valid for role 'student', "
                            f"not {role.value!r}.")
                    user_id, _ = await create_user(
                        conn, username, role, args.full_name,
                        read_password(args.password_env), args.student_id)

                if pairs:
                    # models.py:348 - nothing at the schema level stops a
                    # non-instructor being assigned here; that is a known
                    # Phase 1 gap left to application code, which is this.
                    if role is not UserRole.INSTRUCTOR:
                        err(f"instructor_assignments are for INSTRUCTOR logins; "
                            f"{username!r} is {role.value}.")
                    await assign(conn, user_id, username, pairs)

                print()
                await show_users(conn)

                if args.dry_run:
                    await trans.rollback()
                    print(f"{_Y}{_B}✓ Dry run complete - all changes rolled back.{_RS}")
                    print(f"  Re-run without --dry-run to apply.\n")
                else:
                    await trans.commit()
                    print(f"{_G}{_B}✓ Applied.{_RS}\n")
            except Exception:
                await trans.rollback()
                raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Create a login, and optionally assign an instructor to "
                    "course+subject pairs")
    ap.add_argument("--list", action="store_true",
                    help="show every user with their assignments, then exit")
    ap.add_argument("--username", help="login name, e.g. ahmed")
    ap.add_argument("--role", choices=ROLE_VALUES,
                    help="required when creating; omit to top up an existing user")
    ap.add_argument("--full-name", help='display name, e.g. "Dr. Ahmed"')
    ap.add_argument("--student-id", type=int,
                    help="students.id to link - required for role 'student', "
                         "rejected for every other role")
    ap.add_argument("--assign", action="append", metavar="COURSE:SUBJECT",
                    help="course+subject pair for an instructor, e.g. 99B:AV-423 "
                         "(repeatable)")
    ap.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV, metavar="NAME",
                    help=f"env var holding the password (default {DEFAULT_PASSWORD_ENV}); "
                         f"prompted for if unset. Never passed on the command line.")
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything inside a transaction, then roll back")
    parsed = ap.parse_args()

    if not parsed.list and not parsed.username:
        ap.error("--username is required (or use --list)")

    asyncio.run(main(parsed))
