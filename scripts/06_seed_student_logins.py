#!/usr/bin/env python3
"""
06_seed_student_logins.py – give every student on the roster a login.

    python scripts/06_seed_student_logins.py --list      # who has one, who does not
    python scripts/06_seed_student_logins.py --dry-run   # rehearse, then roll back
    python scripts/06_seed_student_logins.py             # apply

The username is the student's own student_code, verbatim: CS-001, CS-002 …
Note that login is a case-sensitive match on users.username
(crud.get_user_by_username), so CS-001 is the only spelling that works -
cs-001 will not.

05_add_user.py stays the tool for ONE account of any role. This is the bulk
counterpart for the one case that is inherently plural: a roster. Splitting
them keeps 05's --username / --role / --student-id validation meaningful
instead of conditional on a mode flag.

Idempotent, and specific about why it skips: a student who already has a login
keeps it untouched - password, username and all - which is what protects an
account created earlier by hand from being duplicated or overwritten here.

ONE SHARED PASSWORD - INTERIM STATE, READ THIS
══════════════════════════════════════════════
Every account this creates gets the SAME password, from STUDENTS_SEED_PASSWORD.
That is a deliberate bootstrap, not a design: with self-service password change
not yet built, the shared secret means any of these students can sign in as any
other by typing a different code. It is acceptable only while the accounts hold
nothing a classmate cannot already see, and it is the reason the Phase 2
change-password endpoint matters more than it looks. Do not extend this pattern
to staff accounts.
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
from backend.utils.security import hash_password

settings = get_settings()

# ── Colours ───────────────────────────────────────────────────────────────────
_G, _R, _Y, _C, _B, _RS = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"

def ok(m):   print(f"  {_G}✓{_RS} {m}")
def err(m):  print(f"  {_R}✗{_RS} {m}"); sys.exit(1)
def warn(m): print(f"  {_Y}!{_RS} {m}")
def info(m): print(f"  {_C}▸{_RS} {m}")

# Deliberately one character from STUDENT_SEED_PASSWORD, which seeds a single
# account via 05. Every message below names the variable it actually read, so
# the two cannot be confused after the fact.
DEFAULT_PASSWORD_ENV = "STUDENTS_SEED_PASSWORD"


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


async def load_roster(conn) -> list:
    """Every student, with the login they already have (if any).

    LEFT JOIN on linked_student_id: uq_users_linked_stu makes that at most one
    row, so this cannot fan out.
    """
    rows = (await conn.execute(text("""
        SELECT s.id, s.student_code, s.full_name, s.is_active,
               u.id AS user_id, u.username AS existing_username
        FROM students s
        LEFT JOIN users u ON u.linked_student_id = s.id
        ORDER BY s.id
    """))).all()
    return [dict(r._mapping) for r in rows]


async def taken_usernames(conn) -> set:
    rows = (await conn.execute(text("SELECT username FROM users"))).all()
    return {r[0] for r in rows}


async def show_roster(conn) -> None:
    roster = await load_roster(conn)
    if not roster:
        info("no students on the roster")
        return
    print()
    for s in roster:
        flag = "" if s["is_active"] else "  (inactive)"
        if s["existing_username"]:
            ok(f"{s['student_code']:<8} {s['full_name']:<12} login: {s['existing_username']}{flag}")
        else:
            warn(f"{s['student_code']:<8} {s['full_name']:<12} no login{flag}")
    print()


async def seed(conn, password: str) -> tuple:
    """Create one login per student that lacks one. Returns (created, skipped)."""
    roster = await load_roster(conn)
    taken = await taken_usernames(conn)
    created, skipped = [], []

    for s in roster:
        code = s["student_code"]

        if s["existing_username"]:
            # The binding is linked_student_id, not the username: a student who
            # was given a login by hand keeps it, whatever it is called.
            skipped.append((code, f"already has login {s['existing_username']!r}"))
            continue

        if not s["is_active"]:
            skipped.append((code, "student is inactive"))
            continue

        if code in taken:
            # Not reachable on today's data. Left explicit so a code that
            # collides with a staff username skips one student instead of
            # aborting the roster.
            skipped.append((code, f"username {code!r} already belongs to another account"))
            continue

        await conn.execute(text("""
            INSERT INTO users (role, username, password_hash, full_name,
                               linked_student_id, is_active)
            VALUES ('STUDENT', :u, :h, :n, :sid, TRUE)
        """), {
            "u": code,
            # Hashed per account: bcrypt salts each one, so 14 identical
            # passwords produce 14 different hashes and the column never
            # reveals that they match.
            "h": hash_password(password),
            "n": s["full_name"],
            "sid": s["id"],
        })
        taken.add(code)
        created.append((code, s["full_name"]))

    return created, skipped


async def main(args) -> None:
    print(f"\n{_B}{_C}════════════════════════════════════════{_RS}")
    print(f"{_B}{_C}   Student logins from the roster        {_RS}")
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
                    await show_roster(conn)
                    await trans.rollback()
                    return

                password = read_password(args.password_env)
                created, skipped = await seed(conn, password)

                print()
                for code, name in created:
                    ok(f"{code:<8} created for {name}")
                for code, why in skipped:
                    warn(f"{code:<8} skipped - {why}")

                print()
                info(f"{len(created)} created, {len(skipped)} skipped")

                if args.dry_run:
                    await trans.rollback()
                    print(f"\n{_Y}{_B}✓ Dry run complete - all changes rolled back.{_RS}")
                    print(f"  Re-run without --dry-run to apply.\n")
                else:
                    await trans.commit()
                    print(f"\n{_G}{_B}✓ Applied.{_RS}")
                    if created:
                        print(
                            f"\n  {_Y}All {len(created)} accounts share one password, from "
                            f"{args.password_env}.{_RS}\n"
                            f"  Self-service password change does not exist yet, so any of\n"
                            f"  these students can sign in as another by typing a different\n"
                            f"  code. Interim state - see this script's docstring.\n"
                        )
            except Exception:
                await trans.rollback()
                raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Create a STUDENT login for every student on the roster, "
                    "username = student_code")
    ap.add_argument("--list", action="store_true",
                    help="show every student and whether they have a login, then exit")
    ap.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV, metavar="NAME",
                    help=f"env var holding the shared password (default "
                         f"{DEFAULT_PASSWORD_ENV}); prompted for if unset. Never "
                         f"passed on the command line.")
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything inside a transaction, then roll back")
    asyncio.run(main(ap.parse_args()))
