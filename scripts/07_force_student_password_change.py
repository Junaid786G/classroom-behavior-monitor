#!/usr/bin/env python3
"""
07_force_student_password_change.py – add users.must_change_password and set it
on every existing STUDENT account.

    python scripts/07_force_student_password_change.py --dry-run   # rehearse
    python scripts/07_force_student_password_change.py             # apply

Why this exists
───────────────
06_seed_student_logins.py gave every student on the roster a login, all sharing
one password from STUDENTS_SEED_PASSWORD. Any of them can therefore sign in as
any other by typing a different code. This closes that: a flagged account is
refused by every gated route until its owner sets a password of their own, and
POST /auth/me/password clears the flag as it writes the new hash.

hod and ahmed are deliberately NOT flagged. They are single-owner accounts
whose passwords were never shared, so forcing a change would be ceremony
without a threat behind it.

THE BACKFILL RUNS ONCE, AND ONLY ONCE
═════════════════════════════════════
The column is added IF NOT EXISTS, so the DDL is safely re-runnable - but the
UPDATE is not, and the script does not pretend otherwise. It checks
information_schema BEFORE the ALTER, and backfills only when the column did not
already exist. Without that guard a second run would re-flag every student who
had already chosen their own password, turning a one-time migration into a
recurring lockout. Re-running after a successful apply reports "already
migrated" and changes nothing.

Postgres has transactional DDL, so a failure anywhere rolls back the column and
the backfill together. --dry-run does every step and then deliberately rolls
back.
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

# The role whose accounts hold a shared, someone-else-chose-it password.
TARGET_ROLE = "STUDENT"


async def column_exists(conn) -> bool:
    return bool((await conn.execute(text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'must_change_password'
    """))).scalar())


async def report(conn) -> None:
    rows = (await conn.execute(text("""
        SELECT role, must_change_password, count(*) AS n
        FROM users GROUP BY role, must_change_password ORDER BY role
    """))).all()
    for role, flagged, n in rows:
        line = f"{role:<16} {n:>3} account(s)  must_change_password={flagged}"
        (warn if flagged else ok)(line)


async def main(dry_run: bool) -> None:
    print(f"\n{_B}{_C}══════════════════════════════════════════════════{_RS}")
    print(f"{_B}{_C}  Force password change on seeded student logins   {_RS}")
    if dry_run:
        print(f"{_B}{_Y}  DRY RUN - every step runs, then rolls back       {_RS}")
    print(f"{_B}{_C}══════════════════════════════════════════════════{_RS}")

    if not await check_connection():
        err(f"Cannot reach database. Check DATABASE_URL in .env\n    {settings.database_url}")

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            try:
                print(f"\n{_B}[1/3] Column{_RS}")
                existed = await column_exists(conn)
                if existed:
                    warn("users.must_change_password already exists - "
                         "this database has been migrated before")
                else:
                    info("users.must_change_password does not exist yet")

                await conn.execute(text("""
                    ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS must_change_password
                    BOOLEAN NOT NULL DEFAULT FALSE
                """))
                ok("ALTER TABLE users ADD COLUMN IF NOT EXISTS must_change_password")

                print(f"\n{_B}[2/3] Backfill{_RS}")
                if existed:
                    warn("skipped - the column predates this run, so the accounts "
                         "flagged now are whatever the last run left")
                    info("re-flagging here would lock out every student who has "
                         "since chosen their own password")
                else:
                    n = (await conn.execute(text("""
                        UPDATE users SET must_change_password = TRUE
                        WHERE role = :role
                        RETURNING id
                    """), {"role": TARGET_ROLE})).rowcount
                    ok(f"{n} {TARGET_ROLE} account(s) flagged")
                    info("HOD and INSTRUCTOR left alone - single-owner accounts, "
                         "no shared secret")

                print(f"\n{_B}[3/3] State{_RS}")
                await report(conn)

                if dry_run:
                    await trans.rollback()
                    print(f"\n{_Y}{_B}✓ Dry run complete - column and backfill rolled back.{_RS}")
                    print(f"  Re-run without --dry-run to apply.\n")
                else:
                    await trans.commit()
                    print(f"\n{_G}{_B}✓ Migration applied.{_RS}")
                    if not existed:
                        print(
                            f"\n  {_Y}STUDENTS_SEED_PASSWORD is now a first-login "
                            f"credential only.{_RS}\n"
                            f"  A flagged student can reach GET /auth/me and\n"
                            f"  POST /auth/me/password and nothing else until they\n"
                            f"  set their own password.\n"
                        )
            except Exception:
                await trans.rollback()
                raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add users.must_change_password and flag existing student logins")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every step inside a transaction, then roll back")
    asyncio.run(main(ap.parse_args().dry_run))
