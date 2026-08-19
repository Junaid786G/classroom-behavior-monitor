#!/usr/bin/env python3
"""
01_setup_db.py – One-time database initialisation.

Run from the project root:
    python scripts/01_setup_db.py

What it does
────────────
1. Verifies PostgreSQL connectivity via DATABASE_URL in .env
2. Installs the pgvector extension (requires PostgreSQL ≥ 13 + pgvector)
3. Creates all ORM tables (equivalent to the first Alembic migration)
4. Seeds one default classroom and one test classroom
5. Prints a verification summary
"""

import asyncio
import sys
from pathlib import Path

# Allow importing backend package from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import get_settings
from backend.database import Base, check_connection
from backend.models import Classroom, Student    # ensure models are registered

settings = get_settings()

# ── Colours for terminal output ───────────────────────────────────────────────
_G  = "\033[92m"   # green
_R  = "\033[91m"   # red
_Y  = "\033[93m"   # yellow
_C  = "\033[96m"   # cyan
_B  = "\033[1m"    # bold
_RS = "\033[0m"    # reset

def ok(msg):   print(f"  {_G}✓{_RS} {msg}")
def err(msg):  print(f"  {_R}✗{_RS} {msg}"); sys.exit(1)
def warn(msg): print(f"  {_Y}!{_RS} {msg}")
def info(msg): print(f"  {_C}▸{_RS} {msg}")


# ── Step 1 – connectivity check ───────────────────────────────────────────────
async def check_pg() -> None:
    print(f"\n{_B}[1/5] Checking PostgreSQL connection …{_RS}")
    info(f"URL: {settings.database_url[:60]}…")
    reachable = await check_connection()
    if not reachable:
        err(
            f"Cannot reach database.\n"
            f"    Make sure PostgreSQL is running and DATABASE_URL in .env is correct.\n"
            f"    Current URL: {settings.database_url}"
        )
    ok("PostgreSQL is reachable")


# ── Step 2 – pgvector extension ───────────────────────────────────────────────
async def install_pgvector(engine) -> None:
    print(f"\n{_B}[2/5] Installing pgvector extension …{_RS}")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        ok("pgvector extension ready")
    except Exception as exc:
        warn(
            f"pgvector not available ({exc}).\n"
            f"         ARRAY(Float) columns will be used for embeddings instead.\n"
            f"         Install pgvector for native vector similarity search."
        )


# ── Step 3 – create tables ────────────────────────────────────────────────────
async def create_tables(engine) -> None:
    print(f"\n{_B}[3/5] Creating database tables …{_RS}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    tables = list(Base.metadata.tables.keys())
    ok(f"Created / verified {len(tables)} tables:")
    for t in sorted(tables):
        info(f"  {t}")


# ── Step 4 – seed classrooms ──────────────────────────────────────────────────
async def seed_classrooms() -> None:
    print(f"\n{_B}[4/5] Seeding default classrooms …{_RS}")
    from sqlalchemy import select
    from backend.database import AsyncSessionLocal

    classrooms_to_seed = [
        dict(name="Main Classroom",  building="Block A", floor=1, capacity=30, is_active=True),
        dict(name="Lab A",           building="Block B", floor=2, capacity=20, is_active=True),
        dict(name="Lecture Hall 1",  building="Main",    floor=0, capacity=80, is_active=True),
    ]

    async with AsyncSessionLocal() as db:
        created = 0
        for data in classrooms_to_seed:
            result = await db.execute(
                select(Classroom).where(Classroom.name == data["name"])
            )
            existing = result.scalar_one_or_none()
            if existing:
                info(f"Classroom {data['name']!r} already exists (id={existing.id}) – skipped")
            else:
                cls = Classroom(**data)
                db.add(cls)
                await db.flush()
                created += 1
                ok(f"Created classroom {data['name']!r} (id={cls.id})")
        await db.commit()

    if created == 0:
        info("All classrooms already exist")
    else:
        ok(f"Seeded {created} new classroom(s)")


# ── Step 5 – verification summary ────────────────────────────────────────────
async def verify(engine) -> None:
    print(f"\n{_B}[5/5] Verification …{_RS}")
    async with engine.connect() as conn:
        for table in sorted(Base.metadata.tables.keys()):
            result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            info(f"{table:<30} {count:>6} rows")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"\n{_B}{_C}═══════════════════════════════════════════════{_RS}")
    print(f"{_B}{_C}   CLASSROOM MONITOR – Database Setup Script    {_RS}")
    print(f"{_B}{_C}═══════════════════════════════════════════════{_RS}")

    engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        echo=False,
    )

    try:
        await check_pg()
        await install_pgvector(engine)
        await create_tables(engine)
        await seed_classrooms()
        await verify(engine)
    finally:
        await engine.dispose()

    print(f"\n{_G}{_B}✓ Database setup complete!{_RS}")
    print(f"\nNext steps:")
    print(f"  1. Run   {_C}python scripts/02_register_students.py{_RS}   to enroll students")
    print(f"  2. Run   {_C}uvicorn backend.main:app --reload{_RS}         to start the API")
    print(f"  3. Run   {_C}streamlit run frontend/app.py{_RS}             to start the dashboard\n")


if __name__ == "__main__":
    asyncio.run(main())
