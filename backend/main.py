from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import __version__
from backend.config import get_settings
from backend.database import check_connection, db_lifespan
from backend.gallery.manager import get_gallery, rebuild_gallery_from_db
from backend.pipeline.behavior import get_behavior_analyzer
from backend.pipeline.detector import get_detector
from backend.pipeline.embedder import get_embedder
from backend.pipeline.tracker import get_tracker
from backend.routers import analytics, attendance, auth, catalog, stream, students
from backend.utils.gpu import get_device_info
from backend.utils.logger import setup_logging

settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    setup_logging(log_dir=str(settings.logs_path))
    logger = logging.getLogger("classroom_monitor.main")

    logger.info("Classroom Monitor v%s starting …", __version__)

    # 1. Verify DB connectivity
    async with db_lifespan():

        # 2. Warm up ML models (loads ONNX sessions + allocates GPU memory)
        logger.info("Warming up ML models …")
        try:
            get_detector().warmup()
            logger.info("  ✓ FaceDetector (RetinaFace)")
        except Exception as exc:
            logger.warning("  ✗ FaceDetector failed to warm up: %s", exc)

        try:
            get_embedder().warmup()
            logger.info("  ✓ ArcFaceEmbedder")
        except Exception as exc:
            logger.warning("  ✗ ArcFaceEmbedder failed to warm up: %s", exc)

        try:
            get_tracker()._ensure_loaded()
            logger.info("  ✓ ByteTracker")
        except Exception as exc:
            logger.warning("  ✗ ByteTracker failed to warm up: %s", exc)

        try:
            get_behavior_analyzer().warmup()
            logger.info("  ✓ BehaviorAnalyzer (MediaPipe)")
        except Exception as exc:
            logger.warning("  ✗ BehaviorAnalyzer failed to warm up: %s", exc)

        # 3. Load or rebuild face gallery
        gallery = await get_gallery()
        if settings.gallery_rebuild_on_startup or gallery.size == 0:
            logger.info("Rebuilding FAISS gallery from PostgreSQL …")
            from backend.database import AsyncSessionLocal
            async with AsyncSessionLocal() as db:
                await rebuild_gallery_from_db(db)
        else:
            logger.info("Gallery loaded: %d vectors", gallery.size)

        # 4. Log GPU info
        device_info = get_device_info()
        if device_info["cuda_available"]:
            for d in device_info.get("gpu_devices", []):
                logger.info(
                    "GPU[%d] %s – %d MB free / %d MB total",
                    d["index"], d["name"], d["memory_free_mb"], d["memory_total_mb"],
                )
        else:
            logger.info("Running on CPU (no CUDA available)")

        logger.info("Startup complete – listening on %s:%d", settings.api_host, settings.api_port)
        yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down – saving gallery …")
    gallery = await get_gallery()
    gallery.save()
    logger.info("Bye.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Classroom CCTV Monitor",
    description=(
        "AI-powered classroom attendance & behavioural analysis system. "
        "Stack: FastAPI · PostgreSQL · InsightFace (RetinaFace + ArcFace) "
        "· ByteTrack · MediaPipe FaceMesh · FAISS."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger = logging.getLogger("classroom_monitor.main")
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth.router,       prefix="/api/v1")
app.include_router(stream.router,     prefix="/api/v1")
app.include_router(catalog.router,    prefix="/api/v1")
app.include_router(students.router,   prefix="/api/v1")
app.include_router(attendance.router, prefix="/api/v1")
app.include_router(analytics.router,  prefix="/api/v1")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    from backend.schemas import HealthResponse
    from backend.utils.gpu import cuda_available

    db_ok = await check_connection()
    gallery = await get_gallery()

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db=db_ok,
        gallery_size=gallery.size,
        gpu_available=cuda_available(),
        version=__version__,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"msg": "Classroom Monitor API", "version": __version__, "docs": "/docs"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_config=None,   # we configure logging ourselves in lifespan
    )
