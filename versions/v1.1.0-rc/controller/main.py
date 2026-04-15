"""FastAPI application entry point for Controller."""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __description__, __version__
from .api import routes as api_routes
from .core.config import Settings, get_settings

BODY_PREVIEW_LIMIT = 1600
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure structured logging without duplicating handlers across restarts/tests."""
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    os.makedirs(settings.LOGS_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not getattr(root_logger, "_controller_logging_configured", False):
        if settings.LOG_FORMAT == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"name": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler = logging.FileHandler(os.path.join(settings.LOGS_DIR, "controller.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        setattr(root_logger, "_controller_logging_configured", True)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return str(request_id)
    return str(uuid.uuid4())


def _truncate_text(value: str, limit: int = 240) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}...[truncated]"


def _sanitize_value(value: Any, *, key: str | None = None) -> Any:
    if key and key.lower() in SENSITIVE_KEYS:
        return "[REDACTED]"

    if isinstance(value, str):
        return _truncate_text(value)

    if isinstance(value, list):
        items = [_sanitize_value(item) for item in value[:10]]
        if len(value) > 10:
            items.append("...[truncated]")
        return items

    if isinstance(value, dict):
        sanitized = {}
        for index, (child_key, child_value) in enumerate(value.items()):
            if index >= 40:
                sanitized["__truncated__"] = True
                break
            sanitized[child_key] = _sanitize_value(child_value, key=child_key)
        return sanitized

    return value


async def _sanitized_body_preview(request: Request) -> str:
    try:
        raw = await request.body()
    except Exception:
        return "<body-unavailable>"

    if not raw:
        return "<empty-body>"

    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
        preview = json.dumps(_sanitize_value(payload), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        preview = _truncate_text(text, limit=BODY_PREVIEW_LIMIT)

    if len(preview) > BODY_PREVIEW_LIMIT:
        preview = f"{preview[:BODY_PREVIEW_LIMIT]}...[truncated]"
    return preview


def _error_type_for_status(status_code: int) -> str:
    if status_code in {400, 404, 409, 422, 429}:
        return "invalid_request_error"
    if status_code in {401, 403}:
        return "authentication_error"
    if 500 <= status_code < 600:
        return "server_error"
    return "api_error"


def _message_from_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        for key in ("message", "error", "reason"):
            value = detail.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(detail, ensure_ascii=False)
    if isinstance(detail, list):
        return "request validation failed"
    return str(detail)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    del app

    settings = get_settings()
    logger = setup_logging(settings)
    settings.ensure_directories()

    logger.info("=== Controller %s starting ===", __version__)

    try:
        api_routes.init_intelligence(settings)
        logger.info("Intelligence components initialized")
    except Exception as exc:
        logger.error("Failed to initialize intelligence components: %s", exc, exc_info=True)
        raise

    try:
        api_routes.init_providers(settings)
        available = list(api_routes.providers.keys())
        if available:
            logger.info("Provider startup state: active=%s", available)
        else:
            logger.warning("Provider startup state: mock mode (no active providers)")
    except Exception as exc:
        logger.error("Failed to initialize providers: %s", exc, exc_info=True)
        raise

    try:
        with open(settings.PID_FILE, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        logger.info("PID file written: %s", settings.PID_FILE)
    except Exception as exc:
        logger.warning("Could not write PID file: %s", exc)

    logger.info("=== Controller %s ready ===", __version__)
    yield

    logger.info("=== Controller %s shutting down ===", __version__)

    if api_routes.fallback_provider:
        try:
            await api_routes.fallback_provider.close()
            logger.info("Providers closed")
        except Exception as exc:
            logger.error("Error closing providers: %s", exc, exc_info=True)

    try:
        if os.path.exists(settings.PID_FILE):
            os.remove(settings.PID_FILE)
            logger.info("PID file removed")
    except Exception as exc:
        logger.warning("Could not remove PID file: %s", exc)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Controller",
        description=__description__,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = datetime.now(timezone.utc)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            logging.getLogger(__name__).exception(
                "[%s] %s %s -> unhandled exception after %dms",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        response.headers["x-request-id"] = request_id
        logging.getLogger(__name__).info(
            "[%s] %s %s -> %s in %dms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = _request_id(request)
        body_preview = await _sanitized_body_preview(request)
        errors = jsonable_encoder(exc.errors())
        logging.getLogger(__name__).warning(
            "[%s] Request validation failed method=%s path=%s content_type=%s errors=%s body_preview=%s",
            request_id,
            request.method,
            request.url.path,
            request.headers.get("content-type", ""),
            json.dumps(errors, ensure_ascii=False),
            body_preview,
        )

        payload = {
            "error": {
                "message": "request validation failed",
                "type": "invalid_request_error",
                "code": "validation_error",
                "details": errors,
                "request_id": request_id,
            }
        }
        return JSONResponse(
            status_code=422,
            content=payload,
            headers={"x-request-id": request_id},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = _request_id(request)
        logger = logging.getLogger(__name__)
        logger.warning(
            "[%s] HTTPException method=%s path=%s status=%s detail=%s",
            request_id,
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )

        error = {
            "message": _message_from_detail(exc.detail),
            "type": _error_type_for_status(exc.status_code),
            "code": exc.status_code,
            "request_id": request_id,
        }
        if not isinstance(exc.detail, str):
            error["details"] = exc.detail

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error},
            headers={"x-request-id": request_id},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        request_id = _request_id(request)
        logging.getLogger(__name__).error(
            "[%s] Unhandled exception on %s %s: %s",
            request_id,
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "internal server error",
                    "type": "internal_error",
                    "code": 500,
                    "request_id": request_id,
                }
            },
            headers={"x-request-id": request_id},
        )

    app.include_router(api_routes.router)

    @app.get("/")
    async def root():
        return {
            "name": "Controller",
            "version": __version__,
            "description": __description__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


def main() -> None:
    """Main entry point for running the server."""
    import uvicorn

    settings = get_settings()

    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║                    Controller {__version__:<28}║
║                                                              ║
║  OpenClaw-compatible external provider boundary              ║
╠══════════════════════════════════════════════════════════════╣
║  Host: {settings.CONTROLLER_HOST:<22} Port: {settings.CONTROLLER_PORT:<20}║
║  Log Level: {settings.LOG_LEVEL:<17} Docs: http://localhost:{settings.CONTROLLER_PORT}/docs  ║
╚══════════════════════════════════════════════════════════════╝
        """
    )

    uvicorn.run(
        "controller.main:app",
        host=settings.CONTROLLER_HOST,
        port=settings.CONTROLLER_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
