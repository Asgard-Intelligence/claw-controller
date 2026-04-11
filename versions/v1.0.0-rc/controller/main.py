"""
Controller 1.0.0 RC - FastAPI Application Entry Point

Intelligent routing and safety layer for LLM requests.
No-touch OpenClaw principle: Controller operates as external provider only.
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .core.config import Settings, get_settings
from .api.routes import (
    router, init_intelligence, init_providers,
    providers, fallback_provider
)

# Version info
VERSION = "1.0.0-rc"


def setup_logging(settings: Settings):
    """Configure structured logging."""
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    
    # Ensure logs directory exists
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    
    # File handler
    log_file = os.path.join(settings.LOGS_DIR, "controller.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Format
    if settings.LOG_FORMAT == "json":
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    
    # Setup logging
    logger = setup_logging(settings)
    logger.info(f"=== Controller {VERSION} Starting ===")
    
    # Ensure runtime directories
    settings.ensure_directories()
    
    # Initialize intelligence components
    try:
        init_intelligence(settings)
        logger.info("Intelligence components initialized")
    except Exception as e:
        logger.error(f"Failed to initialize intelligence: {e}")
        raise
    
    # Initialize providers
    try:
        init_providers(settings)
        available = list(providers.keys())
        logger.info(f"Providers initialized: {available if available else 'NONE (mock mode)'}")
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")
    
    # Write PID file
    try:
        with open(settings.PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        logger.info(f"PID file written: {settings.PID_FILE}")
    except Exception as e:
        logger.warning(f"Could not write PID file: {e}")
    
    logger.info(f"=== Controller {VERSION} Ready ===")
    
    yield
    
    # Shutdown
    logger.info(f"=== Controller {VERSION} Shutting Down ===")
    
    # Close providers
    if fallback_provider:
        try:
            await fallback_provider.close()
            logger.info("Providers closed")
        except Exception as e:
            logger.error(f"Error closing providers: {e}")
    
    # Remove PID file
    try:
        if os.path.exists(settings.PID_FILE):
            os.remove(settings.PID_FILE)
            logger.info("PID file removed")
    except Exception as e:
        logger.warning(f"Could not remove PID file: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Controller",
        description="Intelligent LLM routing controller with safety and confidence scoring",
        version=VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = datetime.utcnow()
        
        response = await call_next(request)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger = logging.getLogger(__name__)
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
        )
        
        return response
    
    # Error handling
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "internal_error",
                    "code": 500
                }
            }
        )
    
    # Include routes
    app.include_router(router)
    
    return app


# Create app instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Controller",
        "version": VERSION,
        "description": "Intelligent LLM routing controller with safety and confidence scoring",
        "docs": "/docs",
        "health": "/health"
    }


def main():
    """Main entry point for running the server."""
    import uvicorn
    
    settings = get_settings()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Controller {VERSION}                     ║
║                                                              ║
║  Intelligent LLM routing with safety & confidence scoring   ║
╠══════════════════════════════════════════════════════════════╣
║  Host: {settings.CONTROLLER_HOST:<22} Port: {settings.CONTROLLER_PORT:<20}  ║
║  Log Level: {settings.LOG_LEVEL:<17} Docs: http://localhost:{settings.CONTROLLER_PORT}/docs  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "controller.main:app",
        host=settings.CONTROLLER_HOST,
        port=settings.CONTROLLER_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
