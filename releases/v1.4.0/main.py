"""
Controller v1.4.0 - Main Application
Session-aware, route-aware, identity-preserving hybrid controller.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controller.core.config_v130 import settings
from controller.api.routes_v130 import router as v130_router, init_controller_v130

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Session store: {settings.SESSION_STORE_TYPE}")
    logger.info(f"Safe routing: {settings.SAFE_ROUTING_ENABLED}")
    logger.info(f"Context preservation: {settings.CONTEXT_PRESERVATION_ENABLED}")
    
    # Initialize components
    init_controller_v130(settings)
    
    logger.info("Controller v1.4.0 ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Controller v1.4.0")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Session-aware LLM controller with identity preservation and safe routing",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include backward-compatible routes backed by v1.4 core
app.include_router(v130_router, prefix="")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "features": settings.get_feature_flags(),
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG,
    )
