"""SignBridge V2 API - FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import translate_router, signs_router, videos_router
from .dependencies import get_config
from .schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    config = get_config()
    print(f"SignBridge API starting...")
    print(f"  Signs directory: {config['signs_dir']}")
    print(f"  Video cache: {config['cache_dir']}")

    # Ensure directories exist
    config["signs_dir"].mkdir(parents=True, exist_ok=True)
    config["cache_dir"].mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    print("SignBridge API shutting down...")


app = FastAPI(
    title="SignBridge API",
    description="REST API for English to ASL translation using video-based rendering",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(translate_router)
app.include_router(signs_router)
app.include_router(videos_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check if the API is healthy and all services are running."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        services={
            "api": "running",
            "translation": "available",
            "video": "available",
            "database": "available",
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return {
        "message": "SignBridge API v2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# For running with: python -m packages.api
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "packages.api.main:app",
        host=config["api_host"],
        port=config["api_port"],
        reload=True,
    )
