"""API route modules."""

from .translate import router as translate_router
from .signs import router as signs_router
from .videos import router as videos_router

__all__ = ["translate_router", "signs_router", "videos_router"]
