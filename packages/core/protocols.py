"""Protocol definitions for SignBridge interfaces.

Protocols define interfaces for duck typing, allowing packages to
depend on behaviors rather than concrete implementations.

Usage:
    from packages.core import SignLookup

    def translate(text: str, store: SignLookup) -> list[str]:
        # store.get_sign() and get_verified_sign() are available
        ...
"""

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from .types import SignStatus


@runtime_checkable
class SignLookup(Protocol):
    """Protocol for sign database lookup operations.

    Implement this protocol to provide sign lookup capabilities
    to the translation package. This is the minimal interface
    needed for gloss validation.

    Example:
        class MyStore:
            def get_sign(self, gloss: str) -> Optional[Sign]:
                ...
            def get_verified_sign(self, gloss: str) -> Optional[Sign]:
                ...

        # MyStore automatically satisfies SignLookup
        store = MyStore()
        validator = GlossValidator(store)  # type-checks
    """

    def get_sign(self, gloss: str) -> Optional[Any]:
        """Get a sign by gloss from any status.

        Args:
            gloss: The sign gloss (case-insensitive)

        Returns:
            Sign object or None if not found
        """
        ...

    def get_verified_sign(self, gloss: str) -> Optional[Any]:
        """Get a verified sign only.

        Args:
            gloss: The sign gloss (case-insensitive)

        Returns:
            Sign object if verified, None otherwise
        """
        ...


@runtime_checkable
class SignStore(Protocol):
    """Protocol for full sign storage operations.

    Extended interface for managing signs in the database,
    including CRUD operations and verification workflow.
    """

    def get_sign(self, gloss: str) -> Optional[Any]:
        """Get a sign by gloss from any status."""
        ...

    def get_verified_sign(self, gloss: str) -> Optional[Any]:
        """Get a verified sign only."""
        ...

    def add_sign(
        self,
        gloss: str,
        video_path: Path,
        english: Optional[list[str]] = None,
        category: str = "",
        **kwargs: Any,
    ) -> Any:
        """Add a new sign to the database (pending status).

        Args:
            gloss: The sign gloss
            video_path: Path to video file
            english: English translations
            category: Sign category
            **kwargs: Additional metadata

        Returns:
            The created Sign object
        """
        ...

    def verify_sign(
        self,
        gloss: str,
        quality_score: int,
        verified_by: str,
    ) -> Any:
        """Verify a sign and move to verified status.

        Args:
            gloss: The sign gloss
            quality_score: Quality rating (1-5)
            verified_by: Username of verifier

        Returns:
            The updated Sign object
        """
        ...

    def list_signs(self, status: Optional[SignStatus] = None) -> list[Any]:
        """List signs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of Sign objects
        """
        ...

    def list_verified(self) -> list[Any]:
        """List all verified signs.

        Returns:
            List of verified Sign objects
        """
        ...


@runtime_checkable
class VideoLoader(Protocol):
    """Protocol for loading video clips.

    Defines the interface for loading sign video clips,
    used by the video composition system.
    """

    def get_clip(self, gloss: str) -> Any:
        """Load a video clip for a sign.

        Args:
            gloss: The sign gloss

        Returns:
            VideoClip object with frames and metadata

        Raises:
            FileNotFoundError: If sign video not found
        """
        ...

    def list_available(self) -> list[str]:
        """List all available sign glosses.

        Returns:
            List of gloss strings that have videos
        """
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict.

    Implement this to enable JSON serialization of dataclasses.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the object
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        """Create instance from dictionary.

        Args:
            data: Dictionary with object data

        Returns:
            New instance of the class
        """
        ...
