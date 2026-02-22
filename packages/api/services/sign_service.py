"""Sign service - handles sign CRUD operations and verification."""

from pathlib import Path
from typing import Optional
import shutil

from packages.database import (
    Sign,
    SignStore,
    SignSearch,
    SignVerifier,
    SignStatus as DBSignStatus,
)


class SignService:
    """Handles sign database operations."""

    def __init__(self, sign_store: SignStore):
        self.store = sign_store
        self.search = SignSearch(sign_store)
        self.verifier = SignVerifier(sign_store)

    def list_signs(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """
        List signs with optional filtering.

        Args:
            status: Filter by status (pending, verified, imported, rejected)
            category: Filter by category
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Dictionary with signs list and total count
        """
        # Get all signs (optionally filtered by status)
        db_status = None
        if status:
            db_status = DBSignStatus(status.upper())

        all_signs = self.store.list_signs(status=db_status)

        # Filter by category if specified
        if category:
            all_signs = [s for s in all_signs if s.category == category]

        total = len(all_signs)

        # Apply pagination
        signs = all_signs[offset:offset + limit]

        return {
            "signs": [self._sign_to_dict(s) for s in signs],
            "total": total,
        }

    def get_sign(self, gloss: str) -> Optional[dict]:
        """Get a single sign by gloss."""
        sign = self.store.get_sign(gloss)
        if sign:
            return self._sign_to_dict(sign)
        return None

    def get_verified_sign(self, gloss: str) -> Optional[dict]:
        """Get a verified sign by gloss."""
        sign = self.store.get_verified_sign(gloss)
        if sign:
            return self._sign_to_dict(sign)
        return None

    def create_sign(
        self,
        gloss: str,
        video_path: Path,
        english: list[str] = None,
        category: str = "",
        source: str = "recorded",
        metadata: dict = None,
    ) -> dict:
        """
        Create a new sign (starts as pending).

        Args:
            gloss: The ASL gloss (e.g., "HELLO")
            video_path: Path to the uploaded video file
            english: English translations
            category: Sign category
            source: Source of the sign
            metadata: Additional metadata

        Returns:
            The created sign as a dictionary
        """
        sign = self.store.add_sign(
            gloss=gloss.upper(),
            video_path=str(video_path),
            english=english or [],
            category=category,
            source=source,
            metadata=metadata,
        )
        return self._sign_to_dict(sign)

    def verify_sign(
        self,
        gloss: str,
        quality_score: int,
        verified_by: str,
    ) -> Optional[dict]:
        """
        Verify a pending sign.

        Args:
            gloss: The sign's gloss
            quality_score: Quality score 1-5
            verified_by: Name of the verifier

        Returns:
            The updated sign, or None if not found
        """
        try:
            sign = self.store.verify_sign(
                gloss=gloss.upper(),
                score=quality_score,
                verified_by=verified_by,
            )
            return self._sign_to_dict(sign)
        except Exception:
            return None

    def reject_sign(self, gloss: str, reason: str = "") -> Optional[dict]:
        """Reject a pending sign."""
        try:
            sign = self.store.move_sign(gloss.upper(), DBSignStatus.REJECTED)
            return self._sign_to_dict(sign)
        except Exception:
            return None

    def delete_sign(self, gloss: str) -> bool:
        """Delete a sign completely."""
        return self.store.delete_sign(gloss.upper())

    def search_signs(
        self,
        query: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Search signs by query string.

        Searches gloss, english translations, and category.
        """
        results = self.search.search(
            query=query,
            status=DBSignStatus(status.upper()) if status else None,
            limit=limit,
        )
        return [self._sign_to_dict(s) for s in results]

    def get_stats(self) -> dict:
        """Get database statistics."""
        all_signs = self.store.list_signs()

        # Count by status
        status_counts = {
            "verified": 0,
            "pending": 0,
            "imported": 0,
            "rejected": 0,
        }

        # Count by category
        categories = {}

        for sign in all_signs:
            status_key = sign.status.value.lower() if hasattr(sign.status, 'value') else str(sign.status).lower()
            if status_key in status_counts:
                status_counts[status_key] += 1

            if sign.category:
                categories[sign.category] = categories.get(sign.category, 0) + 1

        return {
            "total_signs": len(all_signs),
            "verified_signs": status_counts["verified"],
            "pending_signs": status_counts["pending"],
            "imported_signs": status_counts["imported"],
            "rejected_signs": status_counts["rejected"],
            "categories": categories,
        }

    def _sign_to_dict(self, sign: Sign) -> dict:
        """Convert a Sign object to API response format."""
        return {
            "gloss": sign.gloss,
            "english": sign.english,
            "category": sign.category,
            "source": sign.source,
            "status": sign.status.value.lower() if hasattr(sign.status, 'value') else str(sign.status).lower(),
            "quality_score": sign.quality_score,
            "verified_by": sign.verified_by,
            "verified_date": sign.verified_date,
            "video_url": f"/api/signs/{sign.gloss}/video" if sign.video else None,
        }
