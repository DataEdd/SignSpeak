"""Sign storage with CRUD operations and verification workflow."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Sign, SignStatus, VideoInfo


class SignStore:
    """Manages sign video database with verification workflow."""

    def __init__(self, base_path: str | Path):
        """Initialize store with base path to signs directory."""
        self.base_path = Path(base_path)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for status in ["verified", "pending", "imported", "rejected"]:
            (self.base_path / status).mkdir(parents=True, exist_ok=True)

    def _status_to_dir(self, status: SignStatus) -> Path:
        """Map status to directory."""
        return self.base_path / status.value

    def _load_sign(self, sign_dir: Path, status: SignStatus) -> Optional[Sign]:
        """Load a sign from its directory."""
        metadata_path = sign_dir / "sign.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return Sign.from_dict(data, status, sign_dir)

    def _save_sign(self, sign: Sign) -> None:
        """Save sign metadata to disk."""
        if sign.path is None:
            raise ValueError("Sign has no path set")
        sign.path.mkdir(parents=True, exist_ok=True)
        metadata_path = sign.path / "sign.json"
        with open(metadata_path, "w") as f:
            json.dump(sign.to_dict(), f, indent=2)

    def add_sign(
        self,
        gloss: str,
        video_path: str | Path,
        english: Optional[list[str]] = None,
        category: str = "",
        source: str = "recorded",
        metadata: Optional[dict] = None,
    ) -> Sign:
        """Add a new sign to pending.

        Args:
            gloss: The sign gloss (e.g., "HELLO")
            video_path: Path to source video file
            english: List of English translations
            category: Sign category (e.g., "greeting")
            source: Source of the sign (e.g., "recorded", "wlasl")
            metadata: Additional metadata dict

        Returns:
            The created Sign object
        """
        gloss = gloss.upper()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check if already exists in any status
        existing = self.get_sign(gloss)
        if existing:
            raise ValueError(f"Sign '{gloss}' already exists in {existing.status.value}")

        # Create sign directory in pending
        sign_dir = self.base_path / "pending" / gloss
        sign_dir.mkdir(parents=True, exist_ok=True)

        # Copy video file
        video_dest = sign_dir / "video.mp4"
        shutil.copy2(video_path, video_dest)

        # Build sign object
        meta = metadata or {}
        sign = Sign(
            gloss=gloss,
            english=english or meta.get("english", []),
            category=category or meta.get("category", ""),
            source=source,
            status=SignStatus.PENDING,
            video=VideoInfo(
                file="video.mp4",
                fps=meta.get("video", {}).get("fps", 30),
                duration_ms=meta.get("video", {}).get("duration_ms", 0),
                resolution=meta.get("video", {}).get("resolution", ""),
            ),
            path=sign_dir,
        )

        self._save_sign(sign)
        return sign

    def get_sign(self, gloss: str) -> Optional[Sign]:
        """Get a sign by gloss from any status directory.

        Args:
            gloss: The sign gloss to look up

        Returns:
            Sign if found, None otherwise
        """
        gloss = gloss.upper()

        # Search in priority order: verified, pending, imported, rejected
        for status in [SignStatus.VERIFIED, SignStatus.PENDING, SignStatus.IMPORTED, SignStatus.REJECTED]:
            sign_dir = self._status_to_dir(status) / gloss
            if sign_dir.exists():
                return self._load_sign(sign_dir, status)

        # Check imported subdirectories (e.g., imported/wlasl/HELLO)
        imported_dir = self.base_path / "imported"
        if imported_dir.exists():
            for source_dir in imported_dir.iterdir():
                if source_dir.is_dir() and source_dir.name != ".gitkeep":
                    sign_dir = source_dir / gloss
                    if sign_dir.exists():
                        return self._load_sign(sign_dir, SignStatus.IMPORTED)

        return None

    def get_verified_sign(self, gloss: str) -> Optional[Sign]:
        """Get a verified sign only.

        Args:
            gloss: The sign gloss to look up

        Returns:
            Sign if found and verified, None otherwise
        """
        gloss = gloss.upper()
        sign_dir = self.base_path / "verified" / gloss
        if sign_dir.exists():
            return self._load_sign(sign_dir, SignStatus.VERIFIED)
        return None

    def verify_sign(
        self,
        gloss: str,
        quality_score: int,
        verified_by: str,
    ) -> Sign:
        """Verify a sign and move to verified directory.

        Args:
            gloss: The sign gloss to verify
            quality_score: Quality score 1-5
            verified_by: Username of verifier

        Returns:
            Updated Sign object

        Raises:
            ValueError: If sign not found or score invalid
        """
        if not 1 <= quality_score <= 5:
            raise ValueError("Quality score must be 1-5")

        sign = self.get_sign(gloss)
        if sign is None:
            raise ValueError(f"Sign '{gloss}' not found")

        if sign.status == SignStatus.VERIFIED:
            raise ValueError(f"Sign '{gloss}' is already verified")

        old_path = sign.path

        # Update sign metadata
        sign.quality_score = quality_score
        sign.verified_by = verified_by
        sign.verified_date = datetime.now().strftime("%Y-%m-%d")
        sign.status = SignStatus.VERIFIED

        # Move to verified directory
        new_path = self.base_path / "verified" / sign.gloss
        if new_path.exists():
            shutil.rmtree(new_path)
        shutil.move(str(old_path), str(new_path))
        sign.path = new_path

        self._save_sign(sign)
        return sign

    def reject_sign(self, gloss: str, reason: str = "") -> Sign:
        """Reject a sign and move to rejected directory.

        Args:
            gloss: The sign gloss to reject
            reason: Optional rejection reason

        Returns:
            Updated Sign object
        """
        sign = self.get_sign(gloss)
        if sign is None:
            raise ValueError(f"Sign '{gloss}' not found")

        if sign.status == SignStatus.REJECTED:
            raise ValueError(f"Sign '{gloss}' is already rejected")

        old_path = sign.path
        sign.status = SignStatus.REJECTED

        # Move to rejected directory
        new_path = self.base_path / "rejected" / sign.gloss
        if new_path.exists():
            shutil.rmtree(new_path)
        shutil.move(str(old_path), str(new_path))
        sign.path = new_path

        self._save_sign(sign)
        return sign

    def delete_sign(self, gloss: str) -> bool:
        """Delete a sign completely.

        Args:
            gloss: The sign gloss to delete

        Returns:
            True if deleted, False if not found
        """
        sign = self.get_sign(gloss)
        if sign is None:
            return False

        if sign.path and sign.path.exists():
            shutil.rmtree(sign.path)
        return True

    def list_signs(self, status: Optional[SignStatus] = None) -> list[Sign]:
        """List all signs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of Sign objects
        """
        signs = []

        if status is None:
            statuses = [SignStatus.VERIFIED, SignStatus.PENDING, SignStatus.IMPORTED, SignStatus.REJECTED]
        else:
            statuses = [status]

        for st in statuses:
            status_dir = self._status_to_dir(st)
            if not status_dir.exists():
                continue

            if st == SignStatus.IMPORTED:
                # Handle imported subdirectories
                for source_dir in status_dir.iterdir():
                    if source_dir.is_dir() and source_dir.name != ".gitkeep":
                        for sign_dir in source_dir.iterdir():
                            if sign_dir.is_dir():
                                sign = self._load_sign(sign_dir, st)
                                if sign:
                                    signs.append(sign)
            else:
                for sign_dir in status_dir.iterdir():
                    if sign_dir.is_dir() and sign_dir.name != ".gitkeep":
                        sign = self._load_sign(sign_dir, st)
                        if sign:
                            signs.append(sign)

        return signs

    def list_pending(self) -> list[Sign]:
        """List all pending signs awaiting verification."""
        return self.list_signs(SignStatus.PENDING)

    def list_verified(self) -> list[Sign]:
        """List all verified production-ready signs."""
        return self.list_signs(SignStatus.VERIFIED)

    def count_signs(self) -> dict[str, int]:
        """Get count of signs by status."""
        return {
            "verified": len(self.list_signs(SignStatus.VERIFIED)),
            "pending": len(self.list_signs(SignStatus.PENDING)),
            "imported": len(self.list_signs(SignStatus.IMPORTED)),
            "rejected": len(self.list_signs(SignStatus.REJECTED)),
        }
