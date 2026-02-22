"""Verification workflow for sign quality control."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Sign, SignStatus, QualityScore


@dataclass
class VerificationResult:
    """Result of a verification check."""
    passed: bool
    score: int
    issues: list[str]
    suggestions: list[str]


class SignVerifier:
    """Handles sign verification workflow and quality scoring."""

    def __init__(self, store: "SignStore"):
        """Initialize with a SignStore instance."""
        self.store = store

    def get_next_pending(self) -> Optional[Sign]:
        """Get the next sign awaiting verification.

        Returns oldest pending sign first (FIFO).
        """
        pending = self.store.list_pending()
        if not pending:
            # Also check imported signs
            imported = self.store.list_signs(SignStatus.IMPORTED)
            if imported:
                return imported[0]
            return None
        return pending[0]

    def check_sign_quality(self, sign: Sign) -> VerificationResult:
        """Run automated quality checks on a sign.

        Args:
            sign: Sign to check

        Returns:
            VerificationResult with issues found
        """
        issues = []
        suggestions = []
        score = 5  # Start at perfect, deduct for issues

        # Check required fields
        if not sign.english:
            issues.append("Missing English translations")
            score -= 1

        if not sign.category:
            suggestions.append("Consider adding a category")

        # Check video file exists
        if sign.path:
            video_path = sign.path / (sign.video.file if sign.video else "video.mp4")
            if not video_path.exists():
                issues.append(f"Video file not found: {video_path}")
                score = 1  # Unusable without video

        # Check video metadata
        if sign.video:
            if sign.video.duration_ms == 0:
                suggestions.append("Video duration not set")
            if not sign.video.resolution:
                suggestions.append("Video resolution not set")

        # Check timing
        if sign.timing:
            if sign.timing.sign_start_ms >= sign.timing.sign_end_ms:
                issues.append("Invalid timing: start >= end")
                score -= 1

        # Ensure score is in valid range
        score = max(1, min(5, score))

        return VerificationResult(
            passed=len(issues) == 0 and score >= 3,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    def verify(
        self,
        gloss: str,
        quality_score: int,
        verified_by: str,
        notes: Optional[str] = None,
    ) -> Sign:
        """Verify a sign and move to production.

        Args:
            gloss: Sign gloss to verify
            quality_score: Human-assigned quality score 1-5
            verified_by: Username of verifier
            notes: Optional verification notes

        Returns:
            Verified Sign object

        Raises:
            ValueError: If score is too low or sign not found
        """
        if quality_score < QualityScore.ACCEPTABLE.value:
            raise ValueError(
                f"Score {quality_score} is below minimum for verification. "
                f"Use reject() instead."
            )

        return self.store.verify_sign(gloss, quality_score, verified_by)

    def reject(
        self,
        gloss: str,
        reason: str,
        rejected_by: str,
    ) -> Sign:
        """Reject a sign and move to rejected.

        Args:
            gloss: Sign gloss to reject
            reason: Reason for rejection
            rejected_by: Username of rejector

        Returns:
            Rejected Sign object
        """
        return self.store.reject_sign(gloss, reason)

    def batch_verify(
        self,
        glosses: list[str],
        quality_score: int,
        verified_by: str,
    ) -> tuple[list[Sign], list[tuple[str, str]]]:
        """Verify multiple signs at once.

        Args:
            glosses: List of sign glosses to verify
            quality_score: Quality score to assign to all
            verified_by: Username of verifier

        Returns:
            Tuple of (verified_signs, failed_list)
            where failed_list is [(gloss, error_message), ...]
        """
        verified = []
        failed = []

        for gloss in glosses:
            try:
                sign = self.verify(gloss, quality_score, verified_by)
                verified.append(sign)
            except Exception as e:
                failed.append((gloss, str(e)))

        return verified, failed

    def get_verification_queue(self) -> list[Sign]:
        """Get all signs awaiting verification.

        Returns pending signs first, then imported signs.
        """
        pending = self.store.list_pending()
        imported = self.store.list_signs(SignStatus.IMPORTED)
        return pending + imported

    def get_verification_stats(self) -> dict:
        """Get verification workflow statistics."""
        counts = self.store.count_signs()

        total_reviewed = counts["verified"] + counts["rejected"]
        if total_reviewed > 0:
            approval_rate = counts["verified"] / total_reviewed * 100
        else:
            approval_rate = 0.0

        return {
            "pending_review": counts["pending"] + counts["imported"],
            "verified": counts["verified"],
            "rejected": counts["rejected"],
            "approval_rate": round(approval_rate, 1),
        }
