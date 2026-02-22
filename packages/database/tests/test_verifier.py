"""Tests for SignVerifier."""

import pytest
import tempfile
from pathlib import Path

from packages.database import SignStore, SignVerifier, SignStatus


@pytest.fixture
def temp_signs_dir():
    """Create temporary signs directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_signs_dir):
    """Create SignStore with temp directory."""
    return SignStore(temp_signs_dir)


@pytest.fixture
def verifier(store):
    """Create SignVerifier with store."""
    return SignVerifier(store)


@pytest.fixture
def sample_video(temp_signs_dir):
    """Create a sample video file."""
    video_path = temp_signs_dir / "sample.mp4"
    video_path.write_bytes(b"fake video content")
    return video_path


class TestSignVerifier:
    def test_get_next_pending(self, store, verifier, sample_video):
        """Test getting next pending sign."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        sign = verifier.get_next_pending()
        assert sign is not None
        assert sign.gloss == "HELLO"

    def test_get_next_pending_empty(self, verifier):
        """Test getting next pending when queue is empty."""
        sign = verifier.get_next_pending()
        assert sign is None

    def test_check_sign_quality(self, store, verifier, sample_video):
        """Test automated quality check."""
        sign = store.add_sign(
            gloss="HELLO",
            video_path=sample_video,
            english=["hello"],
        )

        result = verifier.check_sign_quality(sign)

        assert result.passed is True
        assert result.score >= 3

    def test_check_sign_quality_missing_english(self, store, verifier, sample_video):
        """Test quality check catches missing English."""
        sign = store.add_sign(gloss="HELLO", video_path=sample_video)

        result = verifier.check_sign_quality(sign)

        assert "Missing English translations" in result.issues

    def test_verify(self, store, verifier, sample_video):
        """Test verifying a sign."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        sign = verifier.verify("HELLO", quality_score=5, verified_by="tester")

        assert sign.status == SignStatus.VERIFIED
        assert sign.quality_score == 5

    def test_verify_low_score_rejected(self, store, verifier, sample_video):
        """Test that low scores are rejected."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        with pytest.raises(ValueError, match="below minimum"):
            verifier.verify("HELLO", quality_score=2, verified_by="tester")

    def test_reject(self, store, verifier, sample_video):
        """Test rejecting a sign."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        sign = verifier.reject("HELLO", reason="Poor quality", rejected_by="tester")

        assert sign.status == SignStatus.REJECTED

    def test_batch_verify(self, store, verifier, temp_signs_dir):
        """Test batch verification."""
        for i, gloss in enumerate(["HELLO", "GOODBYE", "THANKS"]):
            video = temp_signs_dir / f"video_{i}.mp4"
            video.write_bytes(b"fake video")
            store.add_sign(gloss=gloss, video_path=video)

        verified, failed = verifier.batch_verify(
            ["HELLO", "GOODBYE", "NOTFOUND"],
            quality_score=5,
            verified_by="tester",
        )

        assert len(verified) == 2
        assert len(failed) == 1
        assert failed[0][0] == "NOTFOUND"

    def test_get_verification_queue(self, store, verifier, temp_signs_dir):
        """Test getting verification queue."""
        for i, gloss in enumerate(["HELLO", "GOODBYE"]):
            video = temp_signs_dir / f"video_{i}.mp4"
            video.write_bytes(b"fake video")
            store.add_sign(gloss=gloss, video_path=video)

        queue = verifier.get_verification_queue()

        assert len(queue) == 2

    def test_get_verification_stats(self, store, verifier, temp_signs_dir):
        """Test verification statistics."""
        for i, gloss in enumerate(["HELLO", "GOODBYE", "THANKS"]):
            video = temp_signs_dir / f"video_{i}.mp4"
            video.write_bytes(b"fake video")
            store.add_sign(gloss=gloss, video_path=video)

        store.verify_sign("HELLO", quality_score=5, verified_by="tester")
        store.reject_sign("GOODBYE", reason="poor")

        stats = verifier.get_verification_stats()

        assert stats["pending_review"] == 1
        assert stats["verified"] == 1
        assert stats["rejected"] == 1
        assert stats["approval_rate"] == 50.0
