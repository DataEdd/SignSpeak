"""Tests for SignStore."""

import json
import pytest
import tempfile
from pathlib import Path

from packages.database import SignStore, Sign, SignStatus


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
def sample_video(temp_signs_dir):
    """Create a sample video file."""
    video_path = temp_signs_dir / "sample.mp4"
    video_path.write_bytes(b"fake video content")
    return video_path


class TestSignStore:
    def test_init_creates_directories(self, temp_signs_dir):
        """Test that init creates required directories."""
        store = SignStore(temp_signs_dir)

        assert (temp_signs_dir / "verified").exists()
        assert (temp_signs_dir / "pending").exists()
        assert (temp_signs_dir / "imported").exists()
        assert (temp_signs_dir / "rejected").exists()

    def test_add_sign(self, store, sample_video):
        """Test adding a new sign."""
        sign = store.add_sign(
            gloss="HELLO",
            video_path=sample_video,
            english=["hello", "hi"],
            category="greeting",
        )

        assert sign.gloss == "HELLO"
        assert sign.english == ["hello", "hi"]
        assert sign.category == "greeting"
        assert sign.status == SignStatus.PENDING
        assert sign.path.exists()
        assert (sign.path / "video.mp4").exists()
        assert (sign.path / "sign.json").exists()

    def test_add_sign_normalizes_gloss(self, store, sample_video):
        """Test that gloss is normalized to uppercase."""
        sign = store.add_sign(gloss="hello", video_path=sample_video)
        assert sign.gloss == "HELLO"

    def test_add_sign_rejects_duplicates(self, store, sample_video):
        """Test that duplicate signs are rejected."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        with pytest.raises(ValueError, match="already exists"):
            store.add_sign(gloss="HELLO", video_path=sample_video)

    def test_add_sign_rejects_missing_video(self, store, temp_signs_dir):
        """Test that missing video file raises error."""
        with pytest.raises(FileNotFoundError):
            store.add_sign(gloss="HELLO", video_path=temp_signs_dir / "missing.mp4")

    def test_get_sign(self, store, sample_video):
        """Test getting a sign by gloss."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        sign = store.get_sign("HELLO")
        assert sign is not None
        assert sign.gloss == "HELLO"

    def test_get_sign_case_insensitive(self, store, sample_video):
        """Test that get_sign is case insensitive."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        sign = store.get_sign("hello")
        assert sign is not None
        assert sign.gloss == "HELLO"

    def test_get_sign_not_found(self, store):
        """Test getting non-existent sign returns None."""
        sign = store.get_sign("NOTFOUND")
        assert sign is None

    def test_verify_sign(self, store, sample_video):
        """Test verifying a sign moves it to verified."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        sign = store.verify_sign("HELLO", quality_score=5, verified_by="tester")

        assert sign.status == SignStatus.VERIFIED
        assert sign.quality_score == 5
        assert sign.verified_by == "tester"
        assert sign.verified_date is not None
        assert "verified" in str(sign.path)

    def test_verify_sign_invalid_score(self, store, sample_video):
        """Test that invalid quality scores are rejected."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        with pytest.raises(ValueError, match="must be 1-5"):
            store.verify_sign("HELLO", quality_score=6, verified_by="tester")

    def test_verify_sign_not_found(self, store):
        """Test verifying non-existent sign raises error."""
        with pytest.raises(ValueError, match="not found"):
            store.verify_sign("NOTFOUND", quality_score=5, verified_by="tester")

    def test_reject_sign(self, store, sample_video):
        """Test rejecting a sign moves it to rejected."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        sign = store.reject_sign("HELLO", reason="Poor quality")

        assert sign.status == SignStatus.REJECTED
        assert "rejected" in str(sign.path)

    def test_delete_sign(self, store, sample_video):
        """Test deleting a sign removes it completely."""
        sign = store.add_sign(gloss="HELLO", video_path=sample_video)
        sign_path = sign.path

        result = store.delete_sign("HELLO")

        assert result is True
        assert not sign_path.exists()
        assert store.get_sign("HELLO") is None

    def test_delete_sign_not_found(self, store):
        """Test deleting non-existent sign returns False."""
        result = store.delete_sign("NOTFOUND")
        assert result is False

    def test_list_signs(self, store, sample_video):
        """Test listing all signs."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        # Create another video for second sign
        video2 = sample_video.parent / "sample2.mp4"
        video2.write_bytes(b"fake video 2")
        store.add_sign(gloss="GOODBYE", video_path=video2)

        signs = store.list_signs()
        assert len(signs) == 2
        glosses = {s.gloss for s in signs}
        assert glosses == {"HELLO", "GOODBYE"}

    def test_list_signs_by_status(self, store, sample_video):
        """Test filtering signs by status."""
        store.add_sign(gloss="HELLO", video_path=sample_video)
        store.verify_sign("HELLO", quality_score=5, verified_by="tester")

        video2 = sample_video.parent / "sample2.mp4"
        video2.write_bytes(b"fake video 2")
        store.add_sign(gloss="GOODBYE", video_path=video2)

        pending = store.list_signs(SignStatus.PENDING)
        verified = store.list_signs(SignStatus.VERIFIED)

        assert len(pending) == 1
        assert pending[0].gloss == "GOODBYE"
        assert len(verified) == 1
        assert verified[0].gloss == "HELLO"

    def test_list_pending(self, store, sample_video):
        """Test listing pending signs."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0].gloss == "HELLO"

    def test_list_verified(self, store, sample_video):
        """Test listing verified signs."""
        store.add_sign(gloss="HELLO", video_path=sample_video)
        store.verify_sign("HELLO", quality_score=5, verified_by="tester")

        verified = store.list_verified()
        assert len(verified) == 1
        assert verified[0].gloss == "HELLO"

    def test_count_signs(self, store, sample_video):
        """Test counting signs by status."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        counts = store.count_signs()

        assert counts["pending"] == 1
        assert counts["verified"] == 0
        assert counts["imported"] == 0
        assert counts["rejected"] == 0

    def test_get_verified_sign(self, store, sample_video):
        """Test getting only verified signs."""
        store.add_sign(gloss="HELLO", video_path=sample_video)

        # Should return None for pending sign
        sign = store.get_verified_sign("HELLO")
        assert sign is None

        # Verify the sign
        store.verify_sign("HELLO", quality_score=5, verified_by="tester")

        # Now should return the sign
        sign = store.get_verified_sign("HELLO")
        assert sign is not None
        assert sign.gloss == "HELLO"
