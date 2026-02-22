"""Tests for SignSearch."""

import pytest
import tempfile
from pathlib import Path

from packages.database import SignStore, SignSearch, SignStatus


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
def search(store):
    """Create SignSearch with store."""
    return SignSearch(store)


@pytest.fixture
def populated_store(store, temp_signs_dir):
    """Create store with sample signs."""
    # Create sample videos
    for i, (gloss, english, category) in enumerate([
        ("HELLO", ["hello", "hi"], "greeting"),
        ("GOODBYE", ["goodbye", "bye"], "greeting"),
        ("THANK_YOU", ["thank you", "thanks"], "politeness"),
        ("PLEASE", ["please"], "politeness"),
        ("HELP", ["help"], "request"),
    ]):
        video = temp_signs_dir / f"video_{i}.mp4"
        video.write_bytes(b"fake video")
        store.add_sign(gloss=gloss, video_path=video, english=english, category=category)

    # Verify some signs
    store.verify_sign("HELLO", quality_score=5, verified_by="tester")
    store.verify_sign("GOODBYE", quality_score=4, verified_by="tester")
    store.verify_sign("THANK_YOU", quality_score=5, verified_by="tester")

    return store


class TestSignSearch:
    def test_search_all(self, search, populated_store):
        """Test searching all signs."""
        results = search.search()
        assert len(results) == 5

    def test_search_by_query_gloss(self, search, populated_store):
        """Test searching by gloss."""
        results = search.search(query="HELLO")
        assert len(results) == 1
        assert results[0].gloss == "HELLO"

    def test_search_by_query_english(self, search, populated_store):
        """Test searching by English word."""
        results = search.search(query="thanks")
        assert len(results) == 1
        assert results[0].gloss == "THANK_YOU"

    def test_search_by_query_partial(self, search, populated_store):
        """Test partial text matching."""
        results = search.search(query="bye")
        assert len(results) == 1
        assert results[0].gloss == "GOODBYE"

    def test_search_by_category(self, search, populated_store):
        """Test searching by category."""
        results = search.search(category="greeting")
        assert len(results) == 2
        glosses = {r.gloss for r in results}
        assert glosses == {"HELLO", "GOODBYE"}

    def test_search_by_min_quality(self, search, populated_store):
        """Test filtering by minimum quality."""
        results = search.search(min_quality=5, verified_only=True)
        assert len(results) == 2
        for r in results:
            assert r.quality_score >= 5

    def test_search_verified_only(self, search, populated_store):
        """Test searching verified signs only."""
        results = search.search(verified_only=True)
        assert len(results) == 3
        for r in results:
            assert r.status == SignStatus.VERIFIED

    def test_search_by_status(self, search, populated_store):
        """Test filtering by status."""
        results = search.search(status=SignStatus.PENDING)
        assert len(results) == 2

    def test_search_combined_filters(self, search, populated_store):
        """Test combining multiple filters."""
        results = search.search(
            category="greeting",
            min_quality=5,
            verified_only=True,
        )
        assert len(results) == 1
        assert results[0].gloss == "HELLO"

    def test_find_by_english(self, search, populated_store):
        """Test finding by exact English word."""
        results = search.find_by_english("hello")
        assert len(results) == 1
        assert results[0].gloss == "HELLO"

    def test_find_by_english_no_match(self, search, populated_store):
        """Test finding by English word with no match."""
        results = search.find_by_english("nonexistent")
        assert len(results) == 0

    def test_find_by_category(self, search, populated_store):
        """Test finding by category."""
        results = search.find_by_category("politeness")
        assert len(results) == 1
        assert results[0].gloss == "THANK_YOU"

    def test_get_categories(self, search, populated_store):
        """Test getting all categories."""
        categories = search.get_categories()
        assert "greeting" in categories
        assert "politeness" in categories

    def test_get_statistics(self, search, populated_store):
        """Test getting database statistics."""
        stats = search.get_statistics()

        assert stats["total_verified"] == 3
        assert stats["total_pending"] == 2
        assert "quality_distribution" in stats
        assert "categories" in stats
        assert stats["categories"]["greeting"] == 2
