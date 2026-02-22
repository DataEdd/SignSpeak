"""Tests for sign service."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from packages.api.services.sign_service import SignService


class TestSignService:
    """Tests for SignService class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.mock_store = MagicMock()
        self.mock_store.list_signs.return_value = []

        with patch("packages.api.services.sign_service.SignSearch") as MockSearch, \
             patch("packages.api.services.sign_service.SignVerifier") as MockVerifier:
            self.mock_search_class = MockSearch
            self.mock_verifier_class = MockVerifier
            self.mock_search = MagicMock()
            self.mock_verifier = MagicMock()
            MockSearch.return_value = self.mock_search
            MockVerifier.return_value = self.mock_verifier
            self.service = SignService(sign_store=self.mock_store)

    def test_init_creates_search_and_verifier(self):
        """Test that init creates SignSearch and SignVerifier instances."""
        with patch("packages.api.services.sign_service.SignSearch") as MockSearch, \
             patch("packages.api.services.sign_service.SignVerifier") as MockVerifier:
            mock_store = MagicMock()
            service = SignService(sign_store=mock_store)

            MockSearch.assert_called_once_with(mock_store)
            MockVerifier.assert_called_once_with(mock_store)
            assert service.store == mock_store

    def test_list_signs_default(self):
        """Test that list_signs returns signs with default parameters."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "VERIFIED"
        mock_sign.quality_score = 5
        mock_sign.verified_by = "tester"
        mock_sign.verified_date = "2024-01-15"
        mock_sign.video = True

        self.mock_store.list_signs.return_value = [mock_sign]

        result = self.service.list_signs()

        self.mock_store.list_signs.assert_called_once_with(status=None)
        assert result["total"] == 1
        assert len(result["signs"]) == 1

    @patch("packages.api.services.sign_service.DBSignStatus")
    def test_list_signs_with_status(self, MockDBSignStatus):
        """Test that list_signs filters by status."""
        mock_status = MagicMock()
        MockDBSignStatus.return_value = mock_status

        self.mock_store.list_signs.return_value = []

        self.service.list_signs(status="verified")

        MockDBSignStatus.assert_called_once_with("VERIFIED")
        self.mock_store.list_signs.assert_called_once_with(status=mock_status)

    def test_list_signs_with_category(self):
        """Test that list_signs filters by category."""
        mock_sign1 = MagicMock()
        mock_sign1.gloss = "HELLO"
        mock_sign1.english = ["hello"]
        mock_sign1.category = "greeting"
        mock_sign1.source = "recorded"
        mock_sign1.status.value = "VERIFIED"
        mock_sign1.quality_score = 5
        mock_sign1.verified_by = "tester"
        mock_sign1.verified_date = "2024-01-15"
        mock_sign1.video = True

        mock_sign2 = MagicMock()
        mock_sign2.gloss = "CAT"
        mock_sign2.english = ["cat"]
        mock_sign2.category = "animal"
        mock_sign2.source = "recorded"
        mock_sign2.status.value = "VERIFIED"
        mock_sign2.quality_score = 5
        mock_sign2.verified_by = "tester"
        mock_sign2.verified_date = "2024-01-15"
        mock_sign2.video = True

        self.mock_store.list_signs.return_value = [mock_sign1, mock_sign2]

        result = self.service.list_signs(category="greeting")

        assert result["total"] == 1
        assert result["signs"][0]["gloss"] == "HELLO"

    def test_list_signs_pagination(self):
        """Test that list_signs applies pagination correctly."""
        mock_signs = []
        for i in range(10):
            mock_sign = MagicMock()
            mock_sign.gloss = f"SIGN{i}"
            mock_sign.english = [f"sign{i}"]
            mock_sign.category = "test"
            mock_sign.source = "recorded"
            mock_sign.status.value = "VERIFIED"
            mock_sign.quality_score = 5
            mock_sign.verified_by = "tester"
            mock_sign.verified_date = "2024-01-15"
            mock_sign.video = True
            mock_signs.append(mock_sign)

        self.mock_store.list_signs.return_value = mock_signs

        result = self.service.list_signs(limit=3, offset=2)

        assert result["total"] == 10
        assert len(result["signs"]) == 3
        assert result["signs"][0]["gloss"] == "SIGN2"
        assert result["signs"][2]["gloss"] == "SIGN4"

    def test_get_sign_exists(self):
        """Test that get_sign returns sign when it exists."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "VERIFIED"
        mock_sign.quality_score = 5
        mock_sign.verified_by = "tester"
        mock_sign.verified_date = "2024-01-15"
        mock_sign.video = True

        self.mock_store.get_sign.return_value = mock_sign

        result = self.service.get_sign("HELLO")

        self.mock_store.get_sign.assert_called_once_with("HELLO")
        assert result["gloss"] == "HELLO"

    def test_get_sign_not_found(self):
        """Test that get_sign returns None when sign does not exist."""
        self.mock_store.get_sign.return_value = None

        result = self.service.get_sign("UNKNOWN")

        self.mock_store.get_sign.assert_called_once_with("UNKNOWN")
        assert result is None

    def test_get_verified_sign(self):
        """Test that get_verified_sign returns only verified signs."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "VERIFIED"
        mock_sign.quality_score = 5
        mock_sign.verified_by = "tester"
        mock_sign.verified_date = "2024-01-15"
        mock_sign.video = True

        self.mock_store.get_verified_sign.return_value = mock_sign

        result = self.service.get_verified_sign("HELLO")

        self.mock_store.get_verified_sign.assert_called_once_with("HELLO")
        assert result["gloss"] == "HELLO"
        assert result["status"] == "verified"

    def test_create_sign_uppercase_gloss(self):
        """Test that create_sign converts gloss to uppercase."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "PENDING"
        mock_sign.quality_score = None
        mock_sign.verified_by = None
        mock_sign.verified_date = None
        mock_sign.video = True

        self.mock_store.add_sign.return_value = mock_sign

        self.service.create_sign(
            gloss="hello",
            video_path=Path("/tmp/hello.mp4"),
        )

        call_kwargs = self.mock_store.add_sign.call_args[1]
        assert call_kwargs["gloss"] == "HELLO"

    def test_create_sign_calls_store(self):
        """Test that create_sign calls store.add_sign with correct arguments."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello", "hi"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "PENDING"
        mock_sign.quality_score = None
        mock_sign.verified_by = None
        mock_sign.verified_date = None
        mock_sign.video = True

        self.mock_store.add_sign.return_value = mock_sign

        self.service.create_sign(
            gloss="HELLO",
            video_path=Path("/tmp/hello.mp4"),
            english=["hello", "hi"],
            category="greeting",
            source="recorded",
            metadata={"extra": "data"},
        )

        self.mock_store.add_sign.assert_called_once_with(
            gloss="HELLO",
            video_path="/tmp/hello.mp4",
            english=["hello", "hi"],
            category="greeting",
            source="recorded",
            metadata={"extra": "data"},
        )

    def test_verify_sign_success(self):
        """Test that verify_sign returns updated sign on success."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "VERIFIED"
        mock_sign.quality_score = 5
        mock_sign.verified_by = "expert"
        mock_sign.verified_date = "2024-01-15"
        mock_sign.video = True

        self.mock_store.verify_sign.return_value = mock_sign

        result = self.service.verify_sign(
            gloss="hello",
            quality_score=5,
            verified_by="expert",
        )

        self.mock_store.verify_sign.assert_called_once_with(
            gloss="HELLO",
            score=5,
            verified_by="expert",
        )
        assert result["status"] == "verified"
        assert result["quality_score"] == 5

    def test_verify_sign_handles_exception(self):
        """Test that verify_sign returns None when exception occurs."""
        self.mock_store.verify_sign.side_effect = Exception("Sign not found")

        result = self.service.verify_sign(
            gloss="UNKNOWN",
            quality_score=5,
            verified_by="expert",
        )

        assert result is None

    def test_reject_sign_success(self):
        """Test that reject_sign returns updated sign on success."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "REJECTED"
        mock_sign.quality_score = None
        mock_sign.verified_by = None
        mock_sign.verified_date = None
        mock_sign.video = True

        self.mock_store.move_sign.return_value = mock_sign

        result = self.service.reject_sign(gloss="hello", reason="poor quality")

        assert result["status"] == "rejected"

    def test_reject_sign_handles_exception(self):
        """Test that reject_sign returns None when exception occurs."""
        self.mock_store.move_sign.side_effect = Exception("Sign not found")

        result = self.service.reject_sign(gloss="UNKNOWN")

        assert result is None

    def test_delete_sign_success(self):
        """Test that delete_sign returns True on success."""
        self.mock_store.delete_sign.return_value = True

        result = self.service.delete_sign("hello")

        self.mock_store.delete_sign.assert_called_once_with("HELLO")
        assert result is True

    @patch("packages.api.services.sign_service.DBSignStatus")
    def test_search_signs(self, MockDBSignStatus):
        """Test that search_signs calls search with correct arguments."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "VERIFIED"
        mock_sign.quality_score = 5
        mock_sign.verified_by = "tester"
        mock_sign.verified_date = "2024-01-15"
        mock_sign.video = True

        mock_status = MagicMock()
        MockDBSignStatus.return_value = mock_status

        self.mock_search.search.return_value = [mock_sign]

        result = self.service.search_signs(
            query="hello",
            status="verified",
            limit=10,
        )

        self.mock_search.search.assert_called_once_with(
            query="hello",
            status=mock_status,
            limit=10,
        )
        assert len(result) == 1
        assert result[0]["gloss"] == "HELLO"

    def test_get_stats_counts_by_status(self):
        """Test that get_stats correctly counts signs by status."""
        mock_signs = []
        statuses = ["VERIFIED", "VERIFIED", "PENDING", "IMPORTED", "REJECTED"]
        for i, status in enumerate(statuses):
            mock_sign = MagicMock()
            mock_sign.gloss = f"SIGN{i}"
            mock_sign.status.value = status
            mock_sign.category = "test"
            mock_signs.append(mock_sign)

        self.mock_store.list_signs.return_value = mock_signs

        result = self.service.get_stats()

        assert result["total_signs"] == 5
        assert result["verified_signs"] == 2
        assert result["pending_signs"] == 1
        assert result["imported_signs"] == 1
        assert result["rejected_signs"] == 1

    def test_get_stats_counts_by_category(self):
        """Test that get_stats correctly counts signs by category."""
        mock_signs = []
        categories = ["greeting", "greeting", "animal", "animal", "animal"]
        for i, category in enumerate(categories):
            mock_sign = MagicMock()
            mock_sign.gloss = f"SIGN{i}"
            mock_sign.status.value = "VERIFIED"
            mock_sign.category = category
            mock_signs.append(mock_sign)

        self.mock_store.list_signs.return_value = mock_signs

        result = self.service.get_stats()

        assert result["categories"]["greeting"] == 2
        assert result["categories"]["animal"] == 3

    def test_sign_to_dict_converts_all_fields(self):
        """Test that _sign_to_dict converts all Sign fields correctly."""
        mock_sign = MagicMock()
        mock_sign.gloss = "HELLO"
        mock_sign.english = ["hello", "hi"]
        mock_sign.category = "greeting"
        mock_sign.source = "recorded"
        mock_sign.status.value = "VERIFIED"
        mock_sign.quality_score = 5
        mock_sign.verified_by = "expert"
        mock_sign.verified_date = "2024-01-15"
        mock_sign.video = True

        result = self.service._sign_to_dict(mock_sign)

        assert result["gloss"] == "HELLO"
        assert result["english"] == ["hello", "hi"]
        assert result["category"] == "greeting"
        assert result["source"] == "recorded"
        assert result["status"] == "verified"
        assert result["quality_score"] == 5
        assert result["verified_by"] == "expert"
        assert result["verified_date"] == "2024-01-15"
        assert result["video_url"] == "/api/signs/HELLO/video"
