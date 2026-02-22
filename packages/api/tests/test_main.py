"""Tests for API main module."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_check_returns_healthy(self):
        """Test that health check returns healthy status."""
        from packages.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_check_response_format(self):
        """Test that health check response includes status, version, and services."""
        from packages.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        assert "status" in data
        assert "version" in data
        assert "services" in data

        # Verify types
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["services"], dict)

    def test_health_check_includes_all_services(self):
        """Test that health check includes api, translation, video, and database services."""
        from packages.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/health")

        assert response.status_code == 200
        services = response.json()["services"]

        # Verify all expected services are present
        assert "api" in services
        assert "translation" in services
        assert "video" in services
        assert "database" in services

        # Verify all services have status values
        assert services["api"] == "running"
        assert services["translation"] == "available"
        assert services["video"] == "available"
        assert services["database"] == "available"


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_info(self):
        """Test that root endpoint returns API information."""
        from packages.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "SignBridge" in data["message"]

    def test_root_includes_docs_link(self):
        """Test that root endpoint includes link to documentation."""
        from packages.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "docs" in data
        assert data["docs"] == "/docs"

    def test_root_includes_health_link(self):
        """Test that root endpoint includes link to health check."""
        from packages.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "health" in data
        assert data["health"] == "/api/health"


class TestAppConfiguration:
    """Tests for FastAPI application configuration."""

    def test_app_title_is_signbridge(self):
        """Test that app title is SignBridge API."""
        from packages.api.main import app

        assert app.title == "SignBridge API"

    def test_app_version_is_2_0_0(self):
        """Test that app version is 2.0.0."""
        from packages.api.main import app

        assert app.version == "2.0.0"

    def test_docs_url_is_correct(self):
        """Test that docs URL is /docs."""
        from packages.api.main import app

        assert app.docs_url == "/docs"

    def test_routers_are_included(self):
        """Test that translate, signs, and videos routers are included."""
        from packages.api.main import app

        # Get all route paths
        route_paths = [route.path for route in app.routes]

        # Check that routes from each router are present
        # Translate router should have /api/translate
        translate_routes = [p for p in route_paths if "/translate" in p]
        assert len(translate_routes) > 0, "Translate router routes not found"

        # Signs router should have /api/signs
        signs_routes = [p for p in route_paths if "/signs" in p]
        assert len(signs_routes) > 0, "Signs router routes not found"

        # Videos router should have /api/videos
        videos_routes = [p for p in route_paths if "/videos" in p]
        assert len(videos_routes) > 0, "Videos router routes not found"
