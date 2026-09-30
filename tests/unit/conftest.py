"""Unit tests that need no database: override the DB fixtures from tests/conftest.py."""

import pytest


@pytest.fixture(autouse=True, scope="session")
def dj_config():
    return None


@pytest.fixture(autouse=True, scope="session")
def pipeline():
    return None
