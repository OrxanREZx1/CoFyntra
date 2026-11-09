"""Django foundation: health endpoint, DRF defaults, migrations in sync with models and the AI contract."""

from io import StringIO

import pytest
from django.core.management import call_command
from django.db import connection

from cofytra_ai.models import Confidence, EpistemicStatus, KnowledgeCategory
from core.models import SharedKnowledge

pytestmark = pytest.mark.django_db


def test_health_is_public_and_checks_database(client):
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_api_requires_authentication_by_default(client):
    from rest_framework.settings import api_settings

    assert [c.__name__ for c in api_settings.DEFAULT_PERMISSION_CLASSES] == ["IsAuthenticated"]


def test_migrations_are_in_sync_with_models():
    # Also catches drift if cofytra_ai changes a vocabulary or limit that the models import.
    call_command("makemigrations", "--check", "--dry-run", stdout=StringIO())


def test_vocabularies_match_the_ai_contract_exactly():
    fields = {f.name: f for f in SharedKnowledge._meta.get_fields() if hasattr(f, "choices")}
    assert [v for v, _ in fields["category"].choices] == [c.value for c in KnowledgeCategory]
    assert [v for v, _ in fields["epistemic_status"].choices] == [s.value for s in EpistemicStatus]
    assert [v for v, _ in fields["extraction_confidence"].choices] == [c.value for c in Confidence]


def test_reports_database_vendor(record_property):
    record_property("database_vendor", connection.vendor)
    assert connection.vendor in {"postgresql", "sqlite"}
